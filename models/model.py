import torch
from torch import nn

from transformers import ViTModel
from transformers import CLIPModel, CLIPProcessor
from models.neck import ViTNeck, tqdmNeck
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from models.textdecoder import TextDecoder

from utils.metric import get_confBins, get_metrics

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DGSSModel(nn.Module):
    def __init__(self, encoder_name, ignore_value, text_prompts=None, nclasses=19, freeze_vision_encoder=False, freeze_text_encoder=True, no_neck=True, depthwise_neck=False, tqdm_neck=False, shallow_m2f=False, use_classes=False, predict_classes=False, use_text_keys=False, use_text_queries=True, nqueries=100):
        super().__init__()

        self.encoder_name = encoder_name

        encoder_config = {"vit":"google/vit-base-patch16-224-in21k",
                          "tiny_clip":"wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M",
                          "clip":"openai/clip-vit-base-patch16"}[encoder_name]
        
        encoder_visual_dim = {"vit":768, "tiny_clip":256, "clip":768}[encoder_name]
        encoder_text_dim = {"vit":512, "tiny_clip":256, "clip":512}[encoder_name]    

        self.freeze_vision_encoder = freeze_vision_encoder

        self.has_text_decoder = text_prompts is not None
        self.freeze_text_encoder = freeze_text_encoder 

        self.encoder = {"vit":ViTModel, "tiny_clip":CLIPModel, "clip":CLIPModel}[encoder_name].from_pretrained(encoder_config)

        if self.encoder_name == "vit" and self.has_text_decoder:
            encoder_config = "openai/clip-vit-base-patch16"
            self.vit_text_encoder = CLIPModel.from_pretrained(encoder_config)

        self.out_indices = {"vit":[3, 5, 7, 11], "tiny_clip":[4, 7, 10], "clip":[3, 5, 7, 11]}[encoder_name][-3:]

        if not tqdm_neck:
            self.neck = ViTNeck(in_channels=[encoder_visual_dim] * 3, out_channels=encoder_visual_dim, depthwise=depthwise_neck, no_neck=no_neck)
        else:
            self.neck = tqdmNeck(width=encoder_visual_dim)

        if shallow_m2f:
            vision_decoder_config = Mask2FormerConfig(num_labels=nclasses, ignore_value=ignore_value, feature_channels=[encoder_visual_dim] * 3, encoder_layers=1, decoder_layers=3, num_queries=(nclasses if self.has_text_decoder else nqueries))
        else:
            vision_decoder_config = Mask2FormerConfig(num_labels=nclasses, ignore_value=ignore_value, feature_channels=[encoder_visual_dim] * 3, num_queries=(nclasses if self.has_text_decoder else nqueries))
        
        self.vision_decoder = Mask2FormerForUniversalSegmentation(vision_decoder_config)
        self.vision_decoder_processor = Mask2FormerImageProcessor() 

        if self.has_text_decoder:
            tokenizer = CLIPProcessor.from_pretrained(encoder_config)
            text_tokenized = tokenizer(text=text_prompts, return_tensors="pt", padding=True)
            self.text_ids = text_tokenized["input_ids"].cuda()
            self.text_att = text_tokenized["attention_mask"].cuda()

            self.text_decoder = TextDecoder(visual_dim=encoder_visual_dim, text_dim=encoder_text_dim, use_classes=use_classes, predict_classes=predict_classes, return_keys=use_text_keys, return_queries=use_text_queries)

            self.predict_classes = predict_classes

            if use_text_keys:
                self.vision_decoder.model.pixel_level_module.decoder.encoder.crss_attn = nn.ModuleList([nn.MultiheadAttention(embed_dim=vision_decoder_config.hidden_dim, 
                                                                                                                             num_heads=vision_decoder_config.num_attention_heads, 
                                                                                                                             batch_first=True) for _ in range(vision_decoder_config.encoder_layers)])
                self.vision_decoder.model.pixel_level_module.decoder.encoder.crss_attn.apply(self._init_weights)

                self.vision_decoder.model.pixel_level_module.decoder.encoder.text_keys_pos = nn.Embedding(nclasses, vision_decoder_config.hidden_dim)
                self.vision_decoder.model.pixel_level_module.decoder.encoder.text_keys_pos.apply(self._init_weights)       

            if use_text_queries:
                del self.vision_decoder.model.transformer_module.queries_features   

        self.ignore_value = ignore_value      


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=.02)
        elif isinstance(m, nn.LayerNorm):            
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
    
    
    def train(self, mode: bool = True):
        super().train(mode)

        if mode and self.freeze_vision_encoder:
            self.encoder.train(False)
            for param in self.encoder.parameters():
                param.requires_grad = False

        elif mode and self.freeze_text_encoder:
            model = self.encoder if "clip" in self.encoder_name else self.vit_text_encoder
            model.text_model.train(False)
            for param in model.text_model.parameters():
                param.requires_grad = False


    def forward(self, pixel_values, bin_masks, classes, labels, return_logits=False):
        if self.encoder_name == "vit":      
            vision_outputs = self.encoder(pixel_values=pixel_values, output_hidden_states=True, interpolate_pos_encoding=True)
        else:
            vision_outputs = self.encoder.get_image_features(pixel_values=pixel_values, output_hidden_states=True, interpolate_pos_encoding=True)
        
        vision_hidden_states = vision_outputs["hidden_states"]
        vision_hidden_states = [h for i,h in enumerate(vision_hidden_states) if i in self.out_indices]

        if self.has_text_decoder:
            if self.encoder_name == "vit":  
                text_outputs = self.vit_text_encoder.get_text_features(input_ids=self.text_ids, attention_mask=self.text_att)
            else:
                text_outputs = self.encoder.get_text_features(input_ids=self.text_ids, attention_mask=self.text_att)

            outs = self.text_decoder(text=text_outputs, visual=vision_hidden_states[-1], classes=classes)

            small_labels = torch.nn.functional.interpolate(labels.unsqueeze(1).float(), size=outs["score_map"].shape[-2:], mode="nearest").squeeze(1).long()
            denseclip_loss = torch.nn.functional.cross_entropy(outs["score_map"], small_labels, ignore_index=self.ignore_value)
            jaccard, accuracy = get_metrics(get_confBins(predictions=outs["score_map"].argmax(dim=1), references=small_labels, ignore_index=self.ignore_value))
            miou, macc = torch.nanmean(jaccard).item(), torch.nanmean(accuracy).item()

            if outs["keys"] is not None:
                # To cross-attention layers in pixel decoder (mask2former encoder) as key-value
                self.vision_decoder.model.pixel_level_module.decoder.encoder.text_keys = outs["keys"]
            
            if outs["queries"] is not None:
                # To cross-attention layers in transformer decoder layers (mask2former decoder) as queries
                self.vision_decoder.model.transformer_module.text_queries = outs["queries"]

        multi_scale_feats = self.neck(vision_hidden_states)

        decoder_outputs = self.vision_decoder(pixel_values=multi_scale_feats, mask_labels=bin_masks, class_labels=classes)

        output = {"loss":decoder_outputs.loss}
        
        if self.has_text_decoder and self.predict_classes:
            output["loss"] += denseclip_loss
            output.update({"aux_loss":denseclip_loss, "aux_miou":miou, "aux_macc":macc})
        
        if return_logits:
            upsampled_logits = self.vision_decoder_processor.post_process_semantic_segmentation(decoder_outputs, target_sizes=[pixel_values.shape[-2:]] * pixel_values.shape[0])
            upsampled_logits = torch.stack(upsampled_logits)
            output["upsampled_logits"] = upsampled_logits
        
        return output
    

    def print_trainable_params(self, round_to_millions=True, decimals=2):
        self.train()

        if self.encoder_name == "vit":
            trainable_params = {"TOTAL": sum(p.numel() for p in self.parameters() if p.requires_grad),
                                "VIT": sum(p.numel() for p in self.encoder.parameters() if p.requires_grad),
                                "NECK": sum(p.numel() for p in self.neck.parameters() if p.requires_grad),
                                "MASK2FORMER": sum(p.numel() for p in self.vision_decoder.parameters() if p.requires_grad)}
                                
            if self.has_text_decoder:
                trainable_params.update({
                                "CLIP_TEXT": sum(p.numel() for p in self.vit_text_encoder.text_model.parameters() if p.requires_grad),
                                "TEXT_DECODER": sum(p.numel() for p in self.text_decoder.parameters() if p.requires_grad)})
        else:
            trainable_params = {"TOTAL": sum(p.numel() for p in self.parameters() if p.requires_grad),
                                "CLIP": sum(p.numel() for p in self.encoder.parameters() if p.requires_grad),
                                "CLIP_VISION": sum(p.numel() for p in self.encoder.vision_model.parameters() if p.requires_grad),
                                "CLIP_TEXT": sum(p.numel() for p in self.encoder.text_model.parameters() if p.requires_grad),
                                "NECK": sum(p.numel() for p in self.neck.parameters() if p.requires_grad),
                                "MASK2FORMER": sum(p.numel() for p in self.vision_decoder.parameters() if p.requires_grad)}
            
            if self.has_text_decoder:
                trainable_params.update({
                                "TEXT_DECODER": sum(p.numel() for p in self.text_decoder.parameters() if p.requires_grad)})
        
        if round_to_millions:
            trainable_params = {k:round((v/1e6), decimals) for k,v in trainable_params.items()}

        print("TRAINABLE PARAMS (M):")
        [print(f"   {k}: {v}") for k,v in trainable_params.items()]
        print()


    def print_frozen_modules(self):
        self.train()

        if self.encoder_name == "vit":
            trainable_modules = {"MODEL": self.training,
                                "VIT": self.encoder.training,
                                "NECK": self.neck.training,
                                "MASK2FORMER": self.vision_decoder.training}
            
            if self.has_text_decoder:
                trainable_modules.update({
                                "CLIP_TEXT": self.vit_text_encoder.text_model.training,
                                "TEXT_DECODER": self.text_decoder.training})
        else:
            trainable_modules = {"MODEL": self.training,
                                "CLIP": self.encoder.training,
                                "CLIP_VISION": self.encoder.vision_model.training,
                                "CLIP_TEXT": self.encoder.text_model.training,
                                "NECK": self.neck.training,
                                "MASK2FORMER": self.vision_decoder.training}
            
            if self.has_text_decoder:
                trainable_modules.update({
                                "TEXT_DECODER": self.text_decoder.training})

        print("IS FROZEN:")
        [print(f"   {k}: {not v}") for k,v in trainable_modules.items()]
        print()
