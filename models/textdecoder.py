import torch
import torch.nn as nn
from torch import Tensor

from typing import Dict, List, Optional, Tuple, Union

from models.denseclip import DenseCLIPContextDecoder
from models.classdecoder import ClassDecoder


class TextDecoder(nn.Module):
    def __init__(self, visual_dim, text_dim, use_classes, predict_classes, return_keys, return_queries=True, out_dim=256):
        super().__init__()
        assert return_keys or return_queries

        if use_classes or predict_classes:
            self.missing_emb = nn.Parameter(torch.randn(19, text_dim)) # missing classes place-holder embeddings

            self.class_decoder = ClassDecoder()

            # self.context_proj = nn.Linear(text_dim, text_dim, bias=False)
            # self.context_proj.apply(self._init_weights)
            # self.class_proj = nn.Linear(text_dim, text_dim, bias=False)
            # self.class_proj.apply(self._init_weights)

            # self.logit_scale = nn.Parameter(torch.tensor(1e-7))
            # # self.ignore_index = -100
            # self.contrastive_loss = nn.CrossEntropyLoss()#ignore_index=self.ignore_index)

            self.squared_error_loss = nn.MSELoss(reduction='none')
        
        self.text_proj = nn.Parameter(torch.randn(text_dim, text_dim)) 
        
        scale = visual_dim ** -0.5
        self.visual_proj = nn.Parameter(torch.randn(visual_dim, text_dim) * scale)        

        self.context_decoder = DenseCLIPContextDecoder(transformer_width=256,
                                                        transformer_heads=4,
                                                        transformer_layers=9,
                                                        visual_dim=text_dim,
                                                        dropout=0.1)

        nn.init.trunc_normal_(self.context_decoder.gamma, std=.02)

        if predict_classes:
            self.missing_class_predictor = nn.Linear(text_dim, 1)
            nn.init.trunc_normal_(self.missing_class_predictor.weight, std=.01)

            self.missing_class_criterion = nn.BCEWithLogitsLoss()

        self.predict = predict_classes
        self.oracle = use_classes and not predict_classes

        if return_keys:
            self.keys_proj = nn.Linear(text_dim, out_dim)
            self.keys_proj.apply(self._init_weights)
        
        self.return_keys = return_keys
        
        if return_queries:
            self.queries_proj = nn.Linear(text_dim, out_dim)
            self.queries_proj.apply(self._init_weights)
        
        self.return_queries = return_queries
        
    
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

    
    def reconstruction_loss(self, pred, gt):
        return self.squared_error_loss(pred, gt).sum(dim=-1).sqrt().mean()


    def forward(self, text: Tensor, visual: Tensor, classes: List = None):
        if 1:
            return self.forward__(text, visual, classes)
        elif self.oracle:
            return self.forward_oracle(text, visual, classes)
        else:
            return self.forward_(text, visual)


    def forward_(self, text: Tensor, visual: Tensor):
        text = text.expand(visual.shape[0],-1,-1)

        text_emb = text @ self.text_proj
        visual_emb = visual @ self.visual_proj

        contextualized_text = self.context_decoder(text=text_emb, visual=visual_emb)

        keys = self.keys_proj(text) if self.return_keys else None          
        queries = self.queries_proj(contextualized_text) if self.return_queries else None  

        return {"keys":keys, "queries":queries}
    

    def forward__(self, text: Tensor, visual: Tensor, classes: List):
        batch_size = visual.shape[0]

        text = text.expand(batch_size,-1,-1)

        text_emb = text @ self.text_proj
        visual_emb = visual @ self.visual_proj

        contextualized_text = self.context_decoder(text=text_emb, visual=visual_emb)
        
        missing_class_emb = self.missing_emb.expand(batch_size,-1,-1)
        memory = torch.cat([contextualized_text, missing_class_emb], dim=1)

        class_emb = self.class_decoder(memory)

        batch_indices = torch.cat([torch.full((19,),i, dtype=torch.long, device=class_emb.device) for i in range(batch_size)])
        gt_class_indices = torch.arange(start=19, end=38, dtype=torch.long, device=class_emb.device).expand(batch_size,-1)
        for i in range(batch_size):
            gt_class_indices[i, classes[i]] = classes[i]
        gt_class_indices = gt_class_indices.flatten()
        gt_class_emb = memory.detach()[batch_indices, gt_class_indices, :].reshape(batch_size,19,-1)

        loss = self.reconstruction_loss(class_emb, gt_class_emb)
        
        keys = self.keys_proj(text) if self.return_keys else None          
        queries = self.queries_proj(class_emb) if self.return_queries else None  

        return {"loss":loss, "keys":keys, "queries":queries}


    def forward_cl(self, text: Tensor, visual: Tensor, classes: List):
        batch_size = visual.shape[0]

        text = torch.cat([text, self.missing_emb]).expand(batch_size,-1,-1)

        # text = text.expand(batch_size,-1,-1)

        text_emb = text @ self.text_proj
        visual_emb = visual @ self.visual_proj

        contextualized_text = self.context_decoder(text=text_emb, visual=visual_emb)

        class_emb = self.class_decoder(contextualized_text)

        context_proj = self.context_proj(contextualized_text)
        class_proj = self.class_proj(class_emb)

        # normalized features
        context_embeds = context_proj / torch.norm(context_proj, p=2, dim=-1, keepdim=True)
        class_embeds = class_proj / torch.norm(class_proj, p=2, dim=-1, keepdim=True)

        # cosine similarity logits
        logits_per_class = torch.matmul(class_embeds, context_embeds.transpose(-1,-2)) * self.logit_scale.exp()

        gt_class_indices = torch.arange(start=19, end=38, dtype=torch.long, device=logits_per_class.device).expand(batch_size,-1)
        # gt_context_indices = torch.full((batch_size,38), self.ignore_index, dtype=torch.long, device=logits_per_class.device)
        for i in range(batch_size):
            gt_class_indices[i, classes[i]] = classes[i]
            # gt_context_indices[i, gt_class_indices[i]] = torch.arange(19, dtype=torch.long, device=logits_per_class.device)
        
        loss = self.contrastive_loss(logits_per_class.transpose(-1,-2), gt_class_indices) #+ self.contrastive_loss(logits_per_class, gt_context_indices)) / 2

        batch_indices = torch.cat([torch.full((19,),i) for i in range(batch_size)])
        pred_indices = logits_per_class.argmax(dim=-1)
        pred_class = contextualized_text[batch_indices,pred_indices.flatten(),:].reshape(batch_size,19,-1)

        acc = torch.mean((pred_indices == gt_class_indices).float())
        
        keys = self.keys_proj(text) if self.return_keys else None          
        queries = self.queries_proj(pred_class) if self.return_queries else None  

        return {"loss":loss, "acc":acc, "keys":keys, "queries":queries}
    
    
    def forward_predict(self, text: Tensor, visual: Tensor, classes: List):
        text = text.expand(visual.shape[0],-1,-1)

        text_emb = text @ self.text_proj
        visual_emb = visual @ self.visual_proj

        contextualized_text = self.context_decoder(text=text_emb, visual=visual_emb)

        gt_missing_classes = (torch.stack([torch.bincount(x, minlength=19) for x in classes]) == 0).float()
        pred = self.missing_class_predictor(contextualized_text).squeeze()
        missing_classes = pred.sigmoid()
        
        loss = self.missing_class_criterion(pred, gt_missing_classes)
        
        contextualized_text = contextualized_text.clone()
        contextualized_text[missing_classes > 0.5] = 0

        missing_emb = self.missing_emb.expand(visual.shape[0],-1,-1)
        contextualized_text[contextualized_text == 0] += missing_emb[contextualized_text == 0]

        acc = torch.mean(((missing_classes > 0.5) == gt_missing_classes).float())
        
        keys = self.keys_proj(text) if self.return_keys else None          
        queries = self.queries_proj(contextualized_text) if self.return_queries else None  

        return {"loss":loss, "acc":acc, "keys":keys, "queries":queries}
    

    def forward_oracle(self, text: Tensor, visual: Tensor, classes: List):
        text = text.repeat(visual.shape[0],1,1)

        missing_classes = torch.stack([torch.bincount(x, minlength=19) for x in classes]) == 0
        text[missing_classes] = 0

        missing_emb = self.missing_emb.expand(visual.shape[0],-1,-1)
        text[text == 0] += missing_emb[text == 0]

        text_emb = text @ self.text_proj
        visual_emb = visual @ self.visual_proj

        contextualized_text = self.context_decoder(text=text_emb, visual=visual_emb)

        queries = self.queries_proj(contextualized_text) if self.return_queries else None  

        return {"keys":None, "queries":queries}
    