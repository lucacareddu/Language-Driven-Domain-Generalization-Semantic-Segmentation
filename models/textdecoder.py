import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt

from models.denseclip import DenseCLIPContextDecoder


class TextDecoder(nn.Module):
    def __init__(self, visual_dim, text_dim, return_keys, return_queries=True, out_dim=256):
        super().__init__()
        assert return_keys or return_queries

        self.class_emb = nn.Parameter(torch.randn(19, text_dim)) # if text is not passed

        # if proj        
        self.text_proj = nn.Linear(text_dim, text_dim, bias=False)
        self.text_proj.apply(self._init_weights)
        self.visual_proj = nn.Linear(visual_dim, text_dim, bias=False)      
        self.visual_proj.apply(self._init_weights)

        self.context_decoder = DenseCLIPContextDecoder(transformer_width=256,
                                                        transformer_heads=4,
                                                        transformer_layers=9,
                                                        visual_dim=text_dim,
                                                        dropout=0.1)

        nn.init.trunc_normal_(self.context_decoder.gamma, std=.02)

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


    def forward(self, text=None, visual=None, proj=True, compute_score_map=True):
        assert visual is not None

        B, N, _ = visual[:,1:].shape

        text = (text if text is not None else self.class_emb).expand(B,-1,-1)

        text_emb = self.text_proj(text) if proj else text
        visual_emb = self.visual_proj(visual) if proj else visual

        contextualized_text = self.context_decoder(text=text_emb, visual=visual_emb)

        score_map = None
        if compute_score_map:
            pixel_emb = visual_emb[:,1:].permute(0,2,1).reshape(B, -1, int(sqrt(N)), int(sqrt(N)))
            pixel_vectors = F.normalize(pixel_emb, dim=1, p=2)
            text_vectors = F.normalize(contextualized_text, dim=-1, p=2)
            score_map = torch.einsum('bchw,bkc->bkhw', pixel_vectors, text_vectors)

        keys = self.keys_proj(text) if self.return_keys else None          
        queries = self.queries_proj(contextualized_text) if self.return_queries else None

        return {"score_map":score_map, "keys":keys, "queries":queries}
    