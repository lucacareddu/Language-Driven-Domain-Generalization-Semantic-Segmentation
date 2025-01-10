import torch
import torch.nn as nn
from torch import Tensor

from typing import Dict, List, Optional, Tuple, Union


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.weights_proj = nn.Linear(dim, 38, bias=qkv_bias)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v, weights_from_q=False, clip_weights=False, mask=None):
        B, N, C = q.shape
        assert k.shape == v.shape
        B, M, C = k.shape
        
        attn_ = None
        if weights_from_q:
            attn_ = self.weights_proj(q) #* 10000#* (38 ** -0.5)
            attn_ = attn_.softmax(dim=-1)

            if clip_weights:
                pass
                # print(attn_[attn_.max(dim=-1, keepdim=True)[1]])

            attn = attn_.unsqueeze(1).expand(-1,self.num_heads,-1,-1)
        else:
            q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads)
            k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads)

            attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale

            attn = attn.softmax(dim=-1)

        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads)

        if mask is not None:
            attn = attn * mask.unsqueeze(1).expand(-1,self.num_heads,-1,-1)

        x = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.1,
    ):
        super().__init__()
        self.self_attn = Attention(d_model, nhead, proj_drop=dropout)
        self.cross_attn = Attention(d_model, nhead, proj_drop=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, mem, weights_from_q=True, clip_weights=False, crssattn_mask=None):
        q = self.norm2(x)
        hidden_states, attn = self.cross_attn(q, mem, mem, weights_from_q=weights_from_q, clip_weights=clip_weights, mask=crssattn_mask)
        x = x + hidden_states
        q = k = v = self.norm1(x)
        x = x + self.self_attn(q, k, v)[0]
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x, attn
    

class ClassDecoder(nn.Module):
    def __init__(self,
                 transformer_width=256,
                 transformer_heads=4,
                 transformer_layers=6,
                 dropout=0,
                 out_dim=256):
        super().__init__()

        self.missing_class_emb = nn.Parameter(torch.empty(19, transformer_width))
        nn.init.trunc_normal_(self.missing_class_emb, std=.02)

        # self.missing_class_pos = nn.Parameter(torch.empty(19, transformer_width))
        # nn.init.trunc_normal_(self.missing_class_pos, std=.01)

        self.memory_proj = nn.Sequential(
            nn.LayerNorm(transformer_width),
            nn.Linear(transformer_width, transformer_width)
        )

        self.text_proj = nn.Sequential(
            nn.LayerNorm(transformer_width),
            nn.Linear(transformer_width, transformer_width),
        )

        self.decoder = nn.ModuleList([
                    TransformerDecoderLayer(transformer_width, transformer_heads, dropout) for _ in range(transformer_layers)
                ])
        
        self.out_proj = nn.Sequential(
            nn.LayerNorm(transformer_width),
            nn.Linear(transformer_width, out_dim)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, class_embeds: Tensor, classes: List):
        batch_size = class_embeds.shape[0]

        mem = self.memory_proj(torch.cat([class_embeds, self.missing_class_emb.expand(batch_size,-1,-1)], dim=1))
        x = self.text_proj(class_embeds)

        class_bins = torch.stack([torch.bincount(c, minlength=19) for c in classes])
        gt_crssattn = torch.zeros((batch_size, 19, 38), dtype=torch.long, device=class_embeds.device)
        gt_crssattn[:,torch.arange(19),torch.arange(19)] = class_bins
        gt_crssattn[:,torch.arange(19),torch.arange(19,38)] = (class_bins == 0).long()

        loss = 0
        acc = 0

        for layer in self.decoder:
            x, pred_crssattn = layer(x, mem)
            loss += nn.functional.cross_entropy(pred_crssattn, gt_crssattn.float())
            acc += (pred_crssattn.argmax(dim=-1) == gt_crssattn.argmax(dim=-1)).sum() / (batch_size * 19.0)
        
        return loss, self.out_proj(x), acc/len(self.decoder)
    