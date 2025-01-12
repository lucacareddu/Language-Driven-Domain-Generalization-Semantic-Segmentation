import torch
import torch.nn as nn

from models.denseclip import Attention


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

    def forward(self, x, mem):
        q = self.norm2(x)
        x = x + self.cross_attn(q, mem, mem)
        q = k = v = self.norm1(x)
        x = x + self.self_attn(q, k, v)
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x
    

class ClassDecoder(nn.Module):
    def __init__(self,
                 transformer_width=256,
                 transformer_heads=4,
                 transformer_layers=3,
                 dropout=0,
                 out_dim=256):
        super().__init__()

        self.class_emb = nn.Parameter(torch.empty(19, transformer_width))
        nn.init.trunc_normal_(self.class_emb, std=.02)

        self.class_pos = nn.Parameter(torch.randn(19, transformer_width))
        nn.init.trunc_normal_(self.class_pos, std=.01)

        self.memory_proj = nn.Sequential(
            nn.LayerNorm(transformer_width),
            nn.Linear(transformer_width, transformer_width, bias=False)
        )

        self.text_proj = nn.Sequential(
            nn.LayerNorm(transformer_width),
            nn.Linear(transformer_width, transformer_width, bias=False),
        )

        self.decoder = nn.ModuleList([
                    TransformerDecoderLayer(transformer_width, transformer_heads, dropout) for _ in range(transformer_layers)
                ])
        
        self.out_proj = nn.Sequential(
            nn.LayerNorm(transformer_width),
            nn.Linear(transformer_width, out_dim, bias=False)
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

    def forward(self, context_embeds):
        batch_size = context_embeds.shape[0]
        
        mem = self.memory_proj(context_embeds)
        x = self.text_proj(self.class_emb).expand(batch_size,-1,-1)

        if 0:
            mem = mem + self.class_pos.repeat(batch_size,2,1)
            x = x + self.class_pos.expand(batch_size,-1,-1)

        for layer in self.decoder:
            x = layer(x, mem)
        
        return self.out_proj(x)
    