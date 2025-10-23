
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import init


class FullAttention(nn.Module):
    def __init__(self, is_casual=False, attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.is_casual = is_casual
        self.dropout_p = attention_dropout
        self.output_attention = output_attention

    def forward(self, queries, keys, values, attn_mask=None):
        # 输入: [B, L, H, D]
        q = queries.permute(0, 2, 1, 3)  # [B,H,L,D]
        k = keys.permute(0, 2, 1, 3)     # [B,H,S,D]
        v = values.permute(0, 2, 1, 3)   # [B,H,S,D]

        A = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=self.is_casual,
        )  # [B,H,L,D]

        out = A.permute(0, 2, 1, 3).contiguous()  # 回到 [B,L,H,D]
        return out
        
        
class AttentionLayer(nn.Module):
    def __init__(self, attn, d_model, n_heads):
        super(AttentionLayer, self).__init__()

        self.inner_attention = attn
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        
        self.out_projection = nn.Linear(d_model, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
        )
        out = out.view(B, L, -1)

        return self.out_projection(out)
    
    
class CustomAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.dot_product = FullAttention()
        self.att = AttentionLayer(d_model=d_model, n_heads=n_heads, attn=self.dot_product)

    def forward(self, x, attn_mask=None):
        out = self.att(x, x, x, attn_mask=attn_mask)
        return out

    
if __name__ == "__main__":
    # 超参数
    B, L, S, d_model, n_heads = 2, 4, 6, 16, 4

    # 构造输入
    x = torch.randn(B, L, d_model)   # [2, 4, 16]
    attn_mask = None   # 也可以用 TriangularCausalMask(B, L).mask

    att = CustomAttention(d_model=d_model, n_heads=n_heads)
    # 构造 AttentionLayer
    out, attn = att(x)

    print("out.shape:", out.shape)   # [B, L, d_model] -> [2, 4, 16]
    print("attn.shape:", attn.shape if attn is not None else None)  
    # [B, H, L, S] -> [2, 4, 4, 6]