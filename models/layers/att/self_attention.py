# coding : utf-8
# Author : Yuxiang Zeng
import torch
from einops import rearrange


class Attention(torch.nn.Module):
    def __init__(self, d_model, num_heads, is_causal=False, dropout=0.10):
        super().__init__()
        # batch_first=True 意味着输入是 (Batch, Seq_Len, Dim)
        self.is_causal = is_causal
        self.att = torch.nn.MultiheadAttention(
            d_model, num_heads, dropout, batch_first=True
        )

    def forward(self, x, attn_mask=None):
        """
        :param x: [Batch, Seq_Len, D_Model]
        :param attn_mask: 外部传入的掩码
        :param is_causal: 是否开启因果掩码 (只能看过去，不能看未来)
        """
        seq_len = x.shape[1]

        # 如果开启因果掩码，且没有外部 mask
        if self.is_causal and attn_mask is None:
            # 1. 创建一个全为 -inf 的矩阵
            # 2. torch.triu 保留上三角，其余位置（下三角）自动变 0
            # 3. diagonal=1 表示对角线本身不设为 -inf (即自己可以看到自己)
            attn_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=x.device),
                diagonal=1,
            )
            # 最终 mask 形状: [Seq_Len, Seq_Len]
            # [ 0, -inf, -inf ]
            # [ 0,  0,   -inf ]
            # [ 0,  0,    0   ]

        # 处理外部传入的 4D mask (如果有的话，兼容你原来的代码逻辑)
        if attn_mask is not None and attn_mask.dim() == 4:
            bs, h, n, m = attn_mask.shape
            attn_mask = attn_mask.reshape(bs * h, n, m)

        # 传入 attn_mask
        out, weights = self.att(x, x, x, attn_mask=attn_mask)
        return out


if __name__ == "__main__":
    inputs = torch.randn(1, 10, 64)
    model = Attention(d_model=64, num_heads=8, dropout=0.10)

    # 开启 is_causal=True
    out = model(inputs, is_causal=True)
    print("Output shape:", out.shape)

    # 验证是否生效：查看 Attention Weights (需要临时修改 forward 返回 weights)
    # 正常情况下，weights 的上三角部分应该全是 0 (因为 score 是 -inf)
