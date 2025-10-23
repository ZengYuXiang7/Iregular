# coding: utf-8
import torch
from torch import nn
from torch.nn import init

class ExternalAttention(nn.Module):
    def __init__(self, d_model, S=64):
        super().__init__()
        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)
        self.softmax = nn.Softmax(dim=-1)        # 在最后一维做 softmax
        self.apply(self._init_weights)            # 推荐用 apply 初始化

    def _init_weights(self, m: nn.Module):
        # 用 apply 遍历所有子模块并做分类初始化
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            init.ones_(m.weight)
            init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                init.zeros_(m.bias)

    def forward(self, queries, att_mask=None):
        # queries: [bs, n, d_model]
        attn = self.mk(queries)           # [bs, n, S]
        attn = self.softmax(attn)         # 对 S 维做 softmax，已保证沿 S 求和为 1
        out = self.mv(attn)               # [bs, n, d_model]
        return out

if __name__ == '__main__':
    x = torch.randn(128, 49, 50)
    ea = ExternalAttention(d_model=50, S=8)
    y = ea(x)
    print(y.shape)  # torch.Size([128, 49, 50])