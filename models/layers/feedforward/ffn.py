# coding : utf-8
# Author : yuxiang Zeng

import torch.nn as nn
import torch.nn.init as init

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # self.apply(self.init_weights)
    
    # def init_weights(self, m):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             init.normal_(m.weight, std=0.02)
    #             if m.bias is not None:
    #                 init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)