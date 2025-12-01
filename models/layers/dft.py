# coding : utf-8
# Author : Yuxiang Zeng
import torch
import numpy as np
import math
from einops import rearrange, reduce, repeat

class DFT(torch.nn.Module):
    def __init__(self, top_k):
        super(DFT, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        L = x.size(1)
        xf = torch.fft.rfft(x, dim=1)        # [B, L_freq, C]
        freq = xf.abs()
        freq[:, 0, :] = 0                   # freq[..., 0] 会是最后一维，这里我们要的是时间频率的 0
        top_k_freq, _ = torch.topk(freq, self.top_k, dim=1)  # [B, top_k, C]
        thr = top_k_freq[:, -1, :].unsqueeze(1)  # [B, 1, C]，方便广播
        xf = torch.where(freq >= thr, xf, xf.new_zeros(()))
        x_season = torch.fft.irfft(xf, n=L, dim=1)  # [B, L, C] = [512, 96, 21]
        return x_season


class moving_avg(torch.nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
