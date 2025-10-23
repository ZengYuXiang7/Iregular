# coding : utf-8
# Author : Yuxiang Zeng
import torch
from models.layers.transformer import Transformer
from models.layers.encoder.graph_enc import GnnFamily

from einops import *


class Backbone(torch.nn.Module):
    def __init__(self, enc_in, config):
        super(Backbone, self).__init__()
        self.config = config
        self.d_model = config.d_model
        self.embedding = torch.nn.Linear(enc_in, self.d_model)

        self.encoder = Transformer(
            self.d_model,
            config.num_layers,
            config.num_heads,
            "rms",
            "ffn",
            config.att_method,
        )

        self.decoder = torch.nn.Linear(config.seq_len, config.pred_len)

        self.pred_head = torch.nn.Linear(self.d_model, enc_in)

    def forward(self, x, x_mark, x_mask):
        # x.shape = [bs, seq_len, d_channels]

        # TODO:: ReVIN

        x_enc = self.embedding(x)
        x_enc = self.encoder(x_enc)

        x_enc = rearrange(x_enc, "bs seq_len d_model -> bs d_model seq_len")
        x_enc = self.decoder(x_enc)
        x_enc = rearrange(x_enc, "bs d_model pred_len -> bs pred_len d_model")

        y = self.pred_head(x_enc)

        # TODO:: ReVIN

        return y


# coding: utf-8
# Author: Yuxiang Zeng (Neural ODE Transformer rewrite)
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from torchdiffeq import odeint, odeint_adjoint


# -------------------------
# Utilities
# -------------------------
class RMSNorm(nn.Module):
    """RMSNorm without bias. eps per common practice."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        # x: [B, T, D]
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x = x / rms
        return self.scale * x


class SelfAttentionODE(nn.Module):
    """轻量自注意力：Scaled Dot-Product。支持 x_mask (padding)。
    不写复杂多种 att_method 分支，保持简洁稳定。"""

    def __init__(self, d_model: int, num_heads: int, bias: bool = False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须整除 num_heads"
        self.h = num_heads
        self.dh = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x, mask=None):
        # x: [B, T, D], mask: [B, T] (True=keep/False=pad) 或 None
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.h, self.dh).transpose(1, 2)  # [B, H, T, Dh]
        k = self.k_proj(x).view(B, T, self.h, self.dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.h, self.dh).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dh**0.5)  # [B,H,T,T]

        if mask is not None:
            # mask: True->有效，False->padding；把 padding 位置设为 -inf
            # 扩展到 [B,1,1,T] 以便 broadcast 到 [B,H,T,T]
            attn_mask = (~mask).view(B, 1, 1, T)  # True 表示要屏蔽的位置
            attn_scores = attn_scores.masked_fill(attn_mask, float("-inf"))

        attn = F.softmax(attn_scores, dim=-1)
        y = torch.matmul(attn, v)  # [B,H,T,Dh]
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(y)


class FFN(nn.Module):
    def __init__(self, d_model: int, hidden_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(d_model * hidden_ratio)
        self.fc1 = nn.Linear(d_model, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# -------------------------
# ODE vector field f(H, s)
# -------------------------
class ODEFunc(nn.Module):
    """定义 dH/ds = f(H,s)。用注意力+FFN 构成向量场。"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = SelfAttentionODE(d_model, num_heads)
        self.norm2 = RMSNorm(d_model)
        self.ffn = FFN(d_model, hidden_ratio=4.0, dropout=dropout)

    def forward(self, s, H, mask=None):
        # s: 标量深度（float tensor, 由求解器传入），H: [B,T,D]
        # 注意：向量场不显式用 s（也可接入 FiLM/时间编码），保持简洁稳定
        A = self.attn(self.norm1(H), mask)  # 注意力部分
        F = self.ffn(self.norm2(H + A))  # 前馈部分（在 A 之上做 normalization）
        dH = A + F  # 合并成向量场
        return dH


class ODEBlock(nn.Module):
    """用 ODE 求解器在深度 s∈[0,1] 上积分 H。"""

    def __init__(
        self,
        func: ODEFunc,
        use_adjoint: bool = True,
        method: str = "dopri5",  # 可选: 'rk4', 'euler', 'dopri5'
        rtol: float = 1e-4,
        atol: float = 1e-5,
        return_intermediate: bool = False,
        steps: int = 2,
    ):
        super().__init__()
        self.func = func
        self.use_adjoint = use_adjoint
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.return_intermediate = return_intermediate

        # t_span（深度网格）；steps=2 表示只要起点与终点，交给自适应步长控制
        if return_intermediate and steps > 2:
            self.t_span = torch.linspace(0.0, 1.0, steps)
        else:
            self.t_span = torch.tensor([0.0, 1.0])

    def forward(self, H0, mask=None):
        # H0: [B,T,D]
        func = lambda t, y: self.func(t, y, mask)

        ode = odeint_adjoint if self.use_adjoint else odeint

        t = self.t_span.to(H0)
        out = ode(func, H0, t, method=self.method, rtol=self.rtol, atol=self.atol)

        if self.return_intermediate:
            # out: [steps, B, T, D]
            return out  # 若需要中间态可返回整个轨迹
        return out[-1]  # 只取 s=1 的终点作为编码结果


# -------------------------
# Backbone with Neural ODE
# -------------------------
class BackboneODE(nn.Module):
    """
    与你原始 Backbone 的接口保持一致：
    - embedding: Linear(enc_in -> d_model)
    - encoder: 由 ODEBlock 替代原 Transformer
    - decoder: Linear(seq_len -> pred_len)，在维度 [D, T] 上做线性映射
    - pred_head: Linear(d_model -> enc_in)
    """

    def __init__(self, enc_in, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model

        # Embedding
        self.embedding = nn.Linear(enc_in, self.d_model)

        # ODE Encoder（连续深度 Transformer）
        func = ODEFunc(
            d_model=self.d_model,
            num_heads=config.num_heads,
            dropout=getattr(config, "dropout", 0.0),
        )
        self.encoder = ODEBlock(
            func=func,
            use_adjoint=getattr(config, "ode_use_adjoint", True),
            method=getattr(config, "ode_method", "dopri5"),
            rtol=getattr(config, "ode_rtol", 1e-4),
            atol=getattr(config, "ode_atol", 1e-5),
            return_intermediate=getattr(config, "ode_return_intermediate", False),
            steps=getattr(config, "ode_steps", 2),
        )

        # 与你原实现保持一致的解码头
        self.decoder = nn.Linear(config.seq_len, config.pred_len)
        self.pred_head = nn.Linear(self.d_model, enc_in)

        # 可选：ReVIN 位置如需加入，可在 forward 前后放置（此处留空位）
        # self.revin = ReVIN(...)

    def forward(self, x, x_mark=None, x_mask=None):
        """
        x:       [B, seq_len, d_channels]
        x_mark:  [B, seq_len, ...] 时间戳/特征（此实现未显式使用；若要 Neural CDE 可扩展）
        x_mask:  [B, seq_len]，True=有效，False=padding
        """
        # 输入检查（卫语句）
        if x_mask is None:
            # 若没有 mask，默认全有效，避免在注意力内写更多分支
            x_mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)

        # TODO:: ReVIN (pre)
        x_enc = self.embedding(x)  # [B, T, D]
        x_enc = self.encoder(x_enc, mask=x_mask)  # [B, T, D] (s=1 的终态)

        # 你的解码头：在时间维上线性映射 seq_len->pred_len
        x_enc = rearrange(x_enc, "b t d -> b d t")
        x_enc = self.decoder(x_enc)  # [B, D, pred_len]
        x_enc = rearrange(x_enc, "b d p -> b p d")

        y = self.pred_head(x_enc)  # [B, pred_len, d_channels]

        # TODO:: ReVIN (post)
        return y
