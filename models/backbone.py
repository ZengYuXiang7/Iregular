# coding : utf-8
# Author : Yuxiang Zeng
import torch
from models.layers.transformer import Transformer
from models.layers.encoder.graph_enc import GnnFamily
from torchdiffeq import odeint_adjoint as odeadj
from einops import *


class Backbone(torch.nn.Module):
    def __init__(self, enc_in, config):
        super(Backbone, self).__init__()
        self.config = config
        self.d_model = config.d_model
        self.embedding = torch.nn.Linear(enc_in, self.d_model)
        self.mask_embedding = torch.nn.Linear(enc_in, self.d_model)
        
        self.encoder = Transformer(
            self.d_model,
            config.num_layers,
            config.num_heads,
            "rms",
            "ffn",
            config.att_method,
        )

        # Neural ODE Module
        self.ode_block = ODEBlock(f(dim=self.d_model))

        self.decoder = torch.nn.Linear(config.seq_len, config.pred_len)
        self.pred_head = torch.nn.Linear(self.d_model, enc_in)

    def forward(self, x, x_mark, x_mask):
        # x.shape = [bs, seq_len, d_channels]
        # x_mask.shape = [bs, seq_len, d_channels]

        # TODO:: ReVIN

        x_enc = self.embedding(x)

        if self.config.try_exp == 1:
            x_enc = x_enc + self.mask_embedding(x_mask)
            
        x_enc = self.encoder(x_enc)

        if self.config.try_exp == 3:
            x_enc = self.ode_block(x_enc)

        x_enc = rearrange(x_enc, "bs seq_len d_model -> bs d_model seq_len")
        x_enc = self.decoder(x_enc)
        x_enc = rearrange(x_enc, "bs d_model pred_len -> bs pred_len d_model")

        y = self.pred_head(x_enc)

        # TODO:: ReVIN

        return y


# --- Neural ODE 模块 (来自您之前的代码) ---
class f(torch.nn.Module):
    def __init__(self, dim):
        super(f, self).__init__()
        # 为了更稳定，可以使用更简单的网络
        self.model = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(dim * 2, dim),
            torch.nn.Tanh(),  # Tanh 将输出限制在 -1 到 1，有助于稳定 ODE
        )

    def forward(self, t, x):
        # 这里的 f(x) 定义了导数 dx/dt
        return self.model(x)


class ODEBlock(torch.nn.Module):
    def __init__(self, f):
        super(ODEBlock, self).__init__()
        self.f = f
        self.integration_time = torch.Tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)

        # 使用 'dopri5' 求解器可能更稳定
        # odeadj 是带有伴随方法的求解器，节省内存
        out = odeadj(
            self.f,
            x,
            self.integration_time,
            method="dopri5",  # 尝试指定一个稳定的求解器
            atol=1e-5,  # 降低容忍度以提高速度
            rtol=1e-5,
        )

        # 返回积分结束时的状态 h(1)
        return out[1]
