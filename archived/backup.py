# coding : utf-8
# Author : Yuxiang Zeng
# 2025年11月28日20:28:31 备份
import torch
from models.layers.feedforward.moe import MoE
from models.layers.feedforward.smoe import SparseMoE
from models.layers.transformer import Transformer
from models.layers.encoder.graph_enc import GnnFamily
from torchdiffeq import odeint_adjoint as odeadj
from einops import *
from models.layers.dft import *
import torch.fft as fft


class FourierLayer(torch.nn.Module):

    def __init__(self, pred_len, k=None, low_freq=1, output_attention=False):
        super().__init__()
        # self.d_model = d_model
        self.pred_len = pred_len
        self.k = k
        self.low_freq = low_freq
        self.output_attention = output_attention

    def forward(self, x):
        """x: (b, t, d)"""

        if self.output_attention:
            return self.dft_forward(x)

        b, t, d = x.shape
        x_freq = fft.rfft(x, dim=1)

        if t % 2 == 0:
            x_freq = x_freq[:, self.low_freq : -1]
            f = fft.rfftfreq(t)[self.low_freq : -1]
        else:
            x_freq = x_freq[:, self.low_freq :]
            f = fft.rfftfreq(t)[self.low_freq :]

        x_freq, index_tuple = self.topk_freq(x_freq)
        f = repeat(f, "f -> b f d", b=x_freq.size(0), d=x_freq.size(2))
        f = f.to(x_freq.device)
        f = rearrange(f[index_tuple], "b f d -> b f () d").to(x_freq.device)

        return self.extrapolate(x_freq, f, t), None

    def extrapolate(self, x_freq, f, t):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t_val = rearrange(
            torch.arange(t + self.pred_len, dtype=torch.float), "t -> () () t ()"
        ).to(x_freq.device)

        amp = rearrange(x_freq.abs() / t, "b f d -> b f () d")
        phase = rearrange(x_freq.angle(), "b f d -> b f () d")

        x_time = amp * torch.cos(2 * math.pi * f * t_val + phase)

        return reduce(x_time, "b f t d -> b t d", "sum")

    def topk_freq(self, x_freq):
        values, indices = torch.topk(
            x_freq.abs(), self.k, dim=1, largest=True, sorted=True
        )
        mesh_a, mesh_b = torch.meshgrid(
            torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2))
        )
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]

        return x_freq, index_tuple

    def dft_forward(self, x):
        T = x.size(1)

        dft_mat = fft.fft(torch.eye(T))
        i, j = torch.meshgrid(torch.arange(self.pred_len + T), torch.arange(T))
        omega = np.exp(2 * math.pi * 1j / T)
        idft_mat = (np.power(omega, i * j) / T).cfloat()

        x_freq = torch.einsum("ft,btd->bfd", [dft_mat, x.cfloat()])

        if T % 2 == 0:
            x_freq = x_freq[:, self.low_freq : T // 2]
        else:
            x_freq = x_freq[:, self.low_freq : T // 2 + 1]

        _, indices = torch.topk(x_freq.abs(), self.k, dim=1, largest=True, sorted=True)
        indices = indices + self.low_freq
        indices = torch.cat([indices, -indices], dim=1)

        dft_mat = repeat(dft_mat, "f t -> b f t d", b=x.shape[0], d=x.shape[-1])
        idft_mat = repeat(idft_mat, "t f -> b t f d", b=x.shape[0], d=x.shape[-1])

        mesh_a, mesh_b = torch.meshgrid(
            torch.arange(x.size(0)), torch.arange(x.size(2))
        )

        dft_mask = torch.zeros_like(dft_mat)
        dft_mask[mesh_a, indices, :, mesh_b] = 1
        dft_mat = dft_mat * dft_mask

        idft_mask = torch.zeros_like(idft_mat)
        idft_mask[mesh_a, :, indices, mesh_b] = 1
        idft_mat = idft_mat * idft_mask

        attn = torch.einsum("bofd,bftd->botd", [idft_mat, dft_mat]).real
        return torch.einsum("botd,btd->bod", [attn, x]), rearrange(
            attn, "b o t d -> b d o t"
        )


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
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(torch.nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        x_trend = self.moving_avg(x)
        x_season = x - x_trend
        return x_trend, x_season


class Backbone(torch.nn.Module):
    def __init__(self, enc_in, config):
        super(Backbone, self).__init__()
        self.config = config
        self.revin = config.revin
        self.d_model = config.d_model
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.patch_len = config.patch_len
        self.patch_num = (self.seq_len) // self.patch_len

        self.denoise_module = DFT(5)

        self.embedding = torch.nn.Linear(enc_in, self.d_model)
        # self.embedding = TokenEmbedding(enc_in, self.d_model)
        self.season_model = FourierLayer(pred_len=0, k=3)
        self.trend_model = series_decomp(25)

        self.trend_linear = torch.nn.Linear(enc_in, self.d_model)
        self.season_linear = torch.nn.Linear(enc_in, self.d_model)

        self.mask_embedding = torch.nn.Linear(enc_in, self.d_model)

        # self.encoder = LinearEncoder(self.d_model * self.patch_len)
        # self.encoder2 = LinearEncoder(self.d_model * self.patch_num)

        self.inter_encoder = Transformer(
            self.d_model * self.patch_len,
            config.num_layers,
            config.num_heads,
            "rms",
            "ffn",
            config.att_method,
        )

        self.intra_encoder = Transformer(
            self.d_model * self.patch_num,
            config.num_layers,
            config.num_heads,
            "rms",
            "ffn",
            config.att_method,
        )

        self.feature_encoder = Transformer(
            self.seq_len,
            1,
            config.num_heads,
            "rms",
            "ffn",
            config.att_method,
        )

        self.moe = MoE(
            d_model=self.d_model,
            d_ff=self.d_model,
            num_m=1,
            num_router_experts=5,
            num_share_experts=0,
            num_k=1,
            loss_coef=0.001,
        )

        self.decoder = torch.nn.Linear(config.seq_len, config.pred_len)

        # Neural ODE Module
        # self.ode_block = ODEBlock(f(dim=self.d_model))

        self.pred_head = torch.nn.Linear(self.d_model, enc_in)

    def forward(self, x, x_mark, x_mask):
        # x.shape = [bs, seq_len, d_channels]
        # x_mask.shape = [bs, seq_len, d_channels]

        if self.revin:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev

        # x = self.denoise_module(x)

        # Decomposition
        x_trend, _ = self.trend_model(x)
        x_season, _ = self.season_model(x)

        # Feature embedding
        # [bs seq_len d_model]
        x_enc = self.embedding(x)
        x_trend = self.trend_linear(x_trend)
        x_season = self.season_linear(x_season)

        x_enc = x_enc + x_trend + x_season

        # Patch
        # 先按时间分 patch
        x_enc = rearrange(
            x_enc,
            "bs (patch_num patch_len) d_model -> bs patch_num patch_len d_model",
            patch_num=self.patch_num,
            patch_len=self.patch_len,
        )
        # x_enc.shape = [bs patch_num patch_len d_model]

        # 第一步
        # patch_num : patch_len * d
        x_inter = rearrange(
            x_enc,
            "bs patch_num patch_len d_model -> bs patch_num (patch_len d_model)",
            patch_num=self.patch_num,
            patch_len=self.patch_len,
        )
        x_inter = self.inter_encoder(x_inter)
        x_inter = rearrange(
            x_inter,
            "bs patch_num (patch_len d_model) -> bs (patch_num patch_len) d_model",
            patch_num=self.patch_num,
            patch_len=self.patch_len,
        )

        # 第二步
        # patch_len : patch_num * d
        x_intra = rearrange(
            x_enc,
            "bs patch_num patch_len d_model -> bs patch_len (patch_num d_model)",
            patch_num=self.patch_num,
            patch_len=self.patch_len,
        )
        x_intra = self.intra_encoder(x_intra)
        x_intra = rearrange(
            x_intra,
            "bs patch_len (patch_num d_model) -> bs (patch_num patch_len) d_model",
            patch_num=self.patch_num,
            patch_len=self.patch_len,
        )

        # patch还原
        x_enc = rearrange(
            x_enc,
            "bs patch_num patch_len d_model -> bs (patch_num patch_len) d_model",
            patch_num=self.patch_num,
            d_model=self.d_model,
        )

        # Patching Fusion
        x_enc = x_enc + x_inter + x_intra
        x_enc = rearrange(
            x_enc,
            "bs seq_len d_model -> bs d_model seq_len",
        )

        # Feature interaction
        x_enc = self.feature_encoder(x_enc)  # [bs, d_model, seq_len]
        x_enc = rearrange(x_enc, "bs d_model seq_len -> bs seq_len d_model")

        # Mixture of Experts
        x_enc = self.moe(x_enc)

        # Time projection
        x_enc = rearrange(x_enc, "bs seq_len d_model -> bs d_model seq_len")
        x_enc = self.decoder(x_enc)
        x_enc = rearrange(x_enc, "bs d_model pred_len -> bs pred_len d_model")

        # Downstream Task
        y = self.pred_head(x_enc)[:, -self.pred_len :, :]

        if self.revin:
            y = y * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            y = y + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return y


##################################################################################
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


class LinearEncoder(torch.nn.Module):
    def __init__(self, d_model):
        super(LinearEncoder, self).__init__()
        self.d_model = d_model
        self.f = torch.nn.Sequential(
            torch.nn.Linear(self.d_model, self.d_model * 2),
            torch.nn.LayerNorm(self.d_model * 2, self.d_model * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.d_model * 2, self.d_model),
            torch.nn.LayerNorm(self.d_model, self.d_model),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        return self.f(x)


class TokenEmbedding(torch.nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = torch.nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x
