import torch
import torch.nn as nn
import math
import torch.nn.init as init

class DiscreteEncoder(nn.Module):
    def __init__(
        self,
        num_operations: int,
        encoding_dim: int,
        encoding_type: str = 'embedding',
        max_freq_log2: int = 10,
        num_freqs: int = 10,
        eps: float = 1e-5,
        output_dim: int | None = None,   # ← 新增：外置映射到指定维度
        post_bias: bool = False,         # 可选：是否给映射层加 bias
    ):
        """
        :param num_operations: 离散操作的种类数（vocab size）
        :param encoding_dim: 编码维度（embedding size / d_model）
        :param encoding_type: 'embedding' | 'onehot' | 'sinusoidal' | 'nerf' | 'nape' | 'trans'
        :param max_freq_log2: nape 中用于缩放频率的对数上限
        :param num_freqs: 频带数量（用于 nerf / nape；trans 用于多频份数）
        :param eps: nape 的微小正数，避免 0 频率
        :param output_dim: 若不为 None，则把编码结果最后一维统一映射到该维度
        :param post_bias: 外置线性映射是否使用 bias
        """
        super().__init__()
        self.encoding_type = encoding_type
        self.num_operations = num_operations
        self.encoding_dim = encoding_dim
        self.num_freqs = num_freqs
        self.eps = eps
        self.max_freq_log2 = max_freq_log2
        self.output_dim = output_dim
        self.post_bias = post_bias

        # -- 构造各编码所需的参数/缓冲 --
        if encoding_type == 'embedding':
            self.operation_embedding = nn.Embedding(num_operations, encoding_dim)
            native_dim = encoding_dim

        elif encoding_type == 'onehot':
            # 直接返回 one-hot，不做线性投影
            native_dim = num_operations

        elif encoding_type == 'sinusoidal':
            half = max(1, encoding_dim // 2)
            inv_freq = 1.0 / (10000 ** (torch.arange(0, half, dtype=torch.float32) / float(half)))
            self.register_buffer('inv_freq_sinusoidal', inv_freq, persistent=False)
            native_dim = encoding_dim

        elif encoding_type == 'trans':
            half = max(1, encoding_dim // 2)
            inv_freq = 1.0 / (10000 ** (torch.arange(0, half, dtype=torch.float32) / float(half)))
            self.register_buffer('inv_freq_trans', inv_freq, persistent=False)
            native_dim = encoding_dim * num_freqs

        elif encoding_type == 'nerf':
            freq = 2.0 ** torch.arange(0, num_freqs, dtype=torch.float32)
            self.register_buffer('freq_nerf', freq, persistent=False)
            self.proj_nerf = nn.Linear(2 * num_freqs, encoding_dim * num_freqs, bias=False)
            native_dim = encoding_dim * num_freqs

        elif encoding_type == 'nape':
            max_freq = max(1, self.num_freqs)
            base = self.eps + torch.linspace(1, float(max_freq), self.num_freqs, dtype=torch.float32)
            freq = base * math.pi / (self.max_freq_log2 + 1.0)
            self.register_buffer('freq_nape', freq, persistent=False)
            self.proj_nape = nn.Linear(2 * num_freqs, encoding_dim * num_freqs, bias=False)
            native_dim = encoding_dim * num_freqs

        else:
            raise ValueError("Unsupported encoding_type. Choose from ['embedding','onehot','sinusoidal','nerf','nape','trans'].")

        # -- 外置统一映射层（可选） --
        self.native_dim = native_dim
        if output_dim is not None and output_dim != native_dim:
            self.proj_out = nn.Linear(native_dim, output_dim, bias=post_bias)
        else:
            self.proj_out = None  # 恒等
            
        self.apply(self.init_weights)
    
    def init_weights(self, m):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
                    
    def forward(self, operation_ids: torch.Tensor) -> torch.Tensor:
        """
        :param operation_ids: (B, n) 的整型 ID
        :return: 
          - embedding:   (B, n, encoding_dim)               -> 若设置 output_dim，则 (B, n, output_dim)
          - onehot:      (B, n, num_operations)             -> 若设置 output_dim，则 (B, n, output_dim)
          - sinusoidal:  (B, n, encoding_dim)               -> 若设置 output_dim，则 (B, n, output_dim)
          - trans:       (B, n, encoding_dim * num_freqs)   -> 若设置 output_dim，则 (B, n, output_dim)
          - nerf:        (B, n, encoding_dim * num_freqs)   -> 若设置 output_dim，则 (B, n, output_dim)
          - nape:        (B, n, encoding_dim * num_freqs)   -> 若设置 output_dim，则 (B, n, output_dim)
        """
        if self.encoding_type == 'embedding':
            x = self._embedding_encoding(operation_ids)
        elif self.encoding_type == 'onehot':
            x = self._onehot_encoding(operation_ids)
        elif self.encoding_type == 'sinusoidal':
            x = self._sinusoidal_encoding(operation_ids)
        elif self.encoding_type == 'trans':
            x = self._trans_encoding(operation_ids)
        elif self.encoding_type == 'nerf':
            x = self._nerf_encoding(operation_ids)
        elif self.encoding_type == 'nape':
            x = self._nape_encoding(operation_ids)
        else:
            x = operation_ids  # 不会到这里

        # 统一映射（若需要）
        if self.proj_out is not None:
            x = self.proj_out(x)
        return x

    # ---------- encoders ----------

    def _embedding_encoding(self, operation_ids: torch.Tensor) -> torch.Tensor:
        return self.operation_embedding(operation_ids)

    def _onehot_encoding(self, operation_ids: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(operation_ids, num_classes=self.num_operations).to(dtype=torch.float32)

    def _sinusoidal_encoding(self, operation_ids: torch.Tensor) -> torch.Tensor:
        B, N = operation_ids.shape
        device = operation_ids.device
        dtype = torch.get_default_dtype()

        pos = operation_ids.to(dtype=dtype).unsqueeze(-1)  # (B, n, 1)
        inv_freq = self.inv_freq_sinusoidal.to(device=device, dtype=dtype)  # (half,)
        sinusoid_inp = pos * inv_freq  # (B, n, half)

        sin = torch.sin(sinusoid_inp)
        cos = torch.cos(sinusoid_inp)
        pe = torch.cat([sin, cos], dim=-1)  # (B, n, 2*half)

        # 对齐到 encoding_dim
        if pe.shape[-1] < self.encoding_dim:
            pe = torch.nn.functional.pad(pe, (0, self.encoding_dim - pe.shape[-1]))
        elif pe.shape[-1] > self.encoding_dim:
            pe = pe[..., : self.encoding_dim]
        return pe

    def _trans_encoding(self, operation_ids: torch.Tensor) -> torch.Tensor:
        """
        多频：在 sinusoidal 的基础上，为每个频带应用缩放 s_r（几何级数衰减），
        然后把每个频带的特征对齐到 encoding_dim 并级联 -> (B,n,encoding_dim * num_freqs)
        """
        B, N = operation_ids.shape
        device = operation_ids.device
        dtype = torch.get_default_dtype()

        pos = operation_ids.to(dtype=dtype).unsqueeze(-1)          # (B, n, 1)
        inv_base = self.inv_freq_trans.to(device=device, dtype=dtype)  # (half,)

        scales = torch.tensor([0.5 ** r for r in range(self.num_freqs)], device=device, dtype=dtype)  # (num_freqs,)
        outs = []
        for s in scales:
            inv_freq = inv_base * s
            theta = pos * inv_freq                                  # (B, n, half)
            base = torch.cat([torch.sin(theta), torch.cos(theta)], dim=-1)  # (B, n, 2*half)

            if base.shape[-1] < self.encoding_dim:
                base = torch.nn.functional.pad(base, (0, self.encoding_dim - base.shape[-1]))
            elif base.shape[-1] > self.encoding_dim:
                base = base[..., : self.encoding_dim]
            outs.append(base)

        return torch.cat(outs, dim=-1)  # (B, n, encoding_dim * num_freqs)

    def _nerf_encoding(self, operation_ids: torch.Tensor) -> torch.Tensor:
        """
        NeRF 风格：ID 归一化到 [0,1]，再乘 2π 与 2^k 形成多频；线性映射到 (encoding_dim * num_freqs)
        """
        B, N = operation_ids.shape
        device = operation_ids.device
        dtype = torch.get_default_dtype()

        denom = max(self.num_operations - 1, 1)
        x = (operation_ids.to(dtype=dtype) / float(denom)).unsqueeze(-1)  # (B, n, 1) in [0,1]

        freq = self.freq_nerf.to(device=device, dtype=dtype)  # (num_freqs,)
        ang = x * (2.0 * math.pi * freq)                      # (B, n, num_freqs)

        feats = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # (B, n, 2*num_freqs)
        return self.proj_nerf(feats)  # (B, n, encoding_dim * num_freqs)

    def _nape_encoding(self, operation_ids: torch.Tensor) -> torch.Tensor:
        """
        NAPE：使用非整数倍 π 的频带，避免在整数 ID 上退化；线性映射到 (encoding_dim * num_freqs)
        """
        B, N = operation_ids.shape
        device = operation_ids.device
        dtype = torch.get_default_dtype()

        pos = operation_ids.to(dtype=dtype).unsqueeze(-1)  # (B, n, 1)
        freq = self.freq_nape.to(device=device, dtype=dtype)  # (num_freqs,)
        ang = pos * freq  # (B, n, num_freqs)

        feats = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # (B, n, 2*num_freqs)
        return self.proj_nape(feats)  # (B, n, encoding_dim * num_freqs)


# ----------------- 使用示例（含明显打印） -----------------
if __name__ == "__main__":
    bs, seq_len = 1, 3
    operation_ids = torch.randint(0, 5, (bs, seq_len))
    dim = 96
    out_dim = 128  # 你想要的统一输出维度

    print("===== 输入 operation_ids =====")
    print(operation_ids, "\n")

    op_encoder = DiscreteEncoder(num_operations=10, encoding_dim=dim, encoding_type='embedding', output_dim=out_dim)
    embeddings = op_encoder(operation_ids)
    print(">>> [Embedding 编码] 输出 shape:", embeddings.shape, "\n")  # (B,n,out_dim)

    op_encoder_onehot = DiscreteEncoder(num_operations=10, encoding_dim=dim, encoding_type='onehot', output_dim=out_dim)
    embeddings_onehot = op_encoder_onehot(operation_ids)
    print(">>> [OneHot 编码] 输出 shape:", embeddings_onehot.shape, "\n")  # (B,n,out_dim)

    op_encoder_sin = DiscreteEncoder(num_operations=10, encoding_dim=dim, encoding_type='sinusoidal', output_dim=out_dim)
    embeddings_sin = op_encoder_sin(operation_ids)
    print(">>> [Sinusoidal 编码] 输出 shape:", embeddings_sin.shape, "\n")  # (B,n,out_dim)

    op_encoder_trans = DiscreteEncoder(num_operations=10, encoding_dim=dim, encoding_type='trans', num_freqs=10, output_dim=out_dim)
    embeddings_trans = op_encoder_trans(operation_ids)
    print(">>> [Trans 编码] 输出 shape:", embeddings_trans.shape, "\n")  # (B,n,out_dim)

    op_encoder_nerf = DiscreteEncoder(num_operations=10, encoding_dim=dim, encoding_type='nerf', num_freqs=10, output_dim=out_dim)
    embeddings_nerf = op_encoder_nerf(operation_ids)
    print(">>> [NeRF 编码] 输出 shape:", embeddings_nerf.shape, "\n")  # (B,n,out_dim)

    op_encoder_nape = DiscreteEncoder(num_operations=10, encoding_dim=dim, encoding_type='nape', num_freqs=10, eps=1e-2, output_dim=out_dim)
    embeddings_nape = op_encoder_nape(operation_ids)
    print(">>> [NAPE 编码] 输出 shape:", embeddings_nape.shape, "\n")  # (B,n,out_dim)