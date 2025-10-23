import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Optional

# =========================
# 基础规则（行业常用）
# =========================
def init_linear_kaiming(m: nn.Linear):
    # MLP / ReLU 系：中间层常用
    init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
    if m.bias is not None:
        init.zeros_(m.bias)

def init_linear_xavier(m: nn.Linear, gain: float = 1.0):
    # 残差/Transformer/近似线性激活（GELU/SiLU）常用
    init.xavier_uniform_(m.weight, gain=gain)
    if m.bias is not None:
        init.zeros_(m.bias)

def init_linear_small_normal(m: nn.Linear, std: float = 0.02):
    # NLP/ViT/最终分类头常用的小方差正态
    init.normal_(m.weight, mean=0.0, std=std)
    if m.bias is not None:
        init.zeros_(m.bias)

def init_conv_kaiming(m: nn.modules.conv._ConvNd, fan_mode: str = "fan_out"):
    # 经典 CNN：Kaiming + fan_out（梯度更稳）
    init.kaiming_normal_(m.weight, mode=fan_mode, nonlinearity="relu")
    if m.bias is not None:
        init.zeros_(m.bias)

def init_bn(m: nn.modules.batchnorm._BatchNorm):
    if m.affine:
        init.ones_(m.weight)
        init.zeros_(m.bias)

def init_ln(m: nn.LayerNorm):
    if m.elementwise_affine:
        init.ones_(m.weight)
        init.zeros_(m.bias)

def init_embedding_xavier(m: nn.Embedding):
    init.xavier_uniform_(m.weight)

# =========================
# Transformer 细化（行业实践）
# =========================
def init_mha(m: nn.MultiheadAttention):
    # 与 PyTorch/论文常见实现对齐：QKV/Out 用 Xavier，bias=0
    if m._qkv_same_embed_dim:
        init.xavier_uniform_(m.in_proj_weight)
        if m.in_proj_bias is not None:
            init.zeros_(m.in_proj_bias)
    else:
        init.xavier_uniform_(m.q_proj_weight)
        init.xavier_uniform_(m.k_proj_weight)
        init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            init.zeros_(m.in_proj_bias)
    init.xavier_uniform_(m.out_proj.weight)
    if m.out_proj.bias is not None:
        init.zeros_(m.out_proj.bias)

def init_transformer_ffn(linear1: nn.Linear, linear2: nn.Linear):
    # FFN 两层：业界常见 Xavier
    init_linear_xavier(linear1)
    init_linear_xavier(linear2)

# =========================
# 一键初始化入口（按模型类型）
# =========================
def init_for_cnn(model: nn.Module, *, verbose: bool = False):
    """
    行业常用 CNN 初始化：
    - Conv: Kaiming Normal (fan_out)
    - BN: gamma=1, beta=0
    - Linear(中间/普通): Kaiming（若无激活依赖也可改 xavier）
    - 最终分类头（若名含 'classifier' 或 'fc' 且输出维较小）：Normal(0, 0.01)
    """
    def _is_classifier(m: nn.Module) -> bool:
        return isinstance(m, nn.Linear) and (m.out_features <= 1000)

    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            init_conv_kaiming(m, fan_mode="fan_out")
            if verbose: print("[init] Conv Kaiming fan_out:", m)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            init_bn(m)
            if verbose: print("[init] BN:", m)
        elif isinstance(m, nn.Linear):
            if _is_classifier(m):
                init_linear_small_normal(m, std=0.01)  # 经典分类头
                if verbose: print("[init] Linear classifier N(0,0.01):", m)
            else:
                init_linear_kaiming(m)  # 中间层
                if verbose: print("[init] Linear Kaiming:", m)
        elif isinstance(m, nn.LayerNorm):
            init_ln(m)

def init_for_mlp(model: nn.Module, *, use_relu: bool = True, verbose: bool = False):
    """
    通用 MLP/小模型：
    - 若后接 ReLU：Kaiming fan_in
    - 若是 GELU/SiLU 或残差：Xavier
    - 输出头（若 name 含 'head'/'classifier'）：Normal(0, 0.01)
    """
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            if ("head" in name or "classifier" in name) and m.out_features <= 1000:
                init_linear_small_normal(m, std=0.01)
                if verbose: print(f"[init] {name}: Linear head N(0,0.01)")
            else:
                if use_relu:
                    init_linear_kaiming(m)
                    if verbose: print(f"[init] {name}: Linear Kaiming")
                else:
                    init_linear_xavier(m)
                    if verbose: print(f"[init] {name}: Linear Xavier")
        elif isinstance(m, nn.LayerNorm):
            init_ln(m)

def init_for_transformer(model: nn.Module, *, head_std: float = 0.02, verbose: bool = False):
    """
    行业主流 Transformer 初始化（Encoder/Decoder/ViT/BERT 风格）：
    - MultiheadAttention: Xavier（Q/K/V/Out）
    - FFN 两层: Xavier
    - LayerNorm: gamma=1, beta=0
    - 词嵌入/位置嵌入: Xavier 或 N(0,0.02)（NLP/ViT 多用 0.02）
    - 最终分类/回归头: N(0, head_std)  (默认 0.02，BERT/ViT 常用)
    """
    for name, m in model.named_modules():
        if isinstance(m, nn.MultiheadAttention):
            init_mha(m)
            if verbose: print(f"[init] {name}: MHA Xavier")
        elif isinstance(m, nn.TransformerEncoderLayer):
            init_mha(m.self_attn)
            init_transformer_ffn(m.linear1, m.linear2)
            init_ln(m.norm1); init_ln(m.norm2)
            if verbose: print(f"[init] {name}: EncoderLayer (MHA+FFN Xavier, LN)")
        elif isinstance(m, nn.TransformerDecoderLayer):
            init_mha(m.self_attn); init_mha(m.multihead_attn)
            init_transformer_ffn(m.linear1, m.linear2)
            init_ln(m.norm1); init_ln(m.norm2); init_ln(m.norm3)
            if verbose: print(f"[init] {name}: DecoderLayer (MHA+FFN Xavier, LN)")
        elif isinstance(m, nn.Embedding):
            # 两种都行：按需选一个
            init_embedding_xavier(m)            # Xavier
            # init.normal_(m.weight, 0.0, 0.02)  # BERT/ViT 风格
            if verbose: print(f"[init] {name}: Embedding Xavier")
        elif isinstance(m, nn.LayerNorm):
            init_ln(m)
        elif isinstance(m, nn.Linear):
            # 对于非特定命名的 Linear（比如自定义 block 内的），默认用 Xavier
            if "head" in name or "classifier" in name:
                init_linear_small_normal(m, std=head_std)
                if verbose: print(f"[init] {name}: Head N(0,{head_std})")
            else:
                init_linear_xavier(m)
                if verbose: print(f"[init] {name}: Linear Xavier")

# =========================
# 统一入口（按字符串选择）
# =========================
def initialize(model: nn.Module, kind: str = "transformer", verbose: bool = False, **kwargs):
    kind = kind.lower()
    if kind in ("cnn", "resnet"):
        init_for_cnn(model, verbose=verbose)
    elif kind in ("mlp", "mlp-relu", "mlp_gelu", "mlp-silu"):
        use_relu = not ("gelu" in kind or "silu" in kind)
        init_for_mlp(model, use_relu=use_relu, verbose=verbose)
    elif kind in ("transformer", "bert", "vit"):
        init_for_transformer(model, verbose=verbose, **kwargs)
    else:
        # 默认更通用的：Xavier + LN
        for m in model.modules():
            if isinstance(m, nn.Linear):
                init_linear_xavier(m)
            elif isinstance(m, nn.LayerNorm):
                init_ln(m)
        if verbose:
            print(f"[init] fallback: Xavier+LN for {model.__class__.__name__}")
    return model