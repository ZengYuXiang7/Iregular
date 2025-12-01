# coding: utf-8
# Author: mkw
# Date: 2025-06-08 14:37
# Description: SeasonalTrendModelConfig

from configs.default_config import *
from dataclasses import dataclass, field


@dataclass
class OurModelConfig(
    ExperimentConfig,
    BaseModelConfig,
    LoggerConfig,
    DatasetInfo,
    TrainingConfig,
    OtherConfig,
):
    data_dropout: float = 0.3  # 不规则采样的缺失率

    # 模型基本参数
    model: str = "ours"
    optim: str = "Adam"
    rounds: int = 3
    epochs: int = 20
    bs: int = 512
    patience: int = 5
    verbose: int = 50

    # 模型维度参数
    input_size: int = 21
    d_model: int = 256
    revin: bool = True
    patch_len: int = 16

    # Transformer 结构
    att_method: str = "self"
    num_layers: int = 2
    num_heads: int = 8
    att_bias: bool = False  # 是否使用距离嵌入

    predict_target: str = "y"  #  latency accuracy

    # Monitor MAE
    monitor_reverse: bool = False
    monitor_metric: str = "MAE"

    try_exp: int = 1  # 1-8

    # Causal Method
    #
    thresh: float = 0.3
    pc_alpha: float = 0.05
    causal_lr: float = 0.05
    pre_gate: float = 0.80
    sub_method: str = "DirectLiNGAM"
    golem_epoch: float = 5000
