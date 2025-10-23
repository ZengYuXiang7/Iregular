# coding : utf-8
# Author : yuxiang Zeng
from dataclasses import dataclass

@dataclass
class DatasetInfo:
    path: str = './data'
    task: str = 'bench201'
    dataset: str = 'weather'          #  
    predict_target: str = 'accuracy'  #  latency accuracy 
    
    eval_set: bool = True
    shuffle: bool = False
    scaler_method: str = 'minmax'
    spliter_ratio: str = '7:1:2'
    sample_method: str = 'ours'  # random  ours
    
    
    seq_len: int = 96
    pred_len: int = 192
    
    
@dataclass
class TrainingConfig:
    bs: int = 16
    lr: float = 0.001
    decay: float = 0.0001
    loss_func: str = 'MSELoss'  # L1Loss  MSELoss
    optim: str = 'Adam'
    epochs: int = 200
    patience: int = 30
    verbose: int = 10
    device: str = 'cuda'
    monitor_metric: str = 'NMAE'
    use_amp: bool = False
    monitor_reverse: bool = False


@dataclass
class BaseModelConfig:
    model: str = 'ours'
    d_model: int = 40
    num_layers: int = 3
    retrain: bool = True


@dataclass
class ExperimentConfig:
    seed: int = 0
    rounds: int = 5
    runid: int = 0
    debug: bool = False
    record: bool = True
    hyper_search: bool = False
    continue_train: bool = False


@dataclass
class LoggerConfig:
    logger: str = 'zyx'
    
    
@dataclass
class OtherConfig:
    classification: bool = False
    ablation: int = 0
    try_exp: int = -1
    ts_var: int = 1
    input_size: int = 1
    