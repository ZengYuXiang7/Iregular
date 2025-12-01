# coding : utf-8
# Author : yuxiang Zeng
# 根据需要来改变这里的内容

import pickle 
import pandas as pd
import numpy as np

from data_process.ts_data import load_weather
from data_provider.data_getitem_ts import TimeSeriesDataset

def load_data(config):
    try:
        with open(f'./data/{config.dataset}_data.pkl', 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        if config.dataset == 'weather':
            data = load_weather(config)
        elif config.dataset == 'MIMIC':
            data = load_mimic(config)
            
        with open(f'./data/{config.dataset}_data.pkl', 'wb') as f:
            pickle.dump(data, f)
    return data


def get_dataset(data, split, config):
    """
    返回一个数据集实例：DatasetClass(data, split, config)
    支持的 model:
      - 'ours'               -> NASDataset
    """
    dataset = config.dataset
    split = str(split).lower()
    if split not in {"train", "valid", "test"}:
        raise ValueError(f"split must be 'train'/'valid'/'test', got: {split}")

    DatasetClass = TimeSeriesDataset

    return DatasetClass(data, split, config)
    