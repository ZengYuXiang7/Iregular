# coding : utf-8
# Author : yuxiang Zeng
# 根据场景需要来改这里的input形状
from torch.utils.data import Dataset
import torch
import numpy as np
from tqdm import *
from torch.utils.data.dataloader import default_collate

class TimeSeriesDataset(Dataset):
    def __init__(self, data, mode, config):
        self.config = config
        self.data = data
        self.mode = mode
        
        # 生成不规则采样，采用掩码居住
        # random_matrix = torch.ones(train_len, 96, 21)
        data_dorpout = config.data_dropout
        dataset_len = len(self.data['x']) - self.config.seq_len - self.config.pred_len + 1
        mask_matrix = np.ones((dataset_len, config.seq_len, config.input_size))
        for i in range(dataset_len):
            mask = np.random.randn(config.seq_len, config.input_size)
            mask[mask <= data_dorpout] = 0
            mask_matrix[i] = mask
            
        self.mask_matrix = mask_matrix

    def __len__(self):
        return len(self.data['x']) - self.config.seq_len - self.config.pred_len + 1
        # return len(self.data[self.config.predict_target])

    def __getitem__(self, idx):
        
        if self.config.model == 'ours':
            s_begin = idx
            s_end = s_begin + self.config.seq_len
            r_begin = s_end
            r_end = r_begin + self.config.pred_len
            
            x = self.data['x'][s_begin:s_end]
            x_mark = self.data['x_mark'][s_begin:s_end]
            x_mask = self.mask_matrix[idx]
            
            y = self.data['y'][r_begin:r_end]
            return x, x_mark, x_mask, y
        
        
        else: 
            raise ValueError(f"Unsupported model type: {self.config.model}")
        
        
    def custom_collate_fn(self, batch):
        if self.config.model == 'ours':
            x, x_mark, x_mask, y = zip(*batch)
            return default_collate(x).to(torch.float32), default_collate(x_mark).to(torch.long), default_collate(x_mask).to(torch.float32), default_collate(y).to(torch.float32)
        

