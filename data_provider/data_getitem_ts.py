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
        dataset_len = (
            len(self.data["x"]) - self.config.seq_len - self.config.pred_len + 1
        )

        # 0 代表不缺失 ， 1 代表缺失
        mask_matrix = np.ones((dataset_len, config.seq_len, config.input_size))
        k = int(config.seq_len * config.input_size * data_dorpout)
        # 让缺失值变成mask = 1，表示这个位置存在缺失
        for i in range(dataset_len):
            # 目前先严格控制一下丢失数，不然随机可能让这个丢失的更多
            timestamp, feature_idx = mask_matrix[i].nonzero()
            mask_matrix[i] = 0
            shuffled_idx = np.random.permutation(len(timestamp))
            timestamp, feature_idx = timestamp[shuffled_idx], feature_idx[shuffled_idx]
            timestamp, feature_idx = timestamp[:k], feature_idx[:k]
            mask_matrix[i][timestamp, feature_idx] = 1
        self.mask_matrix = mask_matrix

    def __len__(self):
        return len(self.data["x"]) - self.config.seq_len - self.config.pred_len + 1
        # return len(self.data[self.config.predict_target])

    def __getitem__(self, idx):

        if self.config.model == "ours":
            s_begin = idx
            s_end = s_begin + self.config.seq_len
            r_begin = s_end
            r_end = r_begin + self.config.pred_len

            x = self.data["x"][s_begin:s_end]

            x_mark = self.data["x_mark"][s_begin:s_end]
            x_mask = self.mask_matrix[idx]

            # 制作缺失值
            x[x_mask == 1] = 0.0

            y = self.data["y"][r_begin:r_end]
            return x, x_mark, x_mask, y

        else:
            raise ValueError(f"Unsupported model type: {self.config.model}")

    def custom_collate_fn(self, batch):
        if self.config.model == "ours":
            x, x_mark, x_mask, y = zip(*batch)
            return (
                default_collate(x).to(torch.float32),
                default_collate(x_mark).to(torch.long),
                default_collate(x_mask).to(torch.float32),
                default_collate(y).to(torch.float32),
            )
