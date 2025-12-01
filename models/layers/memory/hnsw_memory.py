import time
import faiss
import numpy as np
import torch
from einops import einsum
from modules.memory.base_memory import ErrorCompensation


class IndexHamHNSW:

    def __init__(self, config):
        nbits = 64
        self.encoder = faiss.IndexLSH(config.rank, nbits)
        self.innerIndex = faiss.IndexBinaryHNSW(nbits, 32)

    def train(self, hidden):
        self.encoder.train(hidden)

    def add(self, hidden):
        codes = self.encoder.sa_encode(hidden)
        self.innerIndex.train(codes)
        self.innerIndex.add(codes)

    def search(self, hidden, topk):
        codes = self.encoder.sa_encode(hidden)
        return self.innerIndex.search(codes, topk)


class IndexHamHNSWErrorCompensation(ErrorCompensation):

    def __init__(self, config):
        super().__init__(config)
        self.clear()
        self.perfs = []

    def append(self, hidden, target, predict):
        # hidden as tensors
        hidden = hidden.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        self.hiddens.append(hidden)
        self.targets.append(target)
        # self.hiddens[self.idx:self.idx+len(hidden)] = hidden
        # self.targets[self.idx:self.idx+len(target)] = target
        # self.idx += len(hidden)

    def set_ready(self):
        self.ready_hiddens = np.vstack(self.hiddens)
        self.ready_targets = np.concatenate(self.targets)

        t1 = time.time()
        self.index.train(self.ready_hiddens)
        self.index.add(self.ready_hiddens)
        t2 = time.time()
        print('index train and add time:', t2 - t1)
        # self.index.train(self.hiddens)
        # self.index.add(self.hiddens)

    def set_start(self, train_length):
        # self.hiddens = np.zeros((train_length, self.config.rank))
        # self.targets = np.zeros((train_length, ))
        # self.idx = 0
        return True

    def clear(self):
        self.index = IndexHamHNSW(self.config)
        self.hiddens = []
        self.targets = []
        self.ready_hiddens = None
        self.ready_targets = None

    def correction(self, hidden, k):
        hidden = hidden.detach().cpu().numpy()
        dists, I = self.index.search(hidden, k)
        compensation = np.zeros(len(hidden), dtype=np.float32)
        lmdas = np.zeros_like(compensation, dtype=np.float32)
        for i in range(len(lmdas)):
            lmdas[i] = self.config.lmda
            dist = dists[i]
            mask = dist <= (dist[0] + dist[-1]) // 2
            selected = self.ready_targets[I[i, :k]][mask]

            if selected.size > 0:  # 确保有可用数据
                compensation[i] = np.sum(selected.astype(np.float32)) / selected.size
            else:
                compensation[i] = 0.0  # 处理无效补偿情况
        return torch.from_numpy(compensation).to(self.config.device), torch.from_numpy(lmdas).to(self.config.device)

    # def correction(self, hidden, k):
    #     hidden = hidden.detach().cpu().numpy()
    #     dists, I = self.index.search(hidden, k)  # shape: [B, k]
    #
    #     # shape: [B], 默认补偿值和 lambda 值
    #     compensation = np.zeros(len(hidden), dtype=np.float32)
    #     lmdas = np.full(len(hidden), self.config.lmda, dtype=np.float32)
    #
    #     # 计算阈值：dist[0] + dist[-1] // 2 每一行一个
    #     thresholds = (dists[:, 0] + dists[:, -1]) // 2
    #     mask = dists <= thresholds[:, np.newaxis]  # shape: [B, k]
    #
    #     # 获取所有 target（shape: [B, k]）
    #     target_candidates = self.ready_targets[I]
    #
    #     # 将掩码作用到 target 上
    #     masked = np.where(mask, target_candidates, np.nan)  # 无效位置设为 nan
    #     with np.errstate(invalid='ignore'):
    #         compensation = np.nanmean(masked, axis=1)  # shape: [B]
    #
    #     # 把 nan 变成 0.0（避免无选中时出 nan）
    #     compensation = np.nan_to_num(compensation, nan=0.0)
    #
    #     return (
    #         torch.from_numpy(compensation).to(self.config.device),
    #         torch.from_numpy(lmdas).to(self.config.device)
    #     )