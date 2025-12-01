import time
import faiss
import numpy as np
import torch
from einops import einsum
from modules.memory.base_memory import ErrorCompensation


class LSHErrorCompensation(ErrorCompensation):

    def __init__(self, config):
        super().__init__(config)
        self.clear()
        self.perfs = []

    def append(self, hidden, target, predict):
        # hidden as tensors
        hidden = hidden.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        predict = predict.detach().cpu().numpy()

        self.hiddens.append(hidden)
        self.targets.append(target)
        self.predicts.append(predict.reshape(target.shape))

    def set_ready(self):

        # Filter out error that is small
        error = np.abs(np.concatenate(self.predicts) - np.concatenate(self.targets))
        percentile = np.percentile(error, 0)
        maskIdx = error > percentile

        self.ready_hiddens = np.vstack(self.hiddens)[maskIdx]
        self.ready_targets = np.concatenate(self.targets)[maskIdx]
        self.index.train(self.ready_hiddens)
        self.index.add(self.ready_hiddens)

    def clear(self):
        self.index = faiss.IndexLSH(self.config.rank, 64)
        self.index.rotate_data = True
        self.hiddens = []
        self.targets = []
        self.predicts = []
        self.ready_hiddens = None
        self.ready_targets = None

    def correction(self, hidden):

        hidden = hidden.detach().cpu().numpy()
        start = time.time_ns()
        dists, I = self.index.search(hidden, self.config.topk)
        end = time.time_ns()
        self.perfs.append((end - start) / len(hidden))
        compensation = np.zeros(len(hidden), dtype=np.float32)
        lmdas = np.zeros_like(compensation, dtype=np.float32)
        for i in range(len(lmdas)):
            lmdas[i] = self.config.lmda
            compensation[i] = self.ready_targets[I[i, 0]]

        # if len(self.perfs) % 100 == 0:
        #     print(f'Search Time Per Query [ms]: {np.mean(self.perfs) * 1e-6:.4f} ± {np.std(self.perfs) * 1e-6:.4f}')
        #     self.perfs = []
        return torch.from_numpy(compensation).to(self.config.device), torch.from_numpy(lmdas).to(self.config.device)




