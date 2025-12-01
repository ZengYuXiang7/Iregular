import time
import faiss
import numpy as np
import torch
from einops import einsum
from models.layers.memory.base_memory import ErrorCompensation


class L2ErrorCompensation(ErrorCompensation):

    def __init__(self, config):
        super().__init__(config)
        self.clear()
        self.perfs = []

    def clear(self):
        self.index = faiss.IndexFlatL2(self.config.rank)
        self.index.rotate_data = True
        self.hiddens = []
        self.targets = []
        self.predicts = []
        self.ready_hiddens = None
        self.ready_targets = None

    def append(self, hidden, target, predict):
        # 2025年11月26日16:22:59 时序特判
        b, l, d = hidden.shape
        hidden = hidden.reshape(b, l * d)

        # hidden as tensors
        hidden = hidden.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        predict = predict.detach().cpu().numpy()

        self.hiddens.append(hidden)
        self.targets.append(target)
        self.predicts.append(predict.reshape(target.shape))

    def set_ready(self):
        # Filter out error that is small
        error = np.abs(
            np.concatenate(self.predicts, axis=0) - np.concatenate(self.targets, axis=0)
        ).sum(axis=1)
        percentile = np.percentile(error, 0)
        maskIdx = error > percentile
        maskIdx = maskIdx.squeeze(-1)

        # Ready
        self.ready_hiddens = np.concatenate(self.hiddens, axis=0)[maskIdx]
        self.ready_targets = np.concatenate(self.targets, axis=0)[maskIdx]
        print(
            f"Memory is Ready, hiddens.shape = [{self.ready_hiddens.shape}], targets.shape = [{self.ready_targets.shape}]"
        )

        # Construct the memory
        self.index.train(self.ready_hiddens)
        self.index.add(self.ready_hiddens)

    def correction(self, hiddens, preds, mode):
        hiddens = hiddens.detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()

        # 时序预测
        preds = preds.squeeze(-1)

        # Time cost record
        start = time.time_ns()
        dists, topk_idx = self.index.search(hiddens, self.config.topk)
        end = time.time_ns()
        self.perfs.append((end - start) / len(hiddens))

        # ======= 向量化部分：一次性找出 top-K target 并求均值 =======
        # self.ready_targets: [num_train, D ...]
        # topk_idx: [N, K]
        # -> y_topk: [N, K, D ...]
        y_topk = self.ready_targets[topk_idx]

        # 沿着 K 维求均值：axis=1
        # 得到: [N, D ...]，与 preds_np.shape 一致
        if mode == "mean":
            y_correct = y_topk.mean(axis=1).squeeze(-1)
            y_correct = y_correct.astype(np.float32)

        elif mode == "sum":
            weights = dists / dists.sum(axis=-1, keepdims=True)
            w_expanded = weights[:, :, None, None]  # 或 weights[..., None, None]
            weighted = w_expanded * y_topk
            y_correct = weighted.sum(axis=1).astype(np.float32)
            y_correct = y_correct.squeeze(-1)

        elif mode == "softmax":
            # softmax(-d)：距离越小，权重越大
            # 数值稳定的 softmax
            scores = dists - dists.max(axis=-1, keepdims=True)  # 防止 exp 溢出
            x_exp = np.exp(scores)  # [N, K]
            weights = x_exp / (x_exp.sum(axis=-1, keepdims=True) + 1e-8)  # [N, K]

            # 和 sum 分支一样，扩展 + 加权求和
            w_expanded = weights[:, :, None, None]  # [N, K, 1, 1]
            weighted = w_expanded * y_topk  # [N, K, 192, 1]

            y_correct = weighted.sum(axis=1).astype(np.float32)  # [N, 192, 1]
            y_correct = y_correct.squeeze(-1)  # [N, 192]

        # 补偿项
        compensation = torch.from_numpy(y_correct).to(self.config.device)

        # λ 向量：和 compensation 形状一致，每个位置都是 self.config.lmda
        lmdas = torch.full_like(
            compensation, float(self.config.lmda), device=self.config.device
        )

        return compensation, lmdas


if __name__ == "__main__":
    print(1)
