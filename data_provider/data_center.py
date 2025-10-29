# coding : utf-8
# Author : Yuxiang Zeng
# 注意，这里的代码已经几乎完善，非必要不要改动（2025年06月22日15:57:27）
import platform
import multiprocessing
from torch.utils.data import DataLoader
from data_provider.data_control import get_dataset, load_data
import pickle
import torch
from data_provider.data_scaler import get_scaler
import numpy as np
import dgl
from tqdm import *

from data_provider.data_smapler import FixedLengthBatchSampler


# 数据集定义
class DataModule:
    def __init__(self, config):
        self.config = config
        self.path = config.path
        self.data = load_data(config)

        if config.debug:
            self.data = take_subset(
                self.data, ratio=0.1, seed=config.seed, random_sample=False
            )

        self.train_data, self.valid_data, self.test_data = self.get_split_dataset(
            self.data, config
        )

        self.x_scaler, self.y_scaler = self.get_scalers(self.train_data, config)
        self.train_data = self.normalize_data(
            self.train_data, self.x_scaler, self.y_scaler, config
        )
        self.valid_data = self.normalize_data(
            self.valid_data, self.x_scaler, self.y_scaler, config
        )
        self.test_data = self.normalize_data(
            self.test_data, self.x_scaler, self.y_scaler, config
        )

        self.train_set = get_dataset(self.train_data, "train", config)
        self.valid_set = get_dataset(self.valid_data, "valid", config)
        self.test_set = get_dataset(self.test_data, "test", config)

        self.train_loader = self.build_loader(
            self.train_set, bs=config.bs, is_train=True
        )
        self.valid_loader = self.build_loader(
            self.valid_set, bs=config.bs, is_train=False
        )
        self.test_loader = self.build_loader(
            self.test_set, bs=config.bs, is_train=False
        )
        config.log.only_print(
            f"Train_length : {len(self.train_loader.dataset)} Valid_length : {len(self.valid_loader.dataset)} Test_length : {len(self.test_loader.dataset)}"
        )

    def get_split_dataset(self, data, config):
        """
        仅切分，不做归一化。
        输入:
            - data: dict[str -> list]，各字段第一维长度一致
            - config: 需包含 spliter_ratio（如 '8:1:1'），可选 seed
        返回:
            - train_data, valid_data, test_data  (与 data 同结构的 dict，值是切片后的 list)
        """
        assert isinstance(data, dict) and len(data) > 0, "data 必须是非空 dict"

        # 确保全是 list，并校验样本数一致
        N = None
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                v = v.tolist()  # 转成 list
                data[k] = v
            elif not isinstance(v, list):
                v = list(v)  # 其他类型也转 list
                data[k] = v

            if N is None:
                N = len(v)
            else:
                assert (
                    len(v) == N
                ), f"字段 {k} 的样本数 {len(v)} 与其他字段不一致（应为 {N})"

        # 解析比例
        ratio_str = config.spliter_ratio
        parts = list(map(int, ratio_str.strip().split(":")))
        total = sum(parts)
        tr, vr, te = [p / total for p in parts]

        idx = list(range(N))
        n_train = int(N * tr)
        n_valid = int(N * vr)

        # 随机打乱索引, 但是时序不允许随机
        # rng = np.random.default_rng(config.seed)
        # rng.shuffle(idx)

        train_idx = idx[:n_train]
        valid_idx = idx[n_train : n_train + n_valid]
        test_idx = idx[n_train + n_valid :]

        # 按索引切片为三个 dict
        def slice_dict(d, indices):
            out = {}
            for k, v in d.items():
                out[k] = [v[i] for i in indices]  # 保持 list
            return out

        train_data = slice_dict(data, train_idx)
        valid_data = slice_dict(data, valid_idx)
        test_data = slice_dict(data, test_idx)

        return train_data, valid_data, test_data

    def get_scalers(self, data: dict, config):
        """
        为指定的字段训练归一化器，并以 dict 形式返回。

        输入:
            data: dict[int -> sample_dict]，例如 {0: {"flops":..., "params":..., "accuracy":...}, ...}
            keys: 需要归一化的字段名
        返回:
            scalers: dict[str -> StandardScaler]
        """
        x_keys = "x"
        x_scaler = {}
        for key in x_keys:
            x_scaler[key] = get_scaler(
                np.array(data[key]), config, selected_method="stander"
            )

        values = np.array(np.array(data[config.predict_target]), dtype=np.float32)
        y_scaler = get_scaler(values, config, selected_method="stander")
        return x_scaler, y_scaler

    def normalize_data(self, data, x_scaler, y_scaler, config):
        """
        对数据进行归一化处理。

        输入:
            data: dict[int -> sample_dict]，例如 {0: {"flops":..., "params":..., "accuracy":...}, ...}
            x_scaler: dict[str -> StandardScaler]，用于归一化输入特征
            y_scaler: StandardScaler，用于归一化目标值
        返回:
            normalized_data: dict[int -> sample_dict]，归一化后的数据
        """
        normalized_data = {}
        for key, value in data.items():
            if key in x_scaler:
                value = np.array(value, dtype=np.float32)
                normalized_data[key] = x_scaler[key].transform(value)
            elif key == config.predict_target and y_scaler:
                value = np.array(value, dtype=np.float32)
                normalized_data[key] = y_scaler.transform(value)
            else:
                normalized_data[key] = np.array(value)
        return normalized_data

    def build_loader(self, dataset, bs, is_train):
        if platform.system() == "Linux" and "ubuntu" in platform.version().lower():
            max_workers = min(multiprocessing.cpu_count() // 3, 12)
            prefetch_factor = 4  # 可调至 6，建议不要超过 8
        else:
            max_workers = 0
            prefetch_factor = None

        return DataLoader(
            dataset,
            batch_size=bs,
            shuffle=is_train,  # 是否需要打乱，视你实验需求决定
            pin_memory=True,
            num_workers=max_workers,
            prefetch_factor=prefetch_factor,
            collate_fn=lambda batch: dataset.custom_collate_fn(batch),
        )

        # if self.config.dataset in ['101_acc', '201_acc']:
        #     # 不需要 sampler，直接用 batch_size
        #     return DataLoader(
        #     dataset,
        #     batch_size=bs,
        #     shuffle=is_train,   # 是否需要打乱，视你实验需求决定
        #     pin_memory=True,
        #     num_workers=max_workers,
        #     prefetch_factor=prefetch_factor,
        #     collate_fn=lambda batch: dataset.custom_collate_fn(batch),
        # )
        # else:
        #     # 其他数据集需要 sampler
        #     sampler = FixedLengthBatchSampler(
        #         data_source=dataset,
        #         dataset='',
        #         batch_size=bs,
        #         include_partial=True,
        #         config=self.config,
        #         seed=self.config.seed,
        #     )
        #     return DataLoader(
        #         dataset,
        #         batch_sampler=sampler,
        #         pin_memory=True,
        #         num_workers=max_workers,
        #         prefetch_factor=prefetch_factor,
        #         collate_fn=lambda batch: dataset.custom_collate_fn(batch),
        #     )


def take_subset(obj, ratio=0.1, seed=0, random_sample=False):
    """
    对所有字段使用同一组索引切片，保证样本对齐：
    - dict[str -> np.ndarray/torch.Tensor/list/tuple]: 统一取前 ratio 部分（或随机抽 ratio 部分）
    - 其他类型：按第0维或序列长度切片
    """

    if ratio is None or ratio >= 1.0:
        return obj

    # 统一做切片的小工具
    def _slice_by_idx(x, idx):
        if isinstance(x, np.ndarray):
            return x[idx]
        if isinstance(x, torch.Tensor):
            return x.index_select(0, torch.as_tensor(idx, device=x.device))
        if isinstance(x, (list, tuple)):
            sel = [x[i] for i in idx]
            return type(x)(sel)
        # 能否用通用索引
        try:
            return x[idx]
        except Exception:
            return x  # 不支持索引就原样返回

    # 计算 n
    def _calc_n(length):
        return max(1, int(length * float(ratio)))

    # dict：对齐切片（核心需求）
    if isinstance(obj, dict):
        # 取样本总数 N（要求各字段第一维一致）
        keys = list(obj.keys())
        assert len(keys) > 0, "空字典无法取子集"
        # 找到第一个可取 len 的字段
        N = None
        for k in keys:
            try:
                N = len(obj[k])
                break
            except Exception:
                continue
        assert N is not None and N > 0, "字典里没有可按第0维切片的字段"

        # 校验第一维一致（尽量只在常见类型上检查）
        for k in keys:
            v = obj[k]
            try:
                lv = len(v)
                assert lv == N, f"字段 {k} 的样本数 {lv} 与其他字段不一致（应为 {N}）"
            except Exception:
                # 无法取 len 的跳过一致性检查
                pass

        n = _calc_n(N)
        if random_sample:
            rng = np.random.default_rng(seed)
            idx = rng.choice(N, size=n, replace=False)
            idx.sort()  # 可选：保持原顺序
        else:
            idx = np.arange(n)

        # 对每个字段按同一 idx 切片
        out = {}
        for k in keys:
            out[k] = _slice_by_idx(obj[k], idx)
        return out

    # 非 dict：按自身长度切片
    try:
        N = len(obj)
    except Exception:
        return obj

    n = _calc_n(N)
    if random_sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N, size=n, replace=False)
        idx.sort()
    else:
        idx = np.arange(n)
    return _slice_by_idx(obj, idx)
