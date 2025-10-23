
from torch.utils.data import Sampler
import numpy as np 
from tqdm import *
import torch 



class FixedLengthBatchSampler(Sampler):
    def __init__(self, data_source, dataset, batch_size, include_partial=True, seed=2024, maxlen=None, length_to_size=None, config=None):
        """
        :param data_source: 数据集对象 (要求能通过索引访问样本，且 __len__ 返回长度)
        :param dataset: 数据集名称 (这里没实际用到，只是历史代码遗留)
        :param batch_size: 默认批大小
        :param include_partial: 是否包含不足 batch_size 的“残余批”
        :param rng: 随机数生成器 (用于打乱样本顺序)，默认 seed=11
        :param maxlen: 最大允许长度 (超过该长度的样本会被丢弃)
        :param length_to_size: 可选字典 {长度: 批大小}，用来实现“不同长度→不同批大小”的策略
        """
        self.config = config
        self.data_source = data_source
        self.dataset = dataset
        self.active = False
        self.rng = np.random.RandomState(seed=seed)
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.include_partial = include_partial
        self.length_to_size = length_to_size

        # 缓存，避免重复计算 (键为长度，值为对应的批大小)
        self._batch_size_cache = {0: self.batch_size}

        # 构建 {长度: [样本索引]} 的映射
        self.length_map = self.get_length_map()

        # 初始化内部状态
        self.reset()
        
        
    def get_length_map(self):
        """
        遍历 dataset，按序列长度分桶。
        返回: {长度: [样本索引列表]}
        """
        length_map = {}
        for i in trange(len(self.data_source)):
            # 假设 dataset[i][0] 是张量，第一维是序列长度
            
            if self.config.model in ['ours', 'gat']:
                data = self.data_source[i][1]
            else:
                data = self.data_source[i][0]
            length = len(data)

            # 丢弃超过 maxlen 的样本
            if self.maxlen is not None and self.maxlen > 0 and length > self.maxlen:
                continue

            # 把样本索引 i 放入对应长度的桶
            length_map.setdefault(length, []).append(i)
        return length_map

    def get_batch_size(self, length):
        """
        根据样本长度决定 batch size。
        如果未提供 length_to_size，则返回默认 batch_size。
        """
        if self.length_to_size is None:
            return self.batch_size

        # 如果缓存里有直接返回
        if length in self._batch_size_cache:
            return self._batch_size_cache[length]

        # 否则从已知的最大长度向上推算
        start = max(self._batch_size_cache.keys())
        batch_size = self._batch_size_cache[start]

        # 从 start+1 到 length 逐一检查是否有映射，更新 batch_size
        for n in range(start + 1, length + 1):
            if n in self.length_to_size:
                batch_size = self.length_to_size[n]
            self._batch_size_cache[n] = batch_size
        return batch_size

    def reset(self):
        """
        重置采样器状态：
        1. 每个长度桶内部打乱顺序
        2. 计算每个桶能分多少整批 (nbatches) 和余数 (surplus)
        3. 构建 batch 顺序列表 order (保存每批对应的长度)
        4. 若 include_partial=True 且余数满足条件，也加入残余批
        5. 打乱 order
        """
        # 1. 桶内打乱
        for length in self.length_map.keys():
            self.rng.shuffle(self.length_map[length])

        # 2. 记录每个长度桶的状态
        state = {}
        for length, arr in self.length_map.items():
            batch_size = self.get_batch_size(length)
            nbatches = len(arr) // batch_size     # 能切多少整批
            surplus = len(arr) % batch_size       # 余数
            state[length] = dict(nbatches=nbatches, surplus=surplus, position=-1)

        # 3. 构建 order 列表，每个长度重复 nbatches 次
        order = []
        for length, v in state.items():
            order += [length] * v['nbatches']

        # 4. 如果 include_partial=True，加入残余批
        if self.include_partial:
            for length, v in state.items():
                # 注意：这里逻辑比较奇怪，只有 surplus >= GPU数量 时才加
                # 一般来说应当 surplus>0 就能加
                if v['surplus'] >= torch.cuda.device_count():
                    order += [length]

        # 5. 打乱批次顺序
        self.rng.shuffle(order)

        # 保存状态
        self.state = state
        self.order = order
        self.index = -1

    def get_next_batch(self):
        """
        根据当前 index，取出下一个 batch 的索引列表。
        """
        index = self.index + 1
        length = self.order[index]               # 本批对应的序列长度
        batch_size = self.get_batch_size(length)
        position = self.state[length]['position'] + 1

        # 计算 batch 在该桶中的起止位置
        start = position * batch_size
        batch_index = self.length_map[length][start:start + batch_size]

        # 更新状态
        self.state[length]['position'] = position
        self.index = index
        return batch_index

    def __iter__(self):
        """
        迭代时会自动 reset，并逐批产出索引列表
        """
        self.reset()
        for _ in range(len(self)):
            yield self.get_next_batch()

    def __len__(self):
        """
        返回总批次数 (整批数+可选残余批)
        """
        return len(self.order)