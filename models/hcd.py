# -*- coding: utf-8 -*-
# 说明：
# 本文件实现了基于“分割-子图学习-融合”的因果发现流程。
# 主要组件：
# 1) CondIndepParCorr：条件偏相关检验（Fisher Z 近似），用于边裁剪与不加盾三元组的处理。
# 2) SAHCD：将原始图按无条件独立性进行裁剪与连通分量分割；对每个子图调用指定因果发现算法（如 ICALiNGAM/PC/NOTEARS/GOLEM 等）学习局部因果矩阵；最后按规则融合回全局因果图并评估。

# ========= 依赖导入 =========
# - scipy.stats.norm：用于 Fisher Z 统计量的尾概率计算（正态分布 CDF）。
# - castle.*：gCastle 的评估指标与多种因果发现算法实现。
# - numpy：矩阵运算与数据结构处理。
# 注意：需要已安装 gcastle（pip install -U gcastle），并在当前 Python 环境中可导入。
import time
from scipy.stats import norm
from castle.metrics import MetricsDAG
from castle.algorithms import DirectLiNGAM
from castle.algorithms import Notears
from castle.algorithms import GOLEM
import numpy as np
from castle.algorithms import PC
from itertools import chain, combinations
from castle.algorithms import ICALiNGAM
from castle.algorithms import GraNDAG

# ========= 条件独立性检验器 =========
# CondIndepParCorr 通过样本协方差（相关系数矩阵）计算在给定条件集 Z 下 X 与 Y 的偏相关系数，
# 再利用 Fisher Z 变换得到近似的显著性（双侧 p 值），用于判断是否独立。

# 用于计算 (X ⟂ Y | Z) 的统计量；初始化时预先计算整体相关系数矩阵以便多次查询。



import numpy as np

def corrcoef_stable_samples_by_cols(X: np.ndarray) -> np.ndarray:
    """
    给定 X 形状 [n, d]（n样本，d变量），返回 d×d 相关系数矩阵。
    避免使用 np.corrcoef/cov，规避某些 NumPy 版本的 returned=True bug。
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D [n, d].")
    n, d = X.shape
    if n < 2:
        raise ValueError("Need at least 2 samples.")

    # 去均值
    Xc = X - X.mean(axis=0, keepdims=True)

    # 列标准差（无偏，ddof=1），避免除零
    std = Xc.std(axis=0, ddof=1)
    zero_mask = (std == 0)
    std[zero_mask] = 1.0  # 暂时设为1，后面把对应行列置0/1

    # 标准化
    Z = Xc / std

    # 相关矩阵 = 协方差矩阵 / (std_i * std_j)；Z 已经标准化，直接做 Z^T Z / (n-1)
    C = (Z.T @ Z) / (n - 1)

    # 数值裁剪 & 对角置 1
    np.clip(C, -1.0, 1.0, out=C)
    np.fill_diagonal(C, 1.0)

    # 对于原本 std 为 0 的变量：与其它变量的相关系数应为 0，自相关置 1
    if zero_mask.any():
        idx = np.where(zero_mask)[0]
        C[idx, :] = 0.0
        C[:, idx] = 0.0
        C[idx, idx] = 1.0

    return C

class CondIndepParCorr:
    # 参数：
    # - data: 形状约为 [变量数, 样本数] 的数组（此处传入的是 data.T）；np.corrcoef 用于得到相关系数矩阵。
    # - n: 样本量，用于 Fisher Z 统计量的自由度项（n - |Z| - 3）。
    def __init__(self, data, n):
        super().__init__()
        self.correlation_matrix = np.corrcoef(data)
        # self.correlation_matrix = corrcoef_stable_samples_by_cols(data)
        self.num_records = n

    # 计算给定索引 x, y 以及条件集 zz（元组/列表）的双侧 p 值统计量：
    # 步骤：
    # 1) 根据 |Z| 情况计算偏相关系数（0：直接取 corr；1：使用二元偏相关公式；≥2：对子矩阵求广义逆 pinv 再取元素）。
    # 2) 用 Fisher Z 近似：z = log1p(2ρ / (1-ρ+1e-5))，并按 n-|Z|-3 缩放获得统计量。
    # 3) 以标准正态分布计算双侧尾概率，返回 p 值（越小越相关，越大越接近独立）。
    def calc_statistic(self, x, y, zz):
        corr_coef = self.correlation_matrix
        if len(zz) == 0:
            par_corr = corr_coef[x, y]
        elif len(zz) == 1:
            z = zz[0]
            par_corr = (corr_coef[x, y] - corr_coef[x, z] * corr_coef[y, z]) / np.sqrt(
                (1 - np.power(corr_coef[x, z], 2)) * (1 - np.power(corr_coef[y, z], 2))
            )
        else:  # zz contains 2 or more variables
            all_var_idx = (x, y) + zz
            corr_coef_subset = corr_coef[np.ix_(all_var_idx, all_var_idx)]
            # consider using pinv instead of inv
            inv_corr_coef = -np.linalg.pinv(corr_coef_subset)
            par_corr = inv_corr_coef[0, 1] / np.sqrt(
                abs(inv_corr_coef[0, 0] * inv_corr_coef[1, 1])
            )
        z = np.log1p(2 * par_corr / (1 - par_corr + 1e-5))
        val_for_cdf = abs(np.sqrt(self.num_records - len(zz) - 3) * 0.5 * z)
        statistic = 2 * (1 - norm.cdf(val_for_cdf))
        return statistic


# ========= SAHCD 主流程 =========
# 思路：
# 1) seperate_data：基于无条件独立检验先对全图进行边裁剪，并对不加盾三元组做“V 结构（collider）”处理，得到稀疏图。
# 2) get_seperate_sets：将裁剪后的图分解为互不相连的连通子图，形成子图节点集合族。
# 3) 对每个子图调用指定的因果发现算法（如 ICALiNGAM/PC/NOTEARS/GOLEM 等）学习局部因果矩阵。
# 4) 将子图结果按“碰撞点集合（colider_set）”与打分规则融合回全局图（global_graph），并统计时间与评估指标。
class HCD:
    """
    self.data:Observation data
    self.Ture_data:ground_truth
    self.n:The number of samples
    self.dim:The number of Vertices
    self.cipc:FisherZ
    self.pre_set_gate:The threshold of fisherz
    self.global_graph:The global graph.
    self.colider_set:A set of colliders shared by subgraphs
    self.avg_time:Average time costed by each sub-graph
    self.max_time:Max time costed by each sub-graph
    self.args:Hyperparameters
    """

    # 初始化：
    # - data / True_data：观测数据与真值因果图（用于评估）。
    # - n / dim：样本量与变量个数。
    # - cipc：条件偏相关检验器（基于 data.T）。
    # - pre_set_gate：Fisher Z 的阈值（p 值阈），用于无条件裁剪。
    # - global_graph：全局邻接矩阵（先设为全 1，对角置 0），后续会按检验与子图结果增减权重。
    # - sepsets/colider_set：子图集合与碰撞点集合（最终为所有子图共享的交集）。
    def __init__(self, pre_gate=0.80, thresh=0.3, method="ICALiNGAM"):
        self.pre_gate = pre_gate
        self.thresh = thresh
        self.method = method

    # 子图切分：无条件独立裁剪 + 不加盾三元组的“V 结构”处理，再提取连通分量集合
    def seperate_data(self):
        # 步骤1：对每一对变量 (i,j) 做无条件独立检验（Z=∅）。若 p 值大于阈值，则认为独立，删除双向边。
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                i_j_conditions = tuple([])
                tmp = self.cipc.calc_statistic(i, j, i_j_conditions)
                # print(tmp)
                if tmp > self.pre_set_gate:
                    # print(tmp)
                    self.global_graph[i, j] = 0
                    self.global_graph[j, i] = 0

        # 步骤2：对每个不加盾三元组 (j - i - k) 且 j,k 之间无边的情况，将 i 指向 j、k 的边删除（保留潜在 “j -> i <- k” 的 V 结构可能）。
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    if (
                        self.global_graph[j, i] + self.global_graph[i, j] > 0
                        and self.global_graph[k, i] + self.global_graph[i, k] > 0
                        and self.global_graph[j, k] + self.global_graph[k, j] == 0
                        and i != j
                        and j != k
                        and i != k
                    ):
                        self.global_graph[i, j] = 0
                        self.global_graph[i, k] = 0

        seperate_sets = self.get_seperate_sets(self.global_graph)
        return seperate_sets

    # 提取连通子图集合：
    # 先通过 BFS/扩张找到每个连通分量（seperate_sets），再去除被包含的子集，得到最大子图集合（final_sets）。
    def get_seperate_sets(self, graph):

        seperate_sets = []
        hashset = np.zeros((self.dim))
        # 遍历所有节点，按可达关系扩张，收集每个连通分量的节点集合
        for i in range(self.dim):
            if hashset[i] == 0:
                hashset[i] = 1
                temp_set = set()
                _temp_set = set()
                _temp_set.add(i)
                while temp_set != _temp_set:
                    temp_set = _temp_set.copy()
                    for j in temp_set:
                        hashset[j] = 1
                        for k in range(self.dim):
                            if graph[j][k] == 1 and k not in temp_set:
                                _temp_set.add(k)
                seperate_sets.append(temp_set)

        final_sets = []

        # 过滤掉被其它集合完全包含的子集，仅保留最大集合，避免重复学习
        for i in seperate_sets:
            flag = 0
            for j in seperate_sets:
                if i == j:
                    continue
                if i.issubset(j):
                    flag = 1
                    break
            if flag == 0:
                final_sets.append(i)

        return final_sets

    # 子图学习与融合：
    # 输入：
    # - data：子图的数据子集（shape: [n_samples, n_nodes_sub]）
    # - node：子图节点在全局图中的索引列表
    # 流程：
    # 1) 按参数选择子图因果算法并训练，得到 sub_model.causal_matrix。
    # 2) 统计训练耗时（avg/max）。
    # 3) 将子图的有向边按规则映射回 global_graph：若端点不在公共碰撞点集合，置信度权重变化更大；端点在集合内则惩罚/奖励幅度不同（体现碰撞点优先级）。
    def get_Sep_IP(self, data, node):
        node = np.array(node)
        nums = data.shape[1]
        n = 1 << nums
        method = self.method
        # 选择并实例化子图因果发现算法（按超参数设置）：
        # - ICALiNGAM / DirectLiNGAM：线性非高斯模型，基于独立成分/回归残差求解有向无环图。
        # - PC：基于条件独立性的约束法，alpha 为显著性水平。
        # - Notears：基于光滑无环约束的优化法，w_threshold 为阈值。
        # - GraNDAG：基于梯度的神经网络参数化 DAG 学习（需 GPU）。
        # - GOLEM：基于可扩展优化的线性/高斯假设方法（支持阈值与学习率等设置）。

        """
            config.thresh = 0.3
            config.pc_alpha = 0.05
            config.causal_lr = 0.05
            pc_alpha, golem_epoch, lr, device_type
        """
        if method == "PC":
            sub_model = PC(alpha=0.05)
        if method == "ICALiNGAM":
            sub_model = ICALiNGAM(thresh=self.thresh)
        if method == "DirectLiNGAM":
            sub_model = DirectLiNGAM(thresh=self.thresh)
        if method == "Notears":
            sub_model = Notears(w_threshold=self.thresh)
        if method == "GraNDAG":
            sub_model = GraNDAG(input_dim=data.shape[1], device_type="gpu")
        if method == "GOLEM":
            sub_model = GOLEM(
                GOLEM(
                    num_iter=5000,
                    graph_thres=self.thresh,
                    device_type="gpu",
                    learning_rate=0.05,
                )
            )

        # 训练子图模型，学习邻接（因果）矩阵；不同算法统一暴露 causal_matrix 属性
        start = time.time()
        sub_model.learn(data)
        cost_time = time.time() - start
        self.avg_time += cost_time
        self.max_time = max(self.max_time, cost_time)

        # 将子图的边评分累加到 global_graph：
        # - 若任一端点不在 colider_set：存在边则 +10，不存在则 -10（置信度变化较大）。
        # - 若两端都在 colider_set：存在边则 +10000，不存在则 -1（对碰撞点结构更敏感）。
        # 注意：最终会对 global_graph 做二值化（>0 置 1），因此这里是“投票加权”的思想。
        for i in range(nums):
            for j in range(nums):
                if i not in self.colider_set or j not in self.colider_set:
                    if sub_model.causal_matrix[i][j] == 0:
                        self.global_graph[node[i]][node[j]] -= 10
                    else:
                        self.global_graph[node[i]][node[j]] += 10
                else:
                    if sub_model.causal_matrix[i][j] == 0:
                        self.global_graph[node[i]][node[j]] -= 1
                    else:
                        self.global_graph[node[i]][node[j]] += 10000

    def learn(self, data):
        self.n = data.shape[0]
        self.dim = data.shape[1]
        self.cipc = CondIndepParCorr(data.T, self.n)
        self.IPset = []
        self.pre_set_gate = self.pre_gate
        self.Sepset = []
        self.global_graph = np.ones(
            (
                self.dim,
                self.dim,
            ),
            dtype=int,
        )
        self.sepsets = []
        self.colider_set = {}
        self.avg_time = 0
        self.max_time = 0
        for i in range(self.dim):
            self.global_graph[i][i] = 0

        self.data = np.array(data)

        # 先进行无条件裁剪与三元组处理，得到子图节点集合族
        seperatesets = self.seperate_data()
        
        # print(seperatesets)
        # print(len(seperatesets))
        
        # 计算所有子图集合的交集（公共碰撞点集合），并追加记录到 setsinf.csv
        res = seperatesets[0]
        for i in seperatesets:
            if i == seperatesets[0]:
                continue
            res = res.intersection(i)

        self.sepsets = seperatesets
        self.colider_set = res

        # 遍历每个子图：抽取对应列的数据，学习局部因果图并融合权重到 global_graph  并行
        for set in seperatesets:
            data = self.data[:, list(set)]
            self.get_Sep_IP(data, list(set))

        self.global_graph = np.int64(self.global_graph > 0)

        # 骨架矩阵（忽略方向，仅判断是否存在边）用于评估
        # True_data = np.array(True_data)
        # skeleton_truth = np.int64(self.True_data != 0)
        # final = MetricsDAG(self.global_graph, skeleton_truth)
        # print(final.metrics)
        self.avg_time = self.avg_time / len(seperatesets)

        self.causal_matrix = self.global_graph
        return self.global_graph
