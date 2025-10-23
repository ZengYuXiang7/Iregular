# models/sada.py
# SADA (Split-and-Merge) — class wrapper that matches .learn() / .causal_matrix API
from __future__ import annotations
import numpy as np
import networkx as nx
from scipy.stats import norm
from typing import Iterable, List, Optional, Set, Tuple


# ---------- 相关/偏相关 + Fisher-Z ----------
def _corrcoef_stable_samples_by_cols(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    std = Xc.std(axis=0, ddof=1)
    zero = std == 0
    std[zero] = 1.0
    Z = Xc / std
    C = (Z.T @ Z) / (n - 1)
    np.clip(C, -1.0, 1.0, out=C)
    np.fill_diagonal(C, 1.0)
    if zero.any():
        idx = np.where(zero)[0]
        C[idx, :] = 0.0
        C[:, idx] = 0.0
        C[idx, idx] = 1.0
    return C


class _FisherZCI:
    def __init__(self, X: np.ndarray):
        self.X = np.asarray(X, dtype=float)
        self.n, self.d = self.X.shape
        self.C = _corrcoef_stable_samples_by_cols(self.X)

    def _partial_corr(self, i: int, j: int, Z: Iterable[int]) -> float:
        Z = tuple(sorted(set(Z)))
        if len(Z) == 0:
            return self.C[i, j]
        if len(Z) == 1:
            k = Z[0]
            num = self.C[i, j] - self.C[i, k] * self.C[j, k]
            den = np.sqrt((1 - self.C[i, k] ** 2) * (1 - self.C[j, k] ** 2))
            return np.clip(num / (den + 1e-12), -1.0, 1.0)
        idx = (i, j) + Z
        S = self.C[np.ix_(idx, idx)]
        P = -np.linalg.pinv(S)
        return np.clip(P[0, 1] / np.sqrt(abs(P[0, 0] * P[1, 1]) + 1e-12), -1.0, 1.0)

    def pval(self, i: int, j: int, Z: Iterable[int]) -> float:
        Z = tuple(set(Z))
        dof = self.n - len(Z) - 3
        if dof <= 0:
            return 0.0
        r = np.clip(self._partial_corr(i, j, Z), -0.999999, 0.999999)
        z = 0.5 * np.log((1 + r) / (1 - r))
        stat = np.sqrt(dof) * abs(z)
        return 2 * (1 - norm.cdf(stat))


# ---------- cut 搜索（启发式近似） ----------
def _find_min_separator(
    ci: _FisherZCI, u: int, v: int, V: Set[int], alpha: float, max_cond: int = 3
) -> Set[int]:
    rest = list(sorted(V - {u, v}))
    if ci.pval(u, v, []) > alpha:
        return set()
    Z: Set[int] = set()
    improved = True
    while improved and len(Z) < max_cond:
        improved = False
        best_w, best_p = None, -1.0
        for w in rest:
            if w in Z:
                continue
            p = ci.pval(u, v, Z | {w})
            if p > best_p:
                best_p, best_w = p, w
        curr = ci.pval(u, v, Z)
        if best_w is not None and (best_p > alpha or best_p > curr):
            Z.add(best_w)
            improved = True
    return Z


def _find_causal_cut(
    X: np.ndarray, alpha: float, k: int, max_cond: int, rng: np.random.Generator
) -> Tuple[Set[int], Set[int], Set[int]]:
    n, d = X.shape
    ci = _FisherZCI(X)
    V = set(range(d))
    best, best_score = None, -1
    pairs = set()
    while len(pairs) < k:
        u, v = rng.choice(d, size=2, replace=False)
        if u > v:
            u, v = v, u
        pairs.add((u, v))
    for u, v in pairs:
        C = set(_find_min_separator(ci, u, v, V, alpha, max_cond))
        V1, V2 = {u}, {v}
        remaining = list(sorted(V - V1 - V2 - C))
        for w in remaining:
            cond = tuple(sorted(C))
            p1 = min(ci.pval(w, x, cond) for x in V1) if V1 else 1.0
            p2 = min(ci.pval(w, x, cond) for x in V2) if V2 else 1.0
            if p1 > alpha and p2 <= alpha:
                V2.add(w)
                continue
            if p2 > alpha and p1 <= alpha:
                V1.add(w)
                continue
            if p1 > alpha and p2 > alpha:
                C.add(w)
                continue
            (V1 if p1 < p2 else V2).add(w)
        moved = True
        while moved:
            moved = False
            for s in list(sorted(C)):
                condm = tuple(sorted(C - {s}))
                p1 = min(ci.pval(s, x, condm) for x in V1) if V1 else 1.0
                p2 = min(ci.pval(s, x, condm) for x in V2) if V2 else 1.0
                if p1 > alpha and p2 <= alpha:
                    C.remove(s)
                    V2.add(s)
                    moved = True
                elif p2 > alpha and p1 <= alpha:
                    C.remove(s)
                    V1.add(s)
                    moved = True
        score = min(len(V1), len(V2))
        if score > best_score:
            best = (C, V1, V2)
            best_score = score
    if best is None:
        return set(), set(), set(range(d))
    return best


# ---------- 合并（显著性降序加边 + 冗余删除） ----------
def _acyclic_add(G: nx.DiGraph, u: int, v: int) -> bool:
    G.add_edge(u, v)
    if not nx.is_directed_acyclic_graph(G):
        G.remove_edge(u, v)
        return False
    return True


def _merge(edges1, edges2, X: np.ndarray, alpha: float) -> np.ndarray:
    d = X.shape[1]
    all_edges = edges1 + edges2
    all_edges.sort(key=lambda x: x[2], reverse=True)
    G = nx.DiGraph()
    G.add_nodes_from(range(d))
    for u, v, s in all_edges:
        _acyclic_add(G, u, v)
    ci = _FisherZCI(X)
    to_remove = []
    for u, v in list(G.edges()):
        try:
            path = nx.shortest_path(G, u, v)
        except nx.NetworkXNoPath:
            continue
        if len(path) >= 3:
            cond = path[1:-1]
            if ci.pval(u, v, cond) > alpha:
                to_remove.append((u, v))
    G.remove_edges_from(to_remove)
    W = np.zeros((d, d), dtype=float)
    for u, v in G.edges():
        W[u, v] = 1.0
    return W


# ---------- 基础算法 A（可选 LiNGAM 或 PC） ----------
class _LiNGAMSolver:
    def __init__(self, thresh: float = 0.0):
        self.thresh = thresh

    def solve(self, X: np.ndarray):
        from lingam import DirectLiNGAM  # 需要: pip install lingam

        m = DirectLiNGAM().fit(X)
        W = np.asarray(m.adjacency_matrix_, dtype=float)
        np.fill_diagonal(W, 0.0)
        if self.thresh > 0:
            W[np.abs(W) < self.thresh] = 0.0
        S = np.abs(W)
        return W, S


class _PCSolver:
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def solve(self, X: np.ndarray):
        import numpy as np

        try:
            from castle.algorithms import PC
        except Exception as e:
            raise RuntimeError("需要 gcastle，请先安装：pip install gcastle") from e

        # gcastle 的 PC：alpha 为显著性阈值，stable=True 通常更稳
        model = PC(alpha=self.alpha)

        # 有些版本返回矩阵，有些把结果放在 model.causal_matrix 里，这里统一兼容
        W = model.learn(np.asarray(X, dtype=float))
        if isinstance(W, dict):
            if "adjacency_matrix" in W:
                W = W["adjacency_matrix"]
            elif "W_est" in W:
                W = W["W_est"]
            else:
                # 回退到属性
                W = getattr(model, "causal_matrix", None)

        if W is None:
            W = getattr(model, "causal_matrix", None)

        if W is None:
            raise RuntimeError(
                "PC 未返回可识别的邻接矩阵（adjacency_matrix/causal_matrix）。"
            )

        W = np.asarray(W, dtype=float)
        np.fill_diagonal(W, 0.0)
        S = np.abs(W)  # 用作显著性/置信度分数
        return W, S


# ---------- SADA 类 ----------
class SADA:
    """
    与 castle/gcastle 算法一致的接口：
      - __init__(...) 设超参
      - learn(data)    训练得到 self.causal_matrix (d x d, 0/1)
    """

    def __init__(
        self,
        theta: int = 10,
        alpha: float = 0.01,
        k: int = 10,
        max_cond: int = 3,
        sub_method: str = "pc",  # 'lingam' 或 'pc'
        thresh: float = 0.0,  # lingam 的阈值
        pc_alpha: float = 0.01,
        random_state: int = 0,
    ):
        self.theta = int(theta)
        self.alpha = float(alpha)
        self.k = int(k)
        self.max_cond = int(max_cond)
        self.sub_method = sub_method.lower().strip()
        self.thresh = float(thresh)
        self.pc_alpha = float(pc_alpha)
        self.random_state = int(random_state)

        self.causal_matrix: Optional[np.ndarray] = None

    # 递归主体
    def _sada_on_vars(self, X: np.ndarray, vars_idx: List[int]):
        m = len(vars_idx)
        X_sub = X[:, vars_idx]
        if m <= self.theta:
            solver = (
                _LiNGAMSolver(self.thresh)
                if self.sub_method.startswith("lingam")
                else _PCSolver(self.pc_alpha)
            )
            W, S = solver.solve(X_sub)
            # 保证 DAG：按分数降序逐条加边
            G = nx.DiGraph()
            G.add_nodes_from(range(m))
            edges = [
                (i, j, float(abs(S[i, j])))
                for i in range(m)
                for j in range(m)
                if i != j and W[i, j] != 0
            ]
            edges.sort(key=lambda x: x[2], reverse=True)
            for i, j, s in edges:
                if not _acyclic_add(G, i, j):
                    continue
            W2 = np.zeros((m, m))
            for i, j in G.edges():
                W2[i, j] = 1.0
            return W2, np.abs(W)

        rng = np.random.default_rng(self.random_state)
        C, V1, V2 = _find_causal_cut(X_sub, self.alpha, self.k, self.max_cond, rng)
        if len(V1) == 0 or len(V2) == 0:
            solver = (
                _LiNGAMSolver(self.thresh)
                if self.sub_method.startswith("lingam")
                else _PCSolver(self.pc_alpha)
            )
            return solver.solve(X_sub)

        idx_V1C = sorted(list(V1 | C))
        idx_V2C = sorted(list(V2 | C))
        W1, S1 = self._sada_on_vars(X_sub, idx_V1C)
        W2, S2 = self._sada_on_vars(X_sub, idx_V2C)

        def edges_from(W, S, idx_map):
            es = []
            mm = W.shape[0]
            for i in range(mm):
                for j in range(mm):
                    if i != j and W[i, j] != 0:
                        gi, gj = idx_map[i], idx_map[j]
                        es.append((gi, gj, float(abs(S[i, j]))))
            return es

        edges1 = edges_from(W1, S1, idx_V1C)
        edges2 = edges_from(W2, S2, idx_V2C)
        Wm = _merge(edges1, edges2, X_sub, self.alpha)
        return Wm, np.abs(Wm)

    def learn(self, data: np.ndarray):
        X = np.asarray(data, dtype=float)
        n, d = X.shape
        W, _ = self._sada_on_vars(X, list(range(d)))
        self.causal_matrix = (W > 0).astype(int)
        return self.causal_matrix
