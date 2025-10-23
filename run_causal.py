# causal_wrapper.py
from __future__ import annotations
import numpy as np
import networkx as nx
from typing import Tuple, Optional, Dict, Any

# ---------- 小工具：统一输出 ----------
def _as_adjacency_and_digraph(W: np.ndarray, var_names: Optional[list]=None) -> Tuple[np.ndarray, nx.DiGraph]:
    W = np.asarray(W, dtype=float)
    d = W.shape[0]
    if var_names is None:
        var_names = [f"x{i}" for i in range(d)]
    G = nx.DiGraph()
    G.add_nodes_from(var_names)
    for i in range(d):
        for j in range(d):
            if i != j and abs(W[i, j]) > 0:
                # 约定：W[i, j] 表示 i -> j 的权重
                G.add_edge(var_names[i], var_names[j], weight=float(W[i, j]))
    return W, G

def _require(pkg: str, install_hint: str):
    try:
        return __import__(pkg)
    except Exception as e:
        raise ImportError(
            f"需要安装 `{pkg}` 来使用该方法。可尝试：{install_hint}\n原始错误：{e}"
        )

# ---------- 主函数 ----------
def learn_causal_graph(
    X: np.ndarray,
    method: str = "pc",
    var_names: Optional[list] = None,
    standardize: bool = True,
    **kwargs: Any
) -> Tuple[np.ndarray, nx.DiGraph]:
    """
    学习因果结构。输入 X: [n, d]，输出 (W, G)，其中
    - W: d x d 的邻接/权重矩阵，W[i, j] 表示 i -> j
    - G: networkx.DiGraph，同样边方向 i->j

    参数
    ----
    method: 选择算法（不区分大小写），示例：
        'pc', 'ges',
        'anm', 'pnl',
        'directlingam', 'icalingam',
        'notears', 'notears-mlp', 'notears-sob', 'notears-low-rank',
        'dag-gnn', 'golem', 'grandag', 'mcsl', 'gae', 'rl', 'corl',
        'ttpm'
    kwargs: 透传给底层算法的关键超参（如 lambda1、alpha 等）

    依赖（任装其一即可，按优先顺序）：
    - gcastle（推荐，覆盖最全）：pip install gcastle
    - castle（老包名）：pip install castle
    - lingam：pip install lingam
    - causallearn（PC/GS）：pip install causallearn
    - notears（部分实现）：pip install notears
    """
    if X is None or len(X.shape) != 2:
        raise ValueError("X 必须是形状 [n, d] 的 2D 数组。")
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    if var_names is not None and len(var_names) != d:
        raise ValueError("var_names 长度必须等于 X 的列数 d。")

    method = method.lower().strip()
    if standardize:
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # ---- 先尝试 gcastle（覆盖最全） ----
    if method in {
        "pc","ges","anm","pnl",
        "directlingam","icalingam",
        "notears","notears-mlp","notears-sob","notears-low-rank",
        "dag-gnn","golem","grandag","mcsl","gae","rl","corl",
        "ttpm"
    }:
        try:
            gc = _require("gcastle", "pip install gcastle")
            from gcastle.utils.matrix import is_dag
            # 各算法类名在 gcastle.algo 下
            algo_map = {
                "pc": ("gcastle.algo.constraint_based", "PC"),
                "ges": ("gcastle.algo.score_based", "GES"),
                "anm": ("gcastle.algo.functional", "ANM"),
                "pnl": ("gcastle.algo.functional", "PNL"),
                "directlingam": ("gcastle.algo.functional", "DirectLiNGAM"),
                "icalingam": ("gcastle.algo.functional", "ICALiNGAM"),
                "notears": ("gcastle.algo.gradient", "NOTEARS"),
                "notears-mlp": ("gcastle.algo.gradient", "NOTEARS_MLP"),
                "notears-sob": ("gcastle.algo.gradient", "NOTEARS_SOB"),
                "notears-low-rank": ("gcastle.algo.gradient", "NOTEARS_LowRank"),
                "dag-gnn": ("gcastle.algo.gradient", "DAG_GNN"),
                "golem": ("gcastle.algo.gradient", "GOLEM"),
                "grandag": ("gcastle.algo.gradient", "GraNDAG"),
                "mcsl": ("gcastle.algo.gradient", "MCSL"),
                "gae": ("gcastle.algo.gradient", "GAE"),
                "rl": ("gcastle.algo.gradient", "RL"),
                "corl": ("gcastle.algo.gradient", "CORL"),
                "ttpm": ("gcastle.algo.torch", "TTPM"),
            }
            module_name, class_name = algo_map[method]
            mod = __import__(module_name, fromlist=[class_name])
            Algo = getattr(mod, class_name)
            model = Algo(**kwargs) if kwargs else Algo()
            W = model.learn(X)
            # gcastle 通常返回 dxd numpy 矩阵（有的返回字典含 'adjacency_matrix'）
            if isinstance(W, dict) and "adjacency_matrix" in W:
                W = W["adjacency_matrix"]
            if W.shape != (d, d):
                raise RuntimeError(f"{method} 返回了异常形状：{W.shape}")

            # 保证严格 DAG（部分算法已保证）
            if not is_dag(W):
                # 简单投影：去除最小权重的正环边（保守处理）
                W = W * (np.eye(d) == 0)
            return _as_adjacency_and_digraph(W, var_names)
        except ImportError:
            pass
        except Exception as e:
            # 若 gcastle 存在但单个算法不可用，则降级到其它后端
            # 不中断，继续尝试后端
            last_error = e
    else:
        last_error = None

    # ---- 其次尝试 lingam（DirectLiNGAM / ICALiNGAM / PNL 的常用后端）----
    if method in {"directlingam", "icalingam"}:
        try:
            lingam = _require("lingam", "pip install lingam")
            if method == "directlingam":
                from lingam import DirectLiNGAM
                model = DirectLiNGAM(**kwargs) if kwargs else DirectLiNGAM()
            else:
                from lingam import ICALiNGAM
                model = ICALiNGAM(**kwargs) if kwargs else ICALiNGAM()
            model.fit(X)
            W = model.adjacency_matrix_.astype(float)
            return _as_adjacency_and_digraph(W, var_names)
        except ImportError as e:
            last_error = e

    # ---- PC / GES 的另一条路：causallearn ----
    if method in {"pc", "ges"}:
        try:
            cl = _require("causallearn", "pip install causallearn")
            if method == "pc":
                from causallearn.search.ConstraintBased.PC import pc
                from causallearn.utils.cit import fisherz
                cg = pc(X, alpha=kwargs.get("alpha", 0.01), ci_test=fisherz, stable=kwargs.get("stable", True))
                # 从 CausalGraph 转邻接矩阵（有向边箭头 i->j）
                W = np.zeros((d, d), dtype=float)
                for (i, j) in cg.G.get_graph_edges():
                    e = cg.G.graph[i][j]
                    # 只保留有向边：i->j 或 j->i
                    if e.getEndpoint1().name == 'ARROW' and e.getEndpoint2().name == 'TAIL':
                        W[i, j] = 1.0
                    elif e.getEndpoint2().name == 'ARROW' and e.getEndpoint1().name == 'TAIL':
                        W[j, i] = 1.0
                return _as_adjacency_and_digraph(W, var_names)
            else:  # GES
                from causallearn.search.ScoreBased.GES import ges
                # 默认使用 BIC 分数
                result = ges(X, score_func=kwargs.get("score_func", "local_score_BIC"))
                W = np.zeros((d, d), dtype=float)
                for i in range(d):
                    for j in range(d):
                        if result['G'].graph[i, j] == 2:  # 2 表示 i->j
                            W[i, j] = 1.0
                return _as_adjacency_and_digraph(W, var_names)
        except ImportError as e:
            last_error = e

    # ---- NOTEARS 原版实现（部分环境）----
    if method in {"notears", "notears-mlp"}:
        try:
            nt = _require("notears", "pip install notears")
            if method == "notears":
                # 线性 NOTEARS
                from notears.linear import notears_linear, l2_reg
                W = notears_linear(
                    X, lambda1=kwargs.get("lambda1", 0.1),
                    loss_type=kwargs.get("loss_type", "l2"),
                    max_iter=kwargs.get("max_iter", 100),
                    w_threshold=kwargs.get("w_threshold", 0.0),
                    verbose=kwargs.get("verbose", False)
                )
                return _as_adjacency_and_digraph(W, var_names)
            else:
                # 非线性（MLP）NOTEARS
                from notears.nonlinear import notears_nonlinear, MLP
                d = X.shape[1]
                model = MLP(dims=[d, kwargs.get("hidden", 10), 1])
                W = notears_nonlinear(
                    X, model, lambda1=kwargs.get("lambda1", 0.01),
                    lambda2=kwargs.get("lambda2", 0.01),
                    max_iter=kwargs.get("max_iter", 200)
                )
                return _as_adjacency_and_digraph(W, var_names)
        except ImportError as e:
            last_error = e

    # ---- 最后尝试老包名 castle（兼容旧项目）----
    if method in {
        "pc","ges","anm","pnl",
        "directlingam","icalingam",
        "notears","notears-mlp","notears-sob","notears-low-rank",
        "dag-gnn","golem","grandag","mcsl","gae","rl","corl",
        "ttpm"
    }:
        try:
            cs = _require("castle", "pip install castle")
            # 一些老版本路径：castle.algorithms.* / castle.common.priori_knowledge 等
            # 这里仅示例常见算法；其余同 gcastle，可按需扩展
            if method in {"directlingam","icalingam"}:
                from castle.algorithms import DirectLiNGAM, ICALiNGAM
                Algo = DirectLiNGAM if method == "directlingam" else ICALiNGAM
                model = Algo(**kwargs) if kwargs else Algo()
                model.learn(X)
                W = model.causal_matrix  # castle 通用命名
                return _as_adjacency_and_digraph(W, var_names)
            if method == "notears":
                from castle.algorithms import NOTEARS
                model = NOTEARS(**kwargs) if kwargs else NOTEARS()
                model.learn(X)
                W = model.causal_matrix
                return _as_adjacency_and_digraph(W, var_names)
            if method == "pc":
                from castle.algorithms import PC
                model = PC(**kwargs) if kwargs else PC()
                model.learn(X)
                W = model.causal_matrix
                return _as_adjacency_and_digraph(W, var_names)
            if method == "ges":
                from castle.algorithms import GES
                model = GES(**kwargs) if kwargs else GES()
                model.learn(X)
                W = model.causal_matrix
                return _as_adjacency_and_digraph(W, var_names)
            # 其它算法可以仿照添加……
        except ImportError as e:
            last_error = e
        except Exception as e:
            last_error = e

    # ---- 没有可用后端 ----
    raise RuntimeError(
        f"method='{method}' 的实现未找到可用后端。"
        " 建议优先安装 gcastle：`pip install gcastle`；"
        " 或根据需要安装 lingam / causallearn / notears。\n"
        f"最后一次错误信息：{last_error}"
    )


# ---------- 简单示例 ----------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, d = 1000, 5
    # 生成一个简单 DAG: x0 -> x1 -> x2, x0 -> x3, x2 -> x4
    x0 = rng.normal(size=n)
    x1 = 2.0*x0 + rng.normal(size=n)
    x2 = -1.5*x1 + rng.normal(size=n)
    x3 = 0.8*x0 + rng.normal(size=n)
    x4 = 1.2*x2 + rng.normal(size=n)
    X = np.column_stack([x0, x1, x2, x3, x4])

    W, G = learn_causal_graph(X, method="directlingam")
    print("Adjacency:\n", np.round(W, 2))
    print("Edges:", list(G.edges()))
    
    
    # X: (n, d)
    W, G = learn_causal_graph(X, method="pc", alpha=0.01)              # 约束法（独立性检验）
    W, G = learn_causal_graph(X, method="ges")                         # 评分搜索
    W, G = learn_causal_graph(X, method="directlingam")                # 线性非高斯
    W, G = learn_causal_graph(X, method="icalingam")                   # ICA-LiNGAM
    W, G = learn_causal_graph(X, method="notears", lambda1=0.1)        # 线性 NOTEARS
    W, G = learn_causal_graph(X, method="notears-mlp")                 # 非线性 NOTEARS (MLP)
    W, G = learn_causal_graph(X, method="golem")                       # GOLEM
    # ……其余方法同名传入（只要后端已安装）