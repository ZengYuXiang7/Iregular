import glob
import numpy as np
import pandas as pd
import networkx as nx

from models.causal import get_causal_matrix


# 正则化匹配表达全部文件夹
def get_data_csv(dataset):
    return glob.glob(f"./data/microservice/{dataset}/**/data.csv", recursive=True)


# 先实行因果图构建的算法
def find_root_cause(data: pd.DataFrame, method, config):
    ############# PREPROCESSING ###############
    if "time.1" in data:
        data = data.drop(columns=["time.1"])
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.ffill()
    data = data.fillna(0)

    # remove latency-50 columns
    data = data.loc[:, ~data.columns.str.endswith("latency-50")]
    # rename latency-90 columns to latency
    data = data.rename(
        columns={
            c: c.replace("_latency-90", "_latency")
            for c in data.columns
            if c.endswith("_latency-90")
        }
    )
    print(f"Data shape is {data.shape}")
    # data = data[:100]

    # 因果图构建
    causal_matrix, execute_time = get_causal_matrix(data, method, config)

    # 进行Pagerank排序 构建有向图（i → j 表示 i 影响 j）
    G = nx.DiGraph(causal_matrix)

    # 计算 PageRank（越大表示越可能是根因）
    pagerank_scores = nx.pagerank(G, alpha=0.85, tol=1e-6)

    # 转为 DataFrame 排序
    pr_df = pd.DataFrame(list(pagerank_scores.items()), columns=["node", "score"])
    pr_df = pr_df.sort_values(by="score", ascending=False).reset_index(drop=True)

    # 输出 Top-K 根因节点
    topk_nodes = pr_df.head(5)
    print("Top-K 根因节点（按重要性降序）:", topk_nodes)
    
    return topk_nodes, execute_time


def execute_once(dataset, config):
    # TODO：：需要做离散编码，然后逐个数据集遍历，导入因果算法，确定根因，计算metric
    all_datasets = get_data_csv(dataset)

    top1_cnt, top2_cnt, top3_cnt, top4_cnt, top5_cnt, total_cnt = 0, 0, 0, 0, 0, 0

    for i, now_dataset_address in enumerate(all_datasets):
        _, b = now_dataset_address.split(dataset)
        service_label, metric_label = b.split("/")[1].split('_')

        # 读取数据集
        data = pd.read_csv(now_dataset_address)
        all_metrics = data.columns

        # 执行根因发现算法
        root_cause_idx, execute_time = find_root_cause(data, "GOLEM", config)
        root_cause = all_metrics[root_cause_idx['node']].to_numpy()

        # 统计 Topk 正确率
        # Top1：判断前 1 个元素中是否包含子串
        if any(service_label in item for item in root_cause[:1]):
            top1_cnt += 1
        if any(service_label in item for item in root_cause[:2]):
            top2_cnt += 1
        if any(service_label in item for item in root_cause[:3]):
            top3_cnt += 1
        if any(service_label in item for item in root_cause[:4]):
            top4_cnt += 1
        if any(service_label in item for item in root_cause[:5]):
            top5_cnt += 1
        total_cnt += 1
        
        print(f"[Top5 Hit = {top5_cnt}/{total_cnt}]:  Execute {i} - label={service_label} preds={root_cause} Execution Time={execute_time}")

    # Metrics
    top1_accuracy = top1_cnt / total_cnt
    top2_accuracy = top2_cnt / total_cnt
    top3_accuracy = top3_cnt / total_cnt
    top4_accuracy = top4_cnt / total_cnt
    top5_accuracy = top5_cnt / total_cnt

    avg5_accuracy = (
        top1_accuracy + top2_accuracy + top3_accuracy + top4_accuracy + top5_accuracy
    ) / 5

    print(f"Avg@5 Accuracy: {avg5_accuracy}")

    metrics = {
        "top1_accuracy": top1_accuracy,
        "top2_accuracy": top2_accuracy,
        "top3_accuracy": top3_accuracy,
        "top4_accuracy": top4_accuracy,
        "top5_accuracy": top5_accuracy,
        "avg5_accuracy": avg5_accuracy,
    }

    return metrics


if __name__ == "__main__":
    # Experiment Settings, logger, plotter
    from utils.exp_logger import Logger
    from utils.exp_metrics_plotter import MetricsPlotter
    from utils.utils import set_settings
    from utils.exp_config import get_config

    config = get_config("OurModelConfig")
    set_settings(config)

    datasets = ["fse-ob", "fse-ss", "fse-tt"]
    for dataset in datasets:
        metrics = execute_once(dataset, config)
        print(dataset, metrics)
