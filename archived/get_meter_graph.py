import json
import os 
from typing import List, Tuple
import numpy as np
import torch
import dgl
from nn_meter.utils.graph_tool import ModelGraph
from nn_meter.builder.kernel_predictor_builder.predictor_builder.extract_feature import get_feature_parser


def find_jsonl_files(root_dir: str) -> List[str]:
    """
    递归查找目录下所有 .jsonl 文件
    """
    jsonl_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith('.jsonl'):
                full_path = os.path.join(dirpath, file)
                jsonl_files.append(full_path)
    return jsonl_files

def get_unified_operator_features(model_graph):
    feature_names = [
        'type_placeholder', 'type_conv2d', 'type_relu', 'type_maxpool',
        'type_mean', 'type_reshape', 'type_matmul',
        'input_h', 'input_w', 'input_c', 'output_c',
        'kernel_h', 'kernel_w', 'stride_h', 'stride_w',
        'padding_valid', 'padding_same', 'reduction_indices',
        'reshape_dim1', 'reshape_dim2', 'reshape_dim3', 'reshape_dim4'
    ]
    type_indices = {
        'Placeholder': 0, 'Conv2D': 1, 'Relu': 2,
        'MaxPool': 3, 'Mean': 4, 'Reshape': 5, 'MatMul': 6
    }

    node_features = {}
    for node in model_graph.get_graph().keys():
        attr = model_graph.get_node_attr(node)
        v = torch.zeros(len(feature_names), dtype=torch.float32)
        t = attr.get("type", "")
        if t in type_indices:
            v[type_indices[t]] = 1.0

        a = attr.get("attr", {})
        if t == 'Placeholder':
            shape = a.get("shape", [])
            if len(shape) == 4:
                v[7], v[8], v[9] = shape[1], shape[2], shape[3]
        elif t == 'Conv2D':
            ks = a.get("kernel_shape", [])
            strides = a.get("strides", [])
            pad = a.get("padding", "")
            wshape = a.get("weight_shape", [])
            if len(wshape) >= 4:
                v[9], v[10] = wshape[2], wshape[3]
            if len(ks) >= 2:
                v[11], v[12] = ks[0], ks[1]
            if len(strides) >= 2:
                v[13], v[14] = strides[-2], strides[-1]
            v[15 if pad == 'VALID' else 16 if pad == 'SAME' else -1] = 1.0
        elif t == 'MaxPool':
            ks, strides = a.get("ksize", []), a.get("strides", [])
            pad = a.get("padding", "")
            if len(ks) >= 2:
                v[11], v[12] = ks[-2], ks[-1]
            if len(strides) >= 2:
                v[13], v[14] = strides[-2], strides[-1]
            v[15 if pad == 'VALID' else 16 if pad == 'SAME' else -1] = 1.0
        elif t == 'Mean':
            ri = a.get("reduction_indices", [])
            if ri:
                v[17] = sum(ri)
        elif t == 'Reshape':
            shape = a.get("shape", [])
            for i in range(min(len(shape), 4)):
                v[18 + i] = shape[i]
        # MatMul only uses type one-hot
        node_features[node] = v
    return node_features, feature_names

def convert_nx_to_dgl(nx_graph, node_features):
    node_list = list(nx_graph.nodes())
    g = dgl.graph([])
    g.add_nodes(len(node_list))
    src = [node_list.index(u) for u, _ in nx_graph.edges()]
    dst = [node_list.index(v) for _, v in nx_graph.edges()]
    g.add_edges(src, dst)
    feat = torch.stack([node_features[n] for n in node_list], dim=0)
    g.ndata['feat'] = feat
    return g

def load_and_convert_to_dgl(jsonl_path):
    """
    读取 .jsonl 文件中的多个 graph，转换为 DGL 图列表
    
    返回:
        dgl_graphs: List[dgl.DGLGraph]
        performance_data: List[Dict]
    """
    dgl_graphs = []
    performance_data = []

    with open(jsonl_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            graph_data = entry['graph']
            performance_data.append(entry)

            model_graph = ModelGraph(graph=graph_data)
            model_graph.refresh()
            node_features, _ = get_unified_operator_features(model_graph)
            nxg = model_graph.get_networkx_graph()
            dglg = convert_nx_to_dgl(nxg, node_features)
            dgl_graphs.append(dglg)

    return dgl_graphs, performance_data



def batch_load_graphs_from_dir(root_dir: str) -> List[Tuple[str, List[dgl.DGLGraph], List[dict]]]:
    """
    查找并批量处理所有 jsonl 文件，转换为 DGL 图
    
    返回:
        List[Tuple[文件路径, DGL 图列表, 原始性能数据列表]]
    """
    jsonl_files = find_jsonl_files(root_dir)
    results = []

    for jsonl_path in jsonl_files:
        print(f"正在加载: {jsonl_path}")
        try:
            dgl_graphs, perf_list = load_and_convert_to_dgl(jsonl_path)
            print(f"  → 成功构建 {len(dgl_graphs)} 个图")
            results.append((jsonl_path, dgl_graphs, perf_list))
        except Exception as e:
            print(f"  × 加载失败: {jsonl_path}")
            print(f"    错误信息: {e}")
    
    return results

# 使用示例
if __name__ == '__main__':
    root = 'data/datasets—nnmeter'
    all_graph_results = batch_load_graphs_from_dir(root)

    # 示例：访问第一个文件的第一个图
    if all_graph_results:
        path, graphs, metadata = all_graph_results[0]
        print(f"\n示例：文件 {path}")
        print(f"第一个图的节点数: {graphs[0].num_nodes()}, 特征维度: {graphs[0].ndata['feat'].shape}")
        print(f"对应元数据 ID: {metadata[0].get('id')}")