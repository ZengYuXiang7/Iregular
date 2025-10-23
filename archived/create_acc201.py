import pickle
import numpy as np
import os
from data_provider.data_scaler import get_scaler
from data_process.nas_201_api import NASBench201API as API


def get_arch_str_from_arch_vector(arch_vector):
    _opname_to_index = {
        'none': 0,
        'skip_connect': 1,
        'nor_conv_1x1': 2,
        'nor_conv_3x3': 3,
        'avg_pool_3x3': 4,
        'input': 5,
        'output': 6,
        'global': 7
    }
    _opindex_to_name = {value: key for key, value in _opname_to_index.items()}
    ops = [_opindex_to_name[int(opindex)] for opindex in arch_vector]
    return '|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'.format(*ops)

with open('./data/nasbench201/pkl/desktop-cpu-core-i7-7820x-fp32.pkl', 'rb') as f:
    df = pickle.load(f)


api = API('./data_process/nas_201_api/NAS-Bench-201-v1_0-e61699.pth', verbose=False)

# -------------------------------
# 扫描所有架构，收集 (key, flops, params, acc)
raw_records = []   # [(tuple(key6), flops, params, acc)]

def extract_acc(acc_info: dict):
    """
    稳健提取准确率字段：不同dataset/设置下返回字典的键名可能不同。
    这里做一个优先级尝试。
    """
    for k in ['test-accuracy', 'accuracy', 'val-accuracy', 'valid-accuracy', 'test_acc', 'acc']:
        if k in acc_info:
            return float(acc_info[k])
    # 实在没有就抛错，便于发现字段变化
    raise KeyError(f'Cannot find accuracy key in acc_info. Keys: {list(acc_info.keys())}')

N = len(df)
for i in range(N):
    try:
        # df 若为 numpy 数组，下面索引正常；若为 list，请先转 np.array
        key = np.asarray(df[i, :-1], dtype=np.int32)   # 长度应为 6
        arch_str = get_arch_str_from_arch_vector(key)
        index = api.query_index_by_arch(arch_str)

        cost_info = api.get_cost_info(index, dataset='cifar10-valid')
        flops  = float(cost_info['flops'])
        params = float(cost_info['params'])

        acc_info = api.get_more_info(
            index,
            dataset='cifar10-valid',
            iepoch=None,
            hp='200',
            is_random=False
        )
        acc = extract_acc(acc_info)

        raw_records.append((tuple(key.tolist()), flops, params, acc))

        # 可选：简单进度打印
        if (i + 1) % 1000 == 0 or i == N - 1:
            print(f'Processed {i + 1}/{N}')
    except Exception as e:
        print(f"[Warning] Skipped item {i} due to: {e}")

# -------------------------------
# 组装为 n x 9 的矩阵（前6列=key，后3列=flops/params/acc）
if not raw_records:
    raise RuntimeError("raw_records 为空，请检查上面的 API 与数据。")

keys_mat = np.asarray([list(rec[0]) for rec in raw_records], dtype=np.float32)  # (n,6)
tails   = np.asarray([[rec[1], rec[2], rec[3]] for rec in raw_records], dtype=np.float32)  # (n,3)
df = np.hstack([keys_mat, tails])  # (n, 9)

print("最终矩阵形状:", df.shape)   # 期望 (n, 9)
print("示例前两行：\n", df[:2])

# -------------------------------
current_folder = os.path.join(os.getcwd(), 'data_process')
# 保存原始记录（可选）
with open(f'{current_folder}/nasbench201_acc.pkl', 'wb') as f:
    pickle.dump(raw_records, f)
    
    
# ===== 3. 加载并转为 float32 =====
from utils.exp_config import get_config
config = get_config()

for runid in range(5):
    config.runid = runid
    print(f"Processing run {runid}...")
    # 数据集名和存储目录
    dataset_name = '201_acc'
    save_dir = os.path.join("data", dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    # 生成当前 run 的文件名
    def file_path(var_name):
        """生成形如 dataset_x_run1.pkl 的文件路径"""
        return os.path.join(save_dir, f"{dataset_name}_{var_name}_round{config.runid}.pkl")

    x = df[:, :-1]  # 前6列为架构key
    y = df[:, -1]   # 后1列为 acc
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    # ===== 4. 切分比例 =====
    parts = [int(s) for s in config.spliter_ratio.strip().split(':')]
    total = sum(parts)
    train_ratio, valid_ratio = parts[0] / total, parts[1] / total

    # ===== 5. 打乱切分 =====
    n = len(x)
    train_size = int(n * train_ratio)
    valid_size = int(n * valid_ratio) if config.eval_set else 0

    indices = np.random.permutation(n)
    train_idx = indices[:train_size]
    valid_idx = indices[train_size:train_size + valid_size]
    test_idx  = indices[train_size + valid_size:]

    train_x, train_y = x[train_idx], y[train_idx]
    valid_x, valid_y = x[valid_idx], y[valid_idx]
    test_x,  test_y  = x[test_idx],  y[test_idx]

    # ===== 6. 归一化 =====
    # x_scaler = get_scaler(train_x, config, 'minmax')
    train_x[:, -2:] = train_x[:, -2:].astype(np.float32)
    x_scaler = get_scaler(train_x[:, -2:], config, 'minmax')

    train_x[:, -2:] = x_scaler.transform(train_x[:, -2:])
    valid_x[:, -2:] = x_scaler.transform(valid_x[:, -2:])
    test_x[:, -2:] = x_scaler.transform(test_x[:, -2:])
        
    # train_x = x_scaler.transform(train_x)
    # valid_x = x_scaler.transform(valid_x)
    # test_x  = x_scaler.transform(test_x)

    y_scaler = get_scaler(train_y, config, 'globalminmax')
    train_y = y_scaler.transform(train_y).astype(np.float32)
    valid_y = y_scaler.transform(valid_y).astype(np.float32)
    test_y  = y_scaler.transform(test_y).astype(np.float32)

    # ===== 7. 保存每个变量 =====
    def save_pkl(var_name, data):
        with open(file_path(var_name), 'wb') as f:
            pickle.dump(data, f)

    save_pkl("train_x", train_x)
    save_pkl("valid_x", valid_x)
    save_pkl("test_x",  test_x)
    save_pkl("train_y", train_y)
    save_pkl("valid_y", valid_y)
    save_pkl("test_y",  test_y)