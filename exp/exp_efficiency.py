# coding : utf-8
# Author : yuxiang Zeng
# 注意，这里的代码已经几乎完善，非必要不要改动（2025年1月17日19:47:38）

from thop import profile
from data_provider.data_center import DataModule
from exp.exp_model import Model
import torch
import numpy as np
import time
import gc
    
    
def evaluate_model_efficiency(datamodule, model, log, config):
    device = config.device
    model = model.to(device)
    model.eval()
    
    loader = datamodule.train_loader  # 使用验证集评估效率，确保 bs=1
    # === 获取样本输入 确保bs == 1 ===
    sample_inputs = next(iter(loader))
    inputs = tuple([item.to(device) for item in sample_inputs][:-1])

    # === FLOPs & Params ===
    flops, params = profile(model, inputs=inputs, verbose=False)

    # === 参数单位转换 ===
    param_bytes = params * 4  # float32
    param_kb = param_bytes / 1024
    param_mb = param_kb / 1024
    param_gb = param_mb / 1024

    # === FLOPs 单位转换 ===
    flops_m = flops / 1e6
    flops_g = flops / 1e9

    # === GPU 显存峰值 ===
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        model(*inputs)
    peak_mem_bytes = torch.cuda.max_memory_allocated(device)
    peak_mem_kb = peak_mem_bytes / 1024
    peak_mem_mb = peak_mem_kb / 1024
    peak_mem_gb = peak_mem_mb / 1024

    # === 推理时间 per epoch（取 100 次平均） ===
    eval_repeats = 100
    eval_times = []
    model.eval()
    
    with torch.no_grad():
        for _ in range(eval_repeats):
            inputs = [item.to(config.device) for item in sample_inputs][:-1]

            torch.cuda.synchronize()
            start_eval = time.time()
            model(*inputs)
            torch.cuda.synchronize()
            end_eval = time.time()

            eval_times.append(end_eval - start_eval)

    eval_time_s = np.mean(eval_times)
    eval_time_ms = eval_time_s * 1000

    gc.collect()
    torch.cuda.empty_cache()
    
    efficiency = {
        'Parameters': {
            'Count': params,
            'Size_Bytes': param_bytes,
            'Size_KB': param_kb,
            'Size_MB': param_mb,
            'Size_GB': param_gb,
        },
        'FLOPs': {
            'Count': flops,
            'MFLOPs': flops_m,
            'GFLOPs': flops_g,
        },
        'GPU_Memory_Usage': {
            'Bytes': peak_mem_bytes,
            'KB': peak_mem_kb,
            'MB': peak_mem_mb,
            'GB': peak_mem_gb,
        },
        'Inference_Cost_Per_Epoch': {
            'Seconds': eval_time_s,
            'Milliseconds': eval_time_ms,
        }
    }

    try:
        log('*' * 15 + 'Model Efficiency Evaluation' + '*' * 15)
        log(f"FLOPs: {efficiency['FLOPs']['Count']:.0f} "
            f"(MFLOPs: {efficiency['FLOPs']['MFLOPs']:.2f}, "
            f"GFLOPs: {efficiency['FLOPs']['GFLOPs']:.4f})")
        log(f"Params: {efficiency['Parameters']['Count']:.0f} "
            f"(MB: {efficiency['Parameters']['Size_MB']:.2f}, "
            f"GB: {efficiency['Parameters']['Size_GB']:.4f})")
        log(f"GPU Memory Peak: {efficiency['GPU_Memory_Usage']['MB']:.2f} MB "
            f"({efficiency['GPU_Memory_Usage']['GB']:.4f} GB)")
        log(f"Inference time per epoch: {efficiency['Inference_Cost_Per_Epoch']['Milliseconds']:.2f} ms "
            f"({efficiency['Inference_Cost_Per_Epoch']['Seconds']:.2f} s)")
        log('*' * 15 + 'Model Efficiency Evaluation' + '*' * 15)
    except Exception as e:
        log(f"[ERROR] Model efficiency evaluation failed: {e}")

    return efficiency