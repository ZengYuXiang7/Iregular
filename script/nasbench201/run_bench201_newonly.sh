#!/bin/bash

CONFIGS=(
#   "NarFormerConfig"
  "BRPNASConfig"
)

dst_dataset=(
    "mobile-cpu-snapdragon-855-kryo-485-int8.pkl"
    "mobile-gpu-snapdragon-450-adreno-506-int8.pkl"
    "mobile-dsp-snapdragon-855-hexagon-690-int8.pkl"
    "embeeded-tpu-edgetpu-int8.pkl"
)
spliter_ratios=(
    "5:4:91"
)

# 微调阶段
for CONFIG in "${CONFIGS[@]}"
do
    for spliter_ratio in "${spliter_ratios[@]}"
    do
        for j in "${!dst_dataset[@]}"  
        do
            dataset=${dst_dataset[$j]}   
            python run_train.py --exp_name "$CONFIG" --retrain 1 --logger zyx --transfer False \
            --src_dataset "data/nasbench201/pkl/$dataset" \
            --dst_dataset "data/nasbench201/pkl/$dataset" \
            --spliter_ratio "$spliter_ratio" \
            --dataset nasbench201   # 使用第一个数据集训练的模型进行微调
        done
    done
done

echo "所有微调实验已完成！"
