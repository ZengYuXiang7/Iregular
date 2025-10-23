#!/bin/bash

# 定义数据集数组
CONFIGS=(
#   "FlopsConfig"
#   "MacConfig"
#   "GRUConfig"
#   "LSTMConfig"
#   "BRPNASConfig"
#   "GATConfig"
  "NarFormerConfig"
#   "nnMeterConfig"
)

# 定义splitter_ratio数组
spliter_ratios=(
    # "1:4:95"
    # "3:4:93"
    # "5:4:91"
    "10:4:86"
)

# 遍历每个splitter_ratio值
for spliter_ratio in "${spliter_ratios[@]}"
do
    # 遍历每个配置和数据集，并运行训练脚本
    for config in "${CONFIGS[@]}"
    do
        echo "使用splitter_ratio: $spliter_ratio，配置: $config"
        
        # 调用训练脚本，使用配置和数据集
        python run_train.py --exp_name $config --dataset nnlqp --retrain 1 --logger zyx --transfer False \
        --spliter_ratio "$spliter_ratio" 
        
        # 输出当前任务完成配置和splitter_ratio
        echo "使用splitter_ratio: $spliter_ratio，配置: $config"
    done
done

echo "所有实验已完成！"
