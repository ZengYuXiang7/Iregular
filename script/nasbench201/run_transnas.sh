#!/bin/bash
clear

# 定义数据集数组
CONFIGS=(
  TransModelConfig
)

datasets=(
    "desktop-cpu-core-i7-7820x-fp32.pkl"
)

# 定义splitter_ratio数组
spliter_ratios=(
    "1:4:95"
    "3:4:93"
    "5:4:91"
    "10:4:86"
)

# 遍历每个splitter_ratio值
for dataset in "${datasets[@]}"
do
    # 遍历每个配置和数据集，并运行训练脚本
    for config in "${CONFIGS[@]}"
    do
        for spliter_ratio in "${spliter_ratios[@]}"
        do
            echo "正在训练数据集: $dataset，使用splitter_ratio: $spliter_ratio，配置: $config"
            
            # 调用训练脚本，使用配置和数据集
            # python run_train.py --exp_name $config --retrain 1 --logger zyx --transfer False \
            # --src_dataset "data/nasbench201/pkl/$dataset" \
            # --dst_dataset "data/nasbench201/pkl/$dataset" \
            # --spliter_ratio "$spliter_ratio" --dataset nasbench201 \
            # --rank_loss True --ac_loss True --bs 16

            # 调用训练脚本，使用配置和数据集
            python run_train.py --exp_name $config --retrain 1 --logger zyx --transfer False \
            --src_dataset "data/nasbench201/pkl/$dataset" \
            --dst_dataset "data/nasbench201/pkl/$dataset" \
            --spliter_ratio "$spliter_ratio" --dataset nasbench201 \
            --rank_loss True --ac_loss True --bs 256

            # 输出当前任务完成配置和splitter_ratio
            echo "$dataset 的训练完成，使用splitter_ratio: $spliter_ratio，配置: $config"
        done
    done
done

echo "所有实验已完成！"
