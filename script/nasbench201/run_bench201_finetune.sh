#!/bin/bash

CONFIGS=( 
  "BRPNASConfig"
  #   "NarFormerConfig"
)

# 定义微调数据集数组
src_dataset=(
    "desktop-cpu-core-i7-7820x-fp32.pkl"
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

# # 训练阶段
# for CONFIG in "${CONFIGS[@]}"
# do
#     for spliter_ratio in "${spliter_ratios[@]}"
#     do
#         # 只需使用 src_dataset 中的第一个数据集
#         dataset1=${src_dataset[0]}

#         echo "正在训练 $dataset1 使用配置 $CONFIG"

#         # 在 src_dataset 上进行训练，保存训练后的模型
#         python run_train.py --exp_name "$CONFIG" --retrain 1 --logger zyx --transfer False \
#         --src_dataset "data/nasbench201/pkl/$dataset1" \
#         --dst_dataset "data/nasbench201/pkl/$dataset1" \
#         --spliter_ratio "$spliter_ratio" \
#         --pretrained_model "" --dataset nasbench201   # 第一次训练，不加载预训练模型

#         echo "$dataset1 训练完成，保存模型！"

#         # 在训练完成后，保存的模型文件路径
#         trained_model="models/$dataset1_model.pth"
#     done
# done

# 微调阶段
for CONFIG in "${CONFIGS[@]}"
do
    for spliter_ratio in "${spliter_ratios[@]}"
    do
        # 使用 src_dataset 中的第一个数据集
        dataset1=${src_dataset[0]}

        for j in "${!dst_dataset[@]}"  # 使用 dst_dataset 的索引来遍历
        do
            dataset2=${dst_dataset[$j]}   # 使用 dst_dataset 中的数据集

            echo "正在微调 $dataset1 和 $dataset2 使用配置 $CONFIG"

            # 加载第一个训练好的模型，并在第二个数据集上进行微调
            python run_transfer.py --exp_name "$CONFIG" --retrain 1 --logger zyx --transfer True \
            --src_dataset "data/nasbench201/pkl/$dataset1" \
            --dst_dataset "data/nasbench201/pkl/$dataset2" \
            --spliter_ratio "$spliter_ratio" \
            --pretrained_model "$trained_model" --dataset nasbench201   # 使用 src_dataset 上训练的模型进行微调

            # 输出当前任务完成
            echo "$dataset1 和 $dataset2 使用配置 $CONFIG 微调完成！"
        done
    done
done

echo "所有训练与微调实验已完成！"
