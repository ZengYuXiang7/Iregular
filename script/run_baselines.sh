#!/bin/bash

# 定义数据集数组
CONFIGS=(
#   "FlopsConfig"
#   "MacConfig"
  "LSTMConfig"
  "GRUConfig"
  "BRPNASConfig"
  "NNLQPConfig"
  "GATConfig"
  "NarFormerConfig"
  "NarFormer2Config"
  "NNformerConfig"
  "OurModelConfig"
#   "nnMeterConfig"
)
# datasets=(
#     "desktop-cpu-core-i7-7820x-fp32.pkl"
#     # "mobile-cpu-snapdragon-675-kryo-460-int8.pkl"
# )
spliter_ratios=(
    "1:4:95"
    "3:4:93"
    "5:4:91"
    "10:4:86"
)
for config in "${CONFIGS[@]}"
do
    for spliter_ratio in "${spliter_ratios[@]}"
    do
        echo "使用splitter_ratio: $spliter_ratio，配置: $config"
        
        # 调用训练脚本，使用配置和数据集
        python run_train.py --exp_name $config --retrain 1 --logger wangyang --transfer False \
        --spliter_ratio "$spliter_ratio" --dataset 201_acc
        
    done
done




# "mobile-dsp-snapdragon-675-hexagon-685-int8.pkl"
# "mobile-dsp-snapdragon-855-hexagon-690-int8.pkl"
# "embedded-gpu-jetson-nono-fp16.pkl"
# "desktop-gpu-gtx-1080ti-fp32.pkl"
# "mobile-gpu-snapdragon-450-adreno-506-int8.pkl"
# "mobile-gpu-snapdragon-675-adren0-612-int8.pkl"
# "mobile-cpu-snapdragon-675-kryo-460-int8.pkl"
# "mobile-cpu-snapdragon-855-kryo-485-int8.pkl"
# "embedded-tpu-edgetpu-large.pkl"
# "mobile-cpu-snapdragon-450-contex-a53-int8.pkl"
# "embeeded-tpu-edgetpu-int8.pkl"
# "desktop-gpu-gtx-1080ti-large.pkl"
# "mobile-gpu-snapdragon-855-adren0-640-int8.pkl"