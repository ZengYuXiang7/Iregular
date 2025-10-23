#!/bin/bash
clear


python run_train.py --exp_name FlopsConfig --dataset nnlqp --retrain 1 --logger zyx --transfer False \
  --src_dataset data/nasbench201/pkl/desktop-cpu-core-i7-7820x-fp32.pkl \
  --dst_dataset data/nasbench201/pkl/desktop-cpu-core-i7-7820x-fp32.pkl


# python run_train.py --exp_name TransModelConfig --retrain 1 --logger zyx --transfer True \
#   --src_dataset data/pickle/embedded-gpu-jetson-nono-fp16_data.pkl \
#   --dst_dataset data/pickle/desktop-gpu-gtx-1080ti-fp32_data.pkl