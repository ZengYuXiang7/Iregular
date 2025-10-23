#!/bin/bash
clear

CONFIGS=(
  # "FlopsConfig"
  # "MacConfig"
  # "GRUConfig"
  # "LSTMConfig"
  # "BRPNASConfig"
  # "GATConfig"
  "NarFormerConfig"
  # "nnMeterConfig"
)

for CONFIG in "${CONFIGS[@]}"
do
  echo "Running experiment with config: $CONFIG"
  # python run_train.py --exp_name "$CONFIG" --dataset nasbench201 --retrain 1 --logger zyx --transfer False --debug 1
  python run_train.py --exp_name "$CONFIG" --dataset nnlqp --retrain 1 --logger zyx --transfer False 
  
  echo "Finished $CONFIG"
  echo "-----------------------------"
done