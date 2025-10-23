# coding : utf-8
# Author : yuxiang Zeng
import numpy as np
from utils.exp_sh import once_experiment
from datetime import datetime

# 在这里写下超参数探索空间
hyper_dict = {
    # 'd_model': [128, 192, 96, 256],
    'try_exp': [1, 2, 3, 4, 5, 6, 7, 8],  # 1-8
}

######################################################################################################
# 这里是总执行实验顺序！！！！！！！！
def experiment_run():
    Our_model()
    return True


def Ablation():
    return True


def Our_model(hyper=None):
    # monitor_metric = NMAE KendallTau
    once_experiment('OurModelConfig', hyper_dict, monitor_metric='KendallTau', reverse=True, debug=0)
    return True


def log_message(message):
    log_file = "run.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a') as f:
        f.write(f"[{timestamp}] {message}\n")

if __name__ == "__main__":
    try:
        log_message("Experiment Start!!!")
        experiment_run()
    except KeyboardInterrupt as e:
        log_message("Experiment interrupted by user.")
    finally:
        log_message("All commands executed successfully.\n")


