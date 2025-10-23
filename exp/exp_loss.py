# coding : utf-8
# Author : Yuxiang Zeng
import torch
import torch.nn as nn
import random


# 在这里加上每个Batch的loss，如果有其他的loss，请在这里添加，
def compute_loss(model, pred, label, config):
    pred = pred.reshape(label.shape)
    
    total_loss = 0.0
    if config.model == 'ours':
        total_loss += model.loss_function(pred, label)
        
        # ---- full model: mse + spearman/kendall + sr + ac ----
        # loss_sp, loss_kd = model.rank_loss(pred, label)
        # sr = 0.5 * model.sr_loss(pred, label)   
        # ac = 0.1 * model.ac_loss(pred)
        # total_loss = 1.0 * mse + 1.0 * loss_sp + 1.0 * loss_kd + sr + ac    
        # ---- mse + spearman + kendall (较强排序约束) ----
        # loss_sp, loss_kd = model.rank_loss(pred, label)
        # total_loss = 1.0 * mse + 0.5 * loss_sp + 0.5 * loss_kd
        
        # elif config.try_exp == 6:
        # ---- mse + pairwise diff + AC ----
        # sr = 0.5 * model.sr_loss(pred, label)  # 差值一致性
        # ac = 0.1 * model.ac_loss(pred)         # 一致性正则
        # total_loss = 1.0 * mse + sr + ac
        return total_loss
    
    
    loss = model.loss_function(pred, label)
    return loss
    
    # if config.model == 'nnformer':
        # loss = model.nnformer_loss(pred, label)
        # return loss
        
        
        