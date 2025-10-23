# coding : utf-8
# Author : Yuxiang Zeng
# 注意，这里的代码已经几乎完善，非必要不要改动（2025年3月27日23:33:32）
import contextlib
import torch
import math
from time import time
from exp.exp_loss import compute_loss
from exp.exp_metrics import ErrorMetrics
from utils.model_trainer import get_loss_function, get_optimizer
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import LambdaLR
from tqdm import *

def build_stable_warmup_hold_cosine(
    optimizer,
    total_units: int,           # 总推进次数：按step就=总steps，按epoch就=总epochs
    warmup_ratio: float = 0.05, # 热身比例（5%总进度）
    hold_ratio: float = 0.00,   # 热身后保持比例（0~10%常见）
    min_lr_ratio: float = 0.05, # 末端最小lr比例（相对初始lr），建议5%~10%
):
    """
    返回一个 LambdaLR，调度规则：
        0 → warmup:     线性从 0 → 1
        warmup → hold:  保持 1
        hold → end:     余弦从 1 → min_lr_ratio
    注：你需要在训练中每一步（step版）或每个epoch（epoch版）调用 .step()
    """
    total_units = max(1, int(total_units))
    wu = max(1, int(total_units * warmup_ratio))
    hd = max(0, int(total_units * hold_ratio))
    cos_len = max(1, total_units - wu - hd)
    
    def lr_lambda(cur):
        # cur 从 0 开始计数：已推进次数
        if cur < wu:  # warmup
            return (cur + 1) / wu
        if cur < wu + hd:  # hold
            return 1.0
        # cosine
        t = cur - wu - hd
        progress = t / cos_len  # 0 → 1
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return LambdaLR(optimizer, lr_lambda)

                    
class BasicModel(torch.nn.Module):
    def __init__(self, config):
        super(BasicModel, self).__init__()
        self.config = config
        self.scaler = torch.amp.GradScaler(config.device)  # ✅ 初始化 GradScaler
        
    def forward(self, *x):
        y = self.model(*x)
        return y

    def setup_optimizer(self, config):
        self.to(config.device)
        self.loss_function = get_loss_function(config).to(config.device)
        self.optimizer     = get_optimizer(self.parameters(), lr=config.lr, decay=config.decay, config=config)
        
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            # self.optimizer, T_max=config.epochs, eta_min=0
        # )
        if config.model == 'ours':
            self.scheduler = build_stable_warmup_hold_cosine(
                self.optimizer,
                total_units=config.epochs * 0.6,
                warmup_ratio=0.05,   # 5% 热身
                hold_ratio=0.02,     # 2% 保持（可为 0）
                min_lr_ratio=0.05    # 末端保留 5% 初始 lr
            )
        else:
            self.scheduler     = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=config.patience // 2, threshold=0.0, min_lr=1e-6
            )
            

    def train_one_epoch(self, dataModule):
        loss = None
        self.train()
        torch.set_grad_enabled(True)
        t1 = time()

        for train_batch in (dataModule.train_loader):
            all_item = [item.to(self.config.device) for item in train_batch]
            inputs, label = all_item[:-1], all_item[-1]

            self.optimizer.zero_grad()

            if self.config.use_amp:
                with torch.amp.autocast(device_type=self.config.device):
                    pred = self.forward(*inputs)
                    loss = compute_loss(self, pred, label, self.config)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.forward(*inputs)
                loss = compute_loss(self, pred, label, self.config)
                loss.backward()
                self.optimizer.step()

        t2 = time()
        return loss, t2 - t1

    def evaluate_one_epoch(self, dataModule, mode='valid'):
        self.eval()
        torch.set_grad_enabled(False)
        dataloader = dataModule.valid_loader if mode == 'valid' and len(dataModule.valid_loader.dataset) != 0 else dataModule.test_loader
        preds, reals, val_loss = [], [], 0.

        context = (
            torch.amp.autocast(device_type=self.config.device)
            if self.config.use_amp else
            contextlib.nullcontext()
        )
        with context:
            for batch in dataloader:
                all_item = [item.to(self.config.device) for item in batch]
                inputs, label = all_item[:-1], all_item[-1]
                pred = self.forward(*inputs)

                if mode == 'valid':
                    val_loss += compute_loss(self, pred, label, self.config)

                if self.config.classification:
                    pred = torch.max(pred, 1)[1]
                    
                preds.append(pred)
                reals.append(label)

        reals = torch.cat(reals, dim=0)
        preds = torch.cat(preds, dim=0)
        
        if self.config.dataset != 'weather':
            reals, preds = dataModule.y_scaler.inverse_transform(reals), dataModule.y_scaler.inverse_transform(preds)
            
        if mode == 'valid':
            # self.scheduler.step(val_loss)
            self.scheduler.step()

        return ErrorMetrics(reals, preds, self.config)
    
    
    # 专门为全数据集做的实验
    def evaluate_whole_dataset(self, dataModule, mode='whole'):
        self.eval()
        torch.set_grad_enabled(False)
        
        preds, reals = [], []
        
        def get_pred_and_real(loader):
            context = (
            torch.amp.autocast(device_type=self.config.device)
            if self.config.use_amp else
            contextlib.nullcontext()
            )
            with context:
                for batch in loader:
                    all_item = [item.to(self.config.device) for item in batch]
                    inputs, label = all_item[:-1], all_item[-1]
                    pred = self.forward(*inputs)

                    if mode == 'valid':
                        val_loss += compute_loss(self, pred, label, self.config)

                    if self.config.classification:
                        pred = torch.max(pred, 1)[1]
                        
                    preds.append(pred)
                    reals.append(label)
        
        get_pred_and_real(dataModule.train_loader)
        get_pred_and_real(dataModule.valid_loader)
        get_pred_and_real(dataModule.test_loader)

        reals = torch.cat(reals, dim=0)
        preds = torch.cat(preds, dim=0)
        
        reals, preds = dataModule.y_scaler.inverse_transform(reals), dataModule.y_scaler.inverse_transform(preds)
            
        return ErrorMetrics(reals, preds, self.config)