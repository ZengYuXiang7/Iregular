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
    total_units: int,  # 总推进次数：按step就=总steps，按epoch就=总epochs
    warmup_ratio: float = 0.05,  # 热身比例（5%总进度）
    hold_ratio: float = 0.00,  # 热身后保持比例（0~10%常见）
    min_lr_ratio: float = 0.05,  # 末端最小lr比例（相对初始lr），建议5%~10%
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
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return LambdaLR(optimizer, lr_lambda)


class BasicModel(torch.nn.Module):
    def __init__(self, config):
        super(BasicModel, self).__init__()
        self.config = config
        self.scaler = torch.amp.GradScaler(config.device)

    def forward(self, *x):
        y = self.model(*x)
        return y

    def setup_optimizer(self, config):
        self.to(config.device)
        self.loss_function = get_loss_function(config).to(config.device)
        self.optimizer = get_optimizer(
            self.parameters(), lr=config.lr, decay=config.decay, config=config
        )

        if config.model == "ours":
            self.scheduler = build_stable_warmup_hold_cosine(
                self.optimizer,
                total_units=config.epochs * 0.6,
                warmup_ratio=0.05,
                hold_ratio=0.02,
                min_lr_ratio=0.05,
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=config.patience // 2,
                threshold=0.0,
                min_lr=1e-6,
            )


def exec_train_one_epoch(model, dataModule, config):
    # 处理 DataParallel 带来的差异：
    # 如果 model 是 DataParallel，它没有 optimizer 属性，得去 module 里找
    real_model = model.module if isinstance(model, torch.nn.DataParallel) else model

    loss = None
    model.train()  # 这里对 DataParallel 调用 train() 是有效的
    torch.set_grad_enabled(True)
    t1 = time()

    for train_batch in dataModule.train_loader:
        all_item = [item.to(config.device) for item in train_batch]
        inputs, label = all_item[:-1], all_item[-1]

        # 优化器在 real_model (原始模型) 里
        real_model.optimizer.zero_grad()

        if config.use_amp:
            with torch.amp.autocast(device_type=config.device):
                # !!! 关键点 !!!
                # 这里调用的是 model (可能是多卡包装器)，而不是 real_model
                # 这样 inputs 才会自动切分到多张卡上
                pred = model(*inputs)
                loss = compute_loss(real_model, pred, label, config)

            real_model.scaler.scale(loss).backward()
            real_model.scaler.step(real_model.optimizer)
            real_model.scaler.update()
        else:
            pred = model(*inputs)
            loss = compute_loss(real_model, pred, label, config)
            loss.backward()
            real_model.optimizer.step()

    t2 = time()
    return loss.cpu().item(), t2 - t1


def exec_evaluate_one_epoch(model, dataModule, config, mode="valid"):
    real_model = model.module if isinstance(model, torch.nn.DataParallel) else model

    model.eval()
    torch.set_grad_enabled(False)
    dataloader = (
        dataModule.valid_loader
        if mode == "valid" and len(dataModule.valid_loader.dataset) != 0
        else dataModule.get_testloader()
    )
    preds, reals, val_loss = [], [], 0.0

    context = (
        torch.amp.autocast(device_type=config.device)
        if config.use_amp
        else contextlib.nullcontext()
    )
    with context:
        for batch in dataloader:
            all_item = [item.to(config.device) for item in batch]
            inputs, label = all_item[:-1], all_item[-1]

            # 多卡并行推理
            pred = model(*inputs)

            if mode == "valid":
                val_loss += compute_loss(real_model, pred, label, config)

            if config.classification:
                pred = torch.max(pred, 1)[1]

            preds.append(pred)
            reals.append(label)

    reals = torch.cat(reals, dim=0)
    preds = torch.cat(preds, dim=0)

    if config.dataset != "weather":
        reals, preds = dataModule.y_scaler.inverse_transform(
            reals
        ), dataModule.y_scaler.inverse_transform(preds)

    if mode == "valid":
        if isinstance(real_model.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            real_model.scheduler.step(val_loss)
        else:
            real_model.scheduler.step()

    return ErrorMetrics(reals, preds, config)
