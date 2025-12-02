# coding : utf-8
# Author : Yuxiang Zeng
import os
import torch
from tqdm import *
import pickle
from exp.exp_base import BasicModel, exec_evaluate_one_epoch, exec_train_one_epoch
from utils.model_monitor import EarlyStopping
from utils.exp_logger import Logger
from data_provider.data_center import DataModule


def RunOnce(config, runid, model: BasicModel, datamodule: DataModule, log: Logger):

    # 创建保存模型的目录
    os.makedirs(f"./checkpoints/{config.model}", exist_ok=True)
    model_path = f"./checkpoints/{config.model}/{log.filename}_round_{runid}.pt"

    # 判断是否需要重新训练：
    # 若 config.retrain==1 表示强制重训；
    # 或者模型文件不存在 且 设置了 continue_train，则需要重新训练
    retrain_required = (
        config.retrain == 1 or not os.path.exists(model_path) and config.continue_train
    )

    # 如果无需重新训练且已有模型文件，则直接加载模型并评估测试集性能
    if not retrain_required:
        try:
            # 加载之前记录的训练时间
            sum_time = pickle.load(
                open(f"./results/metrics/" + log.filename + ".pkl", "rb")
            )["train_time"][runid]
            # 加载模型权重（weights_only=True 可忽略 optimizer 等无关信息）
            model.load_state_dict(
                torch.load(model_path, weights_only=True, map_location="cpu")
            )
            model.setup_optimizer(config)  # 重新设置优化器
            results = model.evaluate_one_epoch(datamodule, "test")  # 在测试集评估性能
            log.show_results(results, sum_time)
            config.record = False  # 不再记录当前结果
        except Exception as e:
            log.only_print(f"Error: {str(e)}")
            retrain_required = True  # 若加载失败则触发重新训练

    # 若设置为继续训练（即接着上次的结果继续）
    if config.continue_train:
        log.only_print(f"Continue training...")
        model.load_state_dict(
            torch.load(model_path, weights_only=True, map_location="cpu")
        )

    # 若需要重新训练
    if retrain_required:
        results = full_retrain(
            config,
            model,
            datamodule,
            log,
            runid,
            model_path,
        )

    return results


def full_retrain(
    config,
    model: BasicModel,
    datamodule: DataModule,
    log: Logger,
    runid: int,
    model_path: str,
):

    if config.device != "cpu":
        if torch.cuda.device_count() > 1 and config.multi_gpu:
            log.only_print(f"使用 DataParallel, GPU 数量：{torch.cuda.device_count()}")
            model = torch.nn.DataParallel(model)
            model.module.setup_optimizer(config)
        else:
            log.only_print("单 GPU 训练模式")
            model.to(config.device)
            model.setup_optimizer(config)
            # try:
            # model.compile()
            # except Exception as e:
            # print(f"Skip the model.compile() because {e}")

    # 设置EarlyStopping监控器
    monitor = EarlyStopping(config)

    train_time = []

    if config.epochs != 0:
        t = trange(config.epochs, leave=True)
        for epoch in t:
            if monitor.early_stop:
                break  # 若满足early stopping条件则提前终止训练

            # 训练一个epoch并记录耗时
            train_loss, time_cost = exec_train_one_epoch(model, datamodule, config)

            train_time.append(time_cost)

            # 验证集上评估当前模型误差
            valid_error = exec_evaluate_one_epoch(
                model, datamodule, config, mode="valid"
            )

            # 将当前epoch的验证误差传递给early stopping模块进行跟踪
            monitor.track_one_epoch(epoch, model, valid_error, config.monitor_metric)

            # 输出当前epoch的训练误差和验证误差，并记录训练时间
            log.show_epoch_error(
                runid, epoch, monitor, train_loss, valid_error, train_time
            )

            # 更新日志可视化（如绘图）
            log.plotter.append_epochs(train_loss, valid_error)

            # 暂存模型参数（即使不是最优，也为了中断续训做准备）
            torch.save(model.state_dict(), model_path)

            # tqdm的后缀显示
            be, delta_epochs = monitor.best_epoch or epoch, monitor.counter
            metric_name, best_val_str = getattr(config, "monitor_metric", "val_loss"), (
                f"{monitor.best_score:.4f}" if monitor.best_score is not None else "N/A"
            )

            # 右侧后缀信息
            t.set_description(
                f"Training: Best@{be} {metric_name}={best_val_str} Δ={delta_epochs}"
            )

        # 加载最优模型参数（来自early stopping）
        model.load_state_dict(monitor.best_model)

        # 累计训练时间（仅使用前best_epoch轮）
        sum_time = sum(train_time[: monitor.best_epoch])

        # 使用最优模型在测试集评估
        results = exec_evaluate_one_epoch(model, datamodule, config, mode="test")

        # 2025年09月10日15:59:43 专门给排序做时延
        # results = model.evaluate_whole_dataset(datamodule, 'whole')

        log.show_test_error(runid, monitor, results, sum_time)

        # 保存最优模型参数
        torch.save(monitor.best_model, model_path)
        log(f"Model parameters saved to {model_path}")
        print("-" * 130)

    elif config.epochs == 0:
        # 直接在未训练的模型上评估测试集性能
        log("Directly evaluate untrained model on test set...")
        results = exec_evaluate_one_epoch(model, datamodule, config, mode="test")
        monitor.best_epoch, monitor.best_score = -1, 0.0
        log.show_test_error(runid=-1, monitor=monitor, results=results, sum_time=0.0)
        log("*" * 20 + "Experiment Exit" + "*" * 20)
        exit(0)

    # 将训练时间加入返回结果中
    results["train_time"] = sum_time
    return results
