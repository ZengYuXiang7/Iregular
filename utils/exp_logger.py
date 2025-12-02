# coding : utf-8
# Author : yuxiang Zeng

import os
import platform
import shutil
import time
import logging
import pickle
import numpy as np
import torch
from exp.exp_efficiency import *
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, filename, exper_detail, plotter, config, show_params=True):
        self.filename = filename
        self.exper_detail = exper_detail
        self.plotter = plotter
        self.config = config
        self._init_log_file()
        if config.hyper_search:
            self.exper_filename += "_hyper_search"
        # 设置日志记录到文件
        logging.basicConfig(
            level=logging.INFO,
            filename=f"{self.exper_filename}.md",
            filemode="w",
            format="%(message)s",
        )
        self.logger = logging.getLogger(config.model)
        config.log = self
        self._prepare_experiment(show_params)

        # 创建tensorboard
        self.base_log_dir = os.path.join("./runs", config.model, self.filename)
        self.tb_writer = None  # 先占个位，不在 init 里创建

    # 初始化日志文件路径
    def _init_log_file(self):
        fileroot = f"./results/{self.config.model}/" + time.strftime("%Y%m%d") + "/log/"
        os.makedirs(fileroot, exist_ok=True)
        timestamp = time.strftime("%H_%M_%S")
        self.exper_filename = os.path.join(fileroot, f"{timestamp}_{self.filename}")

    # 打印初始配置参数
    def _prepare_experiment(self, show_params):
        self.logger.info("```python")
        if show_params:
            self.log(self._format_config_dict(self.config.__dict__))

    # 保存运行日志到run.log文件
    def save_in_log(self, metrics):
        # === 获取 CPU 名称 ===
        cpu_name = platform.processor()
        if not cpu_name:
            cpu_name = platform.machine()
        device_name = f"CPU-{cpu_name}"

        # === 如果有 GPU，添加 GPU 名称 ===
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            device_name += f"_GPU-{gpu_name}"

        # === 清理非法字符，生成文件名 tag ===
        device_tag = (
            device_name.replace(" ", "-")
            .replace("/", "-")
            .replace("(", "")
            .replace(")", "")
        )

        # === 构造日志文件路径 ===
        log_path = f"./{device_tag}_{self.config.logger}"

        with open(f"./{log_path}.log", "a") as f:
            timestamp = time.strftime("|%Y-%m-%d %H:%M:%S| ")
            f.write(timestamp + self.exper_detail + "\n")
            metric_str = " ".join(
                [f"{k} - {np.mean(v):.4f}" for k, v in metrics.items()]
            )
            f.write(timestamp + metric_str + "\n")

    # 保存结果到pickle文件
    def save_result(self, metrics):
        os.makedirs("./results/metrics/", exist_ok=True)
        config_copy = {k: v for k, v in self.config.__dict__.items() if k != "log"}
        result = {
            "config": config_copy,
            "dataset": self.config.model,
            "model": self.config.model,
            **{k: metrics[k] for k in metrics},
            **{f"{k}_mean": np.mean(metrics[k]) for k in metrics},
            **{f"{k}_std": np.std(metrics[k]) for k in metrics},
        }
        with open(f"./results/metrics/{self.filename}.pkl", "wb") as f:
            pickle.dump(result, f)

    # 日志输出（含彩色打印）
    def log(self, string):
        if string.startswith("\n"):
            string = string[1:]
            print("\n", end="")
            self.logger.info("")
        timestamp = time.strftime("|%Y-%m-%d %H:%M:%S| ")
        self.logger.info(timestamp + string)
        self.only_print(string)

    def __call__(self, string):
        self.log(string)

    # 终端彩色输出辅助函数
    def only_print(self, string):
        timestamp = time.strftime("|%Y-%m-%d %H:%M:%S| ")
        print(f"\033[1;38;2;151;200;129m{timestamp}\033[0m\033[1m{string}\033[0m")

    # 展示一次完整实验结果
    def show_results(self, results, sum_time):
        monitor = self.config.monitor_metric
        summary = f"Valid{monitor}={-results[monitor]:.4f} ｜ "
        summary += " ".join([f"{k}={v:.4f}" for k, v in results.items()])
        summary += f" time={sum_time:.1f} s"
        self.only_print(summary)

    # 展示训练中的某轮 epoch 的误差
    def show_epoch_error(self, runid, epoch, monitor, epoch_loss, results, train_time):
        if self.config.verbose and epoch % self.config.verbose == 0 and epoch > 0:
            self.only_print(self.exper_detail)
            best = f"Best Epoch {monitor.best_epoch} {self.config.monitor_metric} = {-monitor.best_score:.4f}  now = {epoch - monitor.best_epoch}"
            self.only_print(best)
            summary = f"Round={runid + 1} Epoch={epoch + 1:03d} Loss={epoch_loss:.4f} "
            summary += " ".join([f"v{k}={v:.4f}" for k, v in results.items()])
            summary += f" time={sum(train_time):.1f} s"
            self.only_print(summary)
        self.log_tensorboard(epoch, epoch_loss, results, train_time)

    # 展示最终测试结果
    def show_test_error(self, runid, monitor, results, sum_time):
        summary = f"Round={runid + 1} BestEpoch={monitor.best_epoch:3d} "
        summary += f"Valid{self.config.monitor_metric}={-monitor.best_score:.4f} ｜ "
        summary += " ".join([f"{k}={v:.4f}" for k, v in results.items()])
        summary += f" time={sum_time:.1f} s"
        self.log(summary)

        # === 2. TensorBoard 横向 Markdown 表格 (修改版) ===
        if self.tb_writer:
            # --- 第一步：准备列表 ---
            # 1. 核心指标
            headers = ["Best Epoch", f"Valid {self.config.monitor_metric}"]
            values = [f"`{monitor.best_epoch}`", f"`{-monitor.best_score:.4f}`"]

            # 2. 循环添加其他所有指标
            for k, v in results.items():
                headers.append(k)  # 表头
                values.append(f"`{v:.4f}`")  # 数值(加反引号高亮)

            # 3. 添加时间
            headers.append("Total Time")
            values.append(f"`{sum_time:.1f} s`")

            # --- 第二步：拼接 Markdown ---
            # 标题
            md_table = f"### 🏆 Round {runid + 1} Test Summary\n\n"

            # 拼表头: | Best Epoch | Valid RMSE | MAE | ... |
            md_table += "| " + " | ".join(headers) + " |\n"

            # 拼分割线: | :--- | :--- | :--- | ... | (根据列数自动生成)
            md_table += "| " + " | ".join([":---"] * len(headers)) + " |\n"

            # 拼数值行: | 123 | 0.8321 | 2.14 | ... |
            md_table += "| " + " | ".join(values) + " |\n"

            # --- 第三步：写入 ---
            self.tb_writer.add_text("Summary/Test_Results", md_table, 0)
            self.tb_writer.flush()  # 强制写入硬盘

    # 配置参数格式化输出
    def _format_config_dict(self, config_dict, items_per_line=3):
        sorted_items = sorted(config_dict.items())
        lines = [
            ", ".join([f"'{k}': {v}" for k, v in sorted_items[i : i + items_per_line]])
            for i in range(0, len(sorted_items), items_per_line)
        ]
        return "{\n" + "\n".join(["     " + line for line in lines]) + "\n}"

    # 删除空文件夹
    def _delete_empty_directories(self, dir_path):
        # 检查目录是否存在
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            # 遍历目录中的所有文件和子目录，从最底层开始
            for root, dirs, files in os.walk(dir_path, topdown=False):
                # 先删除空的子目录
                for name in dirs:
                    dir_to_remove = os.path.join(root, name)
                    # 如果目录是空的，则删除它
                    try:
                        if not os.listdir(dir_to_remove):  # 判断目录是否为空
                            os.rmdir(dir_to_remove)
                            print(f"Directory {dir_to_remove} has been deleted.")
                    except FileNotFoundError:
                        # 如果目录已经不存在，忽略此错误
                        pass
                # 检查当前目录是否也是空的，如果是则删除它
                try:
                    if not os.listdir(root):  # 判断当前根目录是否为空
                        os.rmdir(root)
                        print(f"Directory {root} has been deleted.")
                except FileNotFoundError:
                    # 如果目录已经不存在，忽略此错误
                    pass
        else:
            print(f"Directory {dir_path} does not exist.")

    # 实验结束时执行的清理操作
    def end_the_experiment(self, model):
        self.logger.info(f"\n{str(model)}")
        self.logger.info("```")
        self._delete_empty_directories("./results/")

    ###############
    def set_runid(self, runid):
        # 1. 之前如果有 writer，先关掉
        if self.tb_writer is not None:
            self.tb_writer.close()

        # 2. 确定文件夹路径
        log_dir = os.path.join(self.base_log_dir, f"Round_{runid}")

        # === [新增] 如果文件夹已存在，直接删除，防止旧数据干扰 ===
        if os.path.exists(log_dir):
            try:
                shutil.rmtree(log_dir)  # 递归删除文件夹
                print(f"Cleaned up old logs in: {log_dir}")
            except OSError as e:
                print(f"Error: {log_dir} : {e.strerror}")
        # ======================================================

        # 3. 重新创建全新的文件夹和 writer
        self.tb_writer = SummaryWriter(log_dir=log_dir)

        return True

    # === 1. 记录超参数 (Hyperparameters) ===
    # 这样你可以在 TensorBoard 的 "HPARAMS" 栏目里筛选出最好的参数组合
    def log_hparams(self, config, metrics):
        # 过滤掉 config 中无法序列化的对象，只留 int, float, str, bool
        hparam_dict = {
            k: v
            for k, v in config.__dict__.items()
            if isinstance(v, (int, float, str, bool))
        }

        # 你的 metrics 可能是 list，这里取均值作为最终指标
        metric_dict = {k: np.mean(v) for k, v in metrics.items()}

        if self.tb_writer:
            self.tb_writer.add_hparams(hparam_dict, metric_dict)

    # === 2. 记录模型结构 (修复版) ===
    def log_model_graph(self, model, datamodule, device):
        if self.tb_writer:
            try:
                # 1. 拿一个 batch 的数据
                batch = next(iter(datamodule.train_loader))

                # 2. 预处理数据
                # 假设 batch 是一个 list: [feat, eig, in, out, dij, label]
                # 我们需要把它们都挪到 device 上
                if isinstance(batch, (list, tuple)):
                    batch = [x.to(device) for x in batch]

                    # 【关键点】
                    # 通常 batch 的最后一个元素是 label/target，模型 forward 不需要它
                    # 如果你的 forward 刚好需要 batch 里除最后一个以外的所有元素：
                    inputs_to_model = tuple(batch[:-1])

                    # ⚠️如果模型需要 batch 里所有的元素（没有单独的 label），就用下面这行：
                    # inputs_to_model = tuple(batch)
                else:
                    # 如果 batch 本身就是单个 tensor
                    inputs_to_model = batch.to(device)

                # 3. 传入 Tuple，add_graph 会自动解包成多个参数传给 forward
                self.tb_writer.add_graph(model, input_to_model=inputs_to_model)
                print("Success: Model graph added to TensorBoard.")

            except Exception as e:
                # 如果还是报错，打印出来方便调试，不卡死程序
                print(f"Warning: Failed to add model graph to TensorBoard. Error: {e}")

    # === 3. 记录权重和梯度直方图 (Histograms) ===
    # 用于检查梯度消失/爆炸，或者权重是否更新
    def log_histograms(self, model, epoch):
        if self.tb_writer:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # 记录权重值分布
                    self.tb_writer.add_histogram(f"Weights/{name}", param, epoch)
                    # 记录梯度值分布 (只有在 backward 之后才有值)
                    if param.grad is not None:
                        self.tb_writer.add_histogram(
                            f"Gradients/{name}", param.grad, epoch
                        )

    # === 4. 记录配置文本 (Markdown 表格版：每行 5 个) ===
    def log_config_text(self, config):
        if self.tb_writer:
            # 1. 准备数据
            # 格式化为 "**Key**: `Value`" 的形式
            # 过滤掉以 '_' 开头的私有属性（可选）
            params = [
                f"**{k}**: `{v}`"
                for k, v in sorted(config.__dict__.items())
                if not k.startswith("_")
            ]

            # 2. 核心逻辑：每 5 个参数一行
            COLUMNS = 5

            # 构建表头 | P1 | P2 | P3 | P4 | P5 |
            headers = [f"Param {i+1}" for i in range(COLUMNS)]
            md_table = "### Experiment Configuration\n\n"
            md_table += "| " + " | ".join(headers) + " |\n"
            md_table += "| " + " | ".join([":---"] * COLUMNS) + " |\n"  # 左对齐

            # 3. 填充数据行
            for i in range(0, len(params), COLUMNS):
                # 取出当前行的切片
                row_items = params[i : i + COLUMNS]

                # 如果不足 5 个，用空字符串补齐，否则 Markdown 表格会乱
                if len(row_items) < COLUMNS:
                    row_items += [""] * (COLUMNS - len(row_items))

                # 拼接这一行
                md_table += "| " + " | ".join(row_items) + " |\n"

            # 4. 写入
            self.tb_writer.add_text("Config_Details", md_table, 0)
            self.tb_writer.flush()

    # === 5. (可选) 记录 Embedding 投影 ===
    # 如果你的模型有 Encoder 输出特征，想看这些特征在空间里怎么聚类的
    def log_embeddings(self, features, labels, epoch):
        if self.tb_writer:
            # features: [N, D_model], labels: [N] (用于着色)
            self.tb_writer.add_embedding(features, metadata=labels, global_step=epoch)

    # [新增] 专门用于写 TensorBoard 的函数
    def log_tensorboard(self, epoch, train_loss, results, train_time):
        if self.tb_writer:
            # 1. 记录 Loss
            self.tb_writer.add_scalar(
                f"Metrics/{self.config.loss_func}", train_loss, epoch
            )
            # 2. 记录所有验证指标 (results 是一个字典)
            for key, value in results.items():
                self.tb_writer.add_scalar(f"Metrics/{key}", value, epoch)

    # 实验结束后记得关闭
    def close(self):
        self.tb_writer.close()
