import subprocess
import functools
import gc
import os
import psutil  # 用于系统内存管理
import torch
from log2csv import extract_logger_data
import csv


def find_max_mean_row(file_path):
    max_mean_value = float("-inf")
    max_mean_row = None

    with open(file_path, mode="r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            mean_value = float(row["mean"])
            if mean_value > max_mean_value:
                max_mean_value = mean_value
                max_mean_row = row

    print(max_mean_row)


def config_procee(config_updates):
    # 转换配置为Sacred格式
    config_args = ["with"]
    for key, value in config_updates.items():
        if isinstance(value, str):
            config_args.append(f'{key}="{value}"')
        else:
            config_args.append(f"{key}={value}")

    return config_args


def clean_memory():
    """清理系统内存和GPU内存"""
    # 清理Python对象
    gc.collect()

    # 清理GPU内存
    try:

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass

    # 建议系统进行内存回收
    try:
        process = psutil.Process(os.getpid())
        process.memory_info()
        if os.name != "nt":  # Linux
            if hasattr(os, "sync"):
                os.sync()
        else:  # Windows系统
            import ctypes

            ctypes.windll.kernel32.GlobalMemoryStatus()  # 刷新Windows内存状态

    except Exception:
        pass


def memory_cleanup(func):
    """清理GPU和系统内存的装饰器"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # 运行前清理
            # clean_memory()
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            print("\n程序被中断，正在清理内存...")
            clean_memory()
            raise
        except Exception as e:
            print(f"发生错误: {e}")
            clean_memory()
            raise
        finally:
            pass

    return wrapper


@memory_cleanup
def run_train(config_updates):
    config_args = config_procee(config_updates)
    command = ["python", "train.py"] + config_args
    print("Executing command:", " ".join(command))
    subprocess.run(command, check=True)


@memory_cleanup
def run_test(config_updates):
    config_args = config_procee(config_updates)
    command = ["python", "test.py"] + config_args
    print("Executing command:", " ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":

    eval_fold_list = [0, 1, 2, 3, 4]
    eval_fold_list = eval_fold_list[:1]
    dataset = "CMR"

    for eval_fold in eval_fold_list:

        prepath = f"./runs/GMRD_{dataset}_CV{eval_fold}_train"

        # 配置参数保持不变
        train_config_updates = {
            "reload_model_path": "{prepath}/1/snapshots/9000.pth",
            "n_steps": 24000,
            "mode": "train",
            "eval_fold": eval_fold,
            "dataset": dataset,
            "save_snapshot_every": 2000,
        }
        # 运行脚本
        print("Running training...")
        run_train(train_config_updates)

        # 验证模型测试
        val_model = range(3, 24, 3)
        # val_model = [7]
        for i in val_model:
            test_config_updates = {
                "reload_model_path": f"{prepath}/2/snapshots/{i}000.pth",
                # "reload_model_path": f"{prepath}/4/snapshots/7000.pth",
                "mode": "val",
                "dataset": dataset,
                "eval_fold": eval_fold,
            }

            print("Running testing...")
            run_test(test_config_updates)

        # 提取日志数据
        base_directory = f"./runs/GMRD_{dataset}_CV{eval_fold}_val"
        output_csv_path = "GMRD_{dataset}_CV{eval_fold}_val.csv"
        extract_logger_data(base_directory, output_csv_path)
        # 输出最高Dice值行
        find_max_mean_row(output_csv_path)
