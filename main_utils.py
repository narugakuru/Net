import subprocess
import functools
import gc
import os
import psutil  # 用于系统内存管理
import torch
import csv
import re


def find_max_mean_row(file_path):
    """找到mean列的最大值"""
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


def get_latest_snapshot_files(prepath):
    """动态获取最新所有模型路径"""
    # 获取prepath路径下的所有文件夹
    subdirs = [
        d for d in os.listdir(prepath) if os.path.isdir(os.path.join(prepath, d))
    ]

    # 过滤出以数字命名的文件夹，并按自然数字大小排序
    numeric_subdirs = sorted((int(d) for d in subdirs if d.isdigit()), reverse=True)

    if not numeric_subdirs:
        return []

    # 选择最大的数字文件夹
    latest_dir = str(numeric_subdirs[0])
    latest_dir_path = os.path.join(prepath, latest_dir, "snapshots")

    # 获取snapshots文件夹下的所有文件路径
    if os.path.exists(latest_dir_path):
        snapshot_files = [
            os.path.join(latest_dir_path, f) for f in os.listdir(latest_dir_path)
        ]
        # 使用/作为路径分隔符
        snapshot_files = [file.replace("\\", "/") for file in snapshot_files]
        return snapshot_files
    else:
        return []


def config_procee(config_updates):
    """转换配置为Sacred格式"""
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


def extract_logger_data(base_dir, output_csv):
    """获取所有子文件夹路径，并按自然排序，提取所有log日志的精度"""

    def natural_sort_key(s):
        return [
            int(text) if text.isdigit() else text.lower()
            for text in re.split(r"(\d+)", s)
        ]

    folder_paths = sorted(
        [
            os.path.join(base_dir, folder).replace("\\", "/")
            for folder in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, folder))
        ],
        key=lambda x: natural_sort_key(os.path.basename(x)),
    )
    print(f"folder_paths: {folder_paths}")

    # 修改正则表达式以匹配任意字典格式
    pattern = re.compile(r"{[^}]+}")

    with open(output_csv, "w", newline="") as csvfile:
        # 先读取第一个日志文件来确定表头
        first_log = None
        for folder in folder_paths:
            log_file_path = f"{folder}/logger.log"
            if os.path.exists(log_file_path):
                with open(log_file_path, "r") as file:
                    content = file.read()
                    matches = pattern.findall(content)
                    if matches:
                        first_log = matches[-1]  # 获取最后一个字典
                        break

        if not first_log:
            print("未找到任何有效的数据")
            return

        # 从第一个日志文件中提取键作为表头
        first_dict = eval(first_log)
        headers = ["exp"] + list(first_dict.keys()) + ["mean"]

        csvwriter = csv.DictWriter(csvfile, fieldnames=headers)
        csvwriter.writeheader()

        for folder in folder_paths:
            log_file_path = f"{folder}/logger.log"

            if os.path.exists(log_file_path):
                with open(log_file_path, "r") as file:
                    content = file.read()
                    matches = pattern.findall(content)

                if matches:
                    last_dict_str = matches[-1]  # 获取最后一个字典
                    try:
                        data_dict = eval(last_dict_str)
                        # 将字典中的数值格式化为4位小数
                        formatted_dict = {
                            k: "{:.4f}".format(float(v)) for k, v in data_dict.items()
                        }
                        exp_name = os.path.basename(folder)
                        row = {"exp": exp_name}
                        row.update(formatted_dict)
                        # 计算平均值并格式化为4位小数
                        values = [float(v) for v in data_dict.values()]
                        row["mean"] = "{:.4f}".format(sum(values) / len(values))
                        csvwriter.writerow(row)
                    except Exception as e:
                        print(f"处理文件 {log_file_path} 时出错: {e}")

    print(f"Data has been written to {output_csv}")
    find_max_mean_row(output_csv)
