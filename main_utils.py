import subprocess
import functools
import gc
import os
import psutil  # 用于系统内存管理
import torch
import csv
import re

def find_max_mean_row(file_path):
    """找到mean列的最大值
    参数:
        file_path (str): 输入的CSV文件路径
    """
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
    """动态获取最新所有模型路径
    参数:
        prepath (str): 要搜索的父路径
    返回:
        List[str]: 最新快照文件的路径列表
    """
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


def get_best_snapshot_files(prepath):
    """动态获取最新所有模型路径
    参数:
        prepath (str): 要搜索的父路径
    返回:
        str: 最新快照文件的路径列表
    """
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

    if os.path.exists(latest_dir_path):
        # 获取snapshots文件夹下的所有文件路径
        snapshot_files = [
            os.path.join(latest_dir_path, f) for f in os.listdir(latest_dir_path)
        ]
        # 使用/作为路径分隔符
        snapshot_files = [file.replace("\\", "/") for file in snapshot_files]

        # 检查是否存在best.pth文件
        best_pth_path = os.path.join(latest_dir_path, "best.pth").replace("\\", "/")
        if best_pth_path in snapshot_files:
            return best_pth_path

    return []


def get_all_train_best(run_path):
    """
    获取 run_path 路径下所有名称带 "train" 的文件夹，
    并对每个文件夹调用 get_best_snapshot_files 函数，
    最终返回包含所有 best.pth 路径的列表。
    """
    best_paths = []  # 用于存储所有 best.pth 的路径
    # 检查 run_path 是否存在
    if os.path.exists(run_path) and os.path.isdir(run_path):
        # 遍历 run_path 下的所有文件和文件夹
        for item in os.listdir(run_path):
            item_path = os.path.join(run_path, item).replace("\\", "/")
            # 检查是否是文件夹且名称中包含 "train"
            if os.path.isdir(item_path) and "train" in item:
                # 调用 get_best_snapshot_files 函数获取 best.pth 路径
                best_pth = get_best_snapshot_files(item_path)
                if best_pth:  # 如果返回的路径不为空
                    best_paths.append(best_pth)
    return best_paths


def config_procee(config_updates):
    """转换配置为Sacred格式
    参数:
        config_updates (dict): 包含配置项的字典
    返回:
        List[str]: 转换后的配置参数列表
    """
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
    """清理GPU和系统内存的装饰器
    参数:
        func (function): 需要装饰的函数
    返回:
        function: 包装后的函数
    """

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
    """运行训练过程
    参数:
        config_updates (dict): 更新的训练配置
    """
    config_args = config_procee(config_updates)
    command = ["python", "train.py"] + config_args
    # /home/raisei/GMRD/
    print("Executing command:", " ".join(command))
    subprocess.run(command, check=True)


@memory_cleanup
def run_test(config_updates):
    """运行测试过程
    参数:
        config_updates (dict): 更新的测试配置
    """
    config_args = config_procee(config_updates)
    command = ["python", "test.py"] + config_args
    print("Executing command:", " ".join(command))
    subprocess.run(command, check=True)


def extract_logger_data_bk(base_dir, output_csv):
    """获取所有子文件夹路径，并按自然排序，提取所有log日志的精度
    参数:
        base_dir (str): 基础目录路径
        output_csv (str): 输出CSV文件路径
    """

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
                        # 乘以100，保留小数点后2位
                        values = values * 100
                        row["mean"] = "{:.2f}".format(sum(values) / len(values))
                        csvwriter.writerow(row)
                    except Exception as e:
                        print(f"处理文件 {log_file_path} 时出错: {e}")

    print(f"Data has been written to {output_csv}")
    find_max_mean_row(output_csv)


import os
import re
import csv


def natural_sort_key(s):
    """
    自然排序的辅助函数，将字符串中的数字部分提取出来用于排序。
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", s)
    ]


def extract_logger_data(base_dir, output_csv):
    # 获取所有子文件夹路径，并按自然排序
    subdirs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    subdirs.sort(key=natural_sort_key)  # 使用自然排序

    # 准备写入CSV的数据
    data_to_write = []

    for subdir in subdirs:
        log_file = os.path.join(subdir, "logger.log")
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                lines = f.readlines()
                last_three_lines = lines[-3:]  # 获取倒数三行

                # 提取字典中的键值对
                for line in last_three_lines:
                    match = re.search(r"\{.*\}", line)
                    if match:
                        dict_str = match.group(0)
                        try:
                            # 将字符串转换为字典
                            dict_data = eval(dict_str)
                            # 动态提取键值对
                            row = {"exp": os.path.basename(subdir)}  # 子文件夹名称
                            values = []
                            for key, value in dict_data.items():
                                value *= 100
                                row[key] = round(float(value), 2)  # 保留小数点后2位
                                values.append(float(value))
                            # 计算均值
                            value *= 100
                            row["mean"] = (
                                round(sum(values) / len(values), 2) if values else 0
                            )
                            data_to_write.append(row)
                        except:
                            continue

    # 动态生成CSV的列名
    fieldnames = ["exp"]
    if data_to_write:
        # 从第一个数据行中提取所有键（除了 'exp' 和 'mean'）
        keys = set()
        for row in data_to_write:
            keys.update(row.keys())
        keys.discard("exp")
        keys.discard("mean")
        fieldnames.extend(sorted(keys))  # 按字母顺序排序
        fieldnames.append("mean")

    # 将数据写入CSV文件
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data_to_write:
            writer.writerow(row)


def get_files(path):
    from pathlib import Path
    import os  # 仍然需要 os 来处理可能的环境变量或用户目录，但 pathlib 更擅长路径操作

    # 定义目标路径
    # 使用 Path 对象可以更好地处理不同操作系统的路径分隔符
    target_dir_path = Path(path)
    # 如果路径包含 ~ 或环境变量，可以先解析
    # target_dir_path = Path(os.path.expanduser(os.path.expandvars("./runs/GMRD_CHAOST2_CV1_train/2/snapshots")))

    file_list = []

    try:
        # 检查路径是否存在并且是一个目录
        if target_dir_path.is_dir():
            # 遍历目录中的所有条目
            for entry in target_dir_path.iterdir():
                # 检查条目是否是文件（而不是子目录）
                if entry.is_file():
                    # 将文件的完整路径（字符串形式）添加到列表中
                    file_list.append(str(entry))
                    # 如果你只需要文件名，可以使用 entry.name
                    # file_list.append(entry.name)

            print(f"成功找到 {len(file_list)} 个文件:")
            # 打印列表内容（可选）
            for file_path in file_list:
                print(file_path)

        else:
            print(f"错误：路径 '{target_dir_path}' 不是一个有效的目录或不存在。")

    except FileNotFoundError:
        print(f"错误：路径 '{target_dir_path}' 不存在。")
    except Exception as e:
        print(f"发生未知错误: {e}")

    # 现在 file_list 变量包含了所有文件的路径列表
    print("\n获取的文件列表:")
    print(file_list)
    return file_list


def output_all_csv(run_dir="./runs"):
    # 定义 ./run 目录路径，方便修改
    folders = os.listdir(run_dir)
    for folder_name in folders:  # 循环遍历的是文件夹名称列表
        folder_path = os.path.join(run_dir, folder_name)  # 构建完整的文件夹路径
        if os.path.isdir(folder_path) and "val" in folder_name:
            output_csv_name = folder_name + ".csv"  #  CSV 文件名与文件夹名相同
            output_csv_path = os.path.join(
                run_dir, output_csv_name
            )  #  CSV 文件保存在 ./run 目录下，与文件夹同级
            extract_logger_data(folder_path, output_csv_path)


if __name__ == "__main__":

    run_dir = "./runs"  # 定义 ./run 目录路径，方便修改
    folders = os.listdir(run_dir)
    for folder_name in folders:  # 循环遍历的是文件夹名称列表
        folder_path = os.path.join(run_dir, folder_name)  # 构建完整的文件夹路径
        if os.path.isdir(folder_path) and "val" in folder_name:
            output_csv_name = folder_name + ".csv"  #  CSV 文件名与文件夹名相同
            output_csv_path = os.path.join(
                run_dir, output_csv_name
            )  #  CSV 文件保存在 ./run 目录下，与文件夹同级
            extract_logger_data(folder_path, output_csv_path)
