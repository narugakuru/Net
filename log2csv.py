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
                                row[key] = round(float(value), 4)  # 保留小数点后4位
                                values.append(float(value))
                            # 计算均值
                            row["mean"] = (
                                round(sum(values) / len(values), 4) if values else 0
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


# 示例调用
base_dir = r"E:\CodeAchieve\PaperCode\GMRD\runs\GMRD_CMR_CV0_val"
output_csv = "output.csv"
extract_logger_data(base_dir, output_csv)
