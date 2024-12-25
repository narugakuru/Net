import os
import re
import csv


def extract_logger_data(base_dir, output_csv):
    # 获取所有子文件夹路径，并按自然排序
    def natural_sort_key(s):
        return [
            int(text) if text.isdigit() else text.lower()
            for text in re.split(r"(\d+)", s)
        ]

    folder_paths = sorted(
        [
            os.path.join(base_dir, folder)
            for folder in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, folder))
        ],
        key=lambda x: natural_sort_key(os.path.basename(x)),
    )

    # 定义正则表达式
    pattern = re.compile(
        r"{'LIVER': [0-9\.]+, 'RK': [0-9\.]+, 'LK': [0-9\.]+, 'SPLEEN': [0-9\.]+}"
    )

    # 打开CSV文件准备写入
    with open(output_csv, "w", newline="") as csvfile:
        headers = ["exp", "LIVER", "RK", "LK", "SPLEEN", "mean"]
        csvwriter = csv.DictWriter(csvfile, fieldnames=headers)
        csvwriter.writeheader()

        # 遍历每个文件夹
        for folder in folder_paths:
            log_file_path = os.path.join(folder, "logger.log")
            if os.path.exists(log_file_path):
                # 提取实验号码（文件夹名称）
                exp_number = os.path.basename(folder)

                # 读取 logger.log 文件的内容
                with open(log_file_path, "r") as file:
                    lines = file.readlines()

                # 获取倒数 n 行的内容
                last_five_lines = lines[-3:]
                data = "".join(last_five_lines)

                # 提取匹配的内容
                matches = pattern.findall(data)

                # 将提取的内容解析为字典并计算平均值
                for match in matches:
                    data_dict = eval(match)
                    data_dict = {k: round(v, 4) for k, v in data_dict.items()}
                    mean_value = round(sum(data_dict.values()) / len(data_dict), 4)
                    data_dict["mean"] = mean_value
                    data_dict["exp"] = exp_number  # 添加实验号码

                    # 写入CSV
                    csvwriter.writerow(data_dict)

    print(f"Data has been written to {output_csv}")


# 示例调用
base_directory = "./runs/GMRD_CHAOST2_CV0_val"  # 替换为实际目录路径
output_csv_path = "output.csv"
extract_logger_data(base_directory, output_csv_path)
