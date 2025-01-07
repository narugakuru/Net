from main_utils import *


if __name__ == "__main__":

    # eval_fold_list = [0, 1, 2, 3, 4]
    eval_fold_list = [0]
    dataset = "CHAOST2"  # CMR CHAOST2 SABS

    for eval_fold in eval_fold_list:
        #### 训练 ####
        prepath = f"./runs/GMRD_{dataset}_CV{eval_fold}_train"

        # 配置参数保持不变
        train_config_updates = {
            # "reload_model_path": f"{prepath}/2/snapshots/6000.pth",
            "n_steps": 30000,
            "mode": "train",
            "eval_fold": eval_fold,
            "dataset": dataset,
            "save_snapshot_every": 2000,
        }
        # 运行脚本
        print("Running training...")
        run_train(train_config_updates)

        #### 验证模型测试 #####
        # 获取最新实验的所有模型文件
        snapshot_files = get_latest_snapshot_files(prepath)
        # files = [
        #     "./runs/GMRD_CMR_CV0_train/1/snapshots/3000.pth",
        #     "./runs/GMRD_CMR_CV0_train/1/snapshots/6000.pth",
        #     "./runs/GMRD_CMR_CV0_train/1/snapshots/9000.pth",
        # ]
        for snapshot in snapshot_files:
            test_config_updates = {
                "reload_model_path": snapshot,
                # "reload_model_path": f"{prepath}/4/snapshots/7000.pth",
                "mode": "val",
                "dataset": dataset,
                "eval_fold": eval_fold,
            }

            print("Running testing...")
            run_test(test_config_updates)

        # 提取日志数据
        base_directory = f"./runs/GMRD_{dataset}_CV{eval_fold}_val"
        output_csv_path = f"GMRD_{dataset}_CV{eval_fold}_val.csv"
        extract_logger_data(base_directory, output_csv_path)
