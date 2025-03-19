from main_utils import *


if __name__ == "__main__":

    eval_fold_list = [0, 1, 2, 3, 4]
    eval_fold_list = [0]

    for eval_fold in eval_fold_list:
        print("Eval fold:", eval_fold)
        dataset = "ABDOMEN_MR"  # CMR CHAOST2 SABS
        #### 训练 ####
        prepath = f"./runs/CDFS_train_{dataset}_cv{eval_fold}"

        # 配置参数保持不变
        train_config_updates = {
            # "reload_model_path": f"{prepath}/2/snapshots/20000.pth",
            "n_steps": 20000,
            "mode": "train",
            "eval_fold": eval_fold,
            "dataset": dataset,
            "save_snapshot_every": 1000,
            # "test_label": [1, 2, 3, 6],
            "test_label": [6],
            # "test_label": [],
            "exclude_label": None,
            "use_gt": True,
        }
        # 运行脚本
        print("Running training...")
        run_train(train_config_updates)

        #### 验证模型测试 #####
        prepath = f"./runs/CDFS_train_{dataset}_cv{eval_fold}"

        dataset_test = "ABDOMEN_CT"  # CMR CHAOST2 SABS ABDOMEN_MR

        # 获取最新实验的所有模型文件
        # snapshot_files = get_latest_snapshot_files(prepath)
        # files = [
        #     "./runs/CMR_CV0_train/1/snapshots/3000.pth",
        #     "./runs/CMR_CV0_train/1/snapshots/6000.pth",
        #     "./runs/CMR_CV0_train/1/snapshots/9000.pth",
        # ]
        snapshot_files = [1]
        # snapshot_files = None

        for snapshot in snapshot_files:
            test_config_updates = {
                # "reload_model_path": snapshot,
                # "reload_model_path": f"./runs/CDFS_train_ABDOMEN_MR_cv0/4/snapshots/28000.pth",
                "reload_model_path": f"./runs/CDFS_train_ABDOMEN_MR_cv1/2/snapshots/20000.pth",  # "reload_model_path": f"{prepath}/4/snapshots/28000.pth",
                "mode": "val",
                "dataset": dataset_test,
                "eval_fold": eval_fold,
                # "eval_fold": 2,
                "test_label": [1, 2, 3, 6],
                # "test_label": [6, 2, 3, 1],
                # "test_label": [1, 2, 3, 4],
            }

            print("Running testing...")

            # run_test(test_config_updates)

            # # 提取日志数据
            base_directory = f"./runs/CDFS_val_{dataset_test}_cv{eval_fold}"
            output_csv_path = f"CDFS_val_{dataset_test}_cv{eval_fold}.csv"
            extract_logger_data(base_directory, output_csv_path)
