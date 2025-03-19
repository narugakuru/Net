import os
import random
import logging
import shutil

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import MultiStepLR
from models.fewshot_GMRD import FewShotSeg
from utils import *
from tqdm import tqdm
from config import ex
import torch


@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        # Set up source folder
        os.makedirs(f"{_run.observers[0].dir}/snapshots", exist_ok=True)
        for source_file, _ in _run.experiment_info["sources"]:
            os.makedirs(
                os.path.dirname(f"{_run.observers[0].dir}/source/{source_file}"),
                exist_ok=True,
            )
            _run.observers[0].save_file(source_file, f"source/{source_file}")
        shutil.rmtree(f"{_run.observers[0].basedir}/_sources")

        # Set up logger -> log to .txt
        file_handler = logging.FileHandler(
            os.path.join(f"{_run.observers[0].dir}", f"logger.log")
        )
        file_handler.setLevel("INFO")
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        _log.handlers.append(file_handler)
        _log.info(f'Run "{_config["exp_str"]}" with ID "{_run.observers[0].dir[-1]}"')

    # Deterministic setting for reproduciablity.
    if _config["seed"] is not None:
        random.seed(_config["seed"])
        torch.manual_seed(_config["seed"])
        torch.cuda.manual_seed_all(_config["seed"])
        cudnn.deterministic = True

    # Enable cuDNN benchmark mode to select the fastest convolution algorithm.
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config["gpu_id"])
    torch.set_num_threads(1)

    _log.info(f"Create model...")
    model = FewShotSeg()

    # Print the model layer
    print("model modules:")
    for dic, m in model.named_children():
        print(f"{dic} : {m}")

    model = model.cuda()
    model.train()

    # 加载本地的模型权重
    checkpoint_path = _config["reload_model_path"]
    if checkpoint_path is not None:
        _log.info(f"Loaded model from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    _log.info(f"Set optimizer...")
    optimizer = torch.optim.SGD(model.parameters(), **_config["optim"])
    lr_milestones = [
        (ii + 1) * _config["max_iters_per_load"]
        for ii in range(_config["n_steps"] // _config["max_iters_per_load"] - 1)
    ]
    scheduler = MultiStepLR(
        optimizer, milestones=lr_milestones, gamma=_config["lr_step_gamma"]
    )

    my_weight = torch.FloatTensor([0.1, 1.0]).cuda()
    criterion = nn.NLLLoss(ignore_index=255, weight=my_weight)

    _log.info(f"Start training...")

    n_sub_epochs = (
        _config["n_steps"] // _config["max_iters_per_load"]
    )  # number of times for reloading
    log_loss = {"total_loss": 0, "query_loss": 0, "align_loss": 0, "aux_loss": 0}

    i_iter = 0

    # 初始化最低损失值为无穷大
    min_total_loss = 1e10

    # Generate random data
    n_way = 1
    n_shot = 1
    n_query = 1
    H, W = 256, 256
    C = 3  # Assuming 3 channels for image

    support_images = [
        [torch.rand(C, H, W).float().cuda() for _ in range(n_shot)]
        for _ in range(n_way)
    ]
    support_fg_mask = [
        [torch.randint(0, 2, (1, H, W)).float().cuda() for _ in range(n_shot)]
        for _ in range(n_way)
    ]
    query_images = [torch.rand(C, H, W).float().cuda() for _ in range(n_query)]
    query_labels = torch.randint(0, 2, (n_query, H, W)).long().cuda()

    # Compute outputs and losses.
    query_pred, align_loss, aux_loss = model(
        support_images, support_fg_mask, query_images, train=True
    )
    aux_loss = 0.5 * aux_loss

    query_loss = criterion(
        torch.log(
            torch.clamp(
                query_pred,
                torch.finfo(torch.float32).eps,
                1 - torch.finfo(torch.float32).eps,
            )
        ),
        query_labels,
    )

    loss = query_loss + align_loss + aux_loss

    # Compute gradient and do SGD step.
    for param in model.parameters():
        param.grad = None

    loss.backward()
    optimizer.step()
    scheduler.step()

    # Log loss
    query_loss = query_loss.detach().data.cpu().numpy()
    aux_loss = aux_loss.detach().data.cpu().numpy()
    align_loss = align_loss.detach().data.cpu().numpy()

    _run.log_scalar("total_loss", loss.item())
    _run.log_scalar("query_loss", query_loss)
    _run.log_scalar("aux_loss", aux_loss)
    _run.log_scalar("align_loss", align_loss)

    log_loss["total_loss"] += loss.item()
    log_loss["query_loss"] += query_loss
    log_loss["align_loss"] += align_loss
    log_loss["aux_loss"] += aux_loss

    # Print loss and take snapshots.
    if (i_iter + 1) % _config["print_interval"] == 0:
        total_loss = log_loss["total_loss"] / _config["print_interval"]
        query_loss = log_loss["query_loss"] / _config["print_interval"]
        align_loss = log_loss["align_loss"] / _config["print_interval"]

        log_loss["total_loss"] = 0
        log_loss["query_loss"] = 0
        log_loss["align_loss"] = 0
        log_loss["aux_loss"] = 0

        _log.info(
            f"step {i_iter + 1}: total_loss: {total_loss}, query_loss: {query_loss}, aux_loss: {aux_loss}"
            f" align_loss: {align_loss}"
        )

        # 比较当前损失与最低损失
        if total_loss < min_total_loss:
            _log.info(f"Min total loss {min_total_loss}-{min_total_loss - total_loss}")
            min_total_loss = total_loss
        else:
            _log.info(f"Min total loss {min_total_loss}")

    if (i_iter + 1) % _config["save_snapshot_every"] == 0:
        _log.info("###### Taking snapshot ######")
        torch.save(
            model.state_dict(),
            os.path.join(f"{_run.observers[0].dir}/snapshots", f"{i_iter + 1}.pth"),
        )

    i_iter += 1

    _log.info("End of training.")
    return 1
