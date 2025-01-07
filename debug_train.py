#!/usr/bin/env python
"""
Debug script for generating random support/query images and masks
"""
import os
import random
import logging
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from models.fewshot_GMRD import FewShotSeg
from dataloaders.datasets import TrainDataset
from config import ex
import matplotlib.pyplot as plt
import numpy as np


@ex.automain
def main(_run, _config, _log):
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    _log = logging.getLogger(__name__)

    # Deterministic settings
    if _config["seed"] is not None:
        random.seed(_config["seed"])
        torch.manual_seed(_config["seed"])
        torch.cuda.manual_seed_all(_config["seed"])
        cudnn.deterministic = True

    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config["gpu_id"])
    torch.set_num_threads(1)

    # Create model
    _log.info("Creating model...")
    model = FewShotSeg()
    model = model.cuda()
    model.eval()  # Set to eval mode for debugging

    # Load data
    _log.info("Loading data...")
    data_config = {
        "data_dir": _config["path"][_config["dataset"]]["data_dir"],
        "dataset": _config["dataset"],
        "n_shot": _config["n_shot"],
        "n_way": _config["n_way"],
        "n_query": _config["n_query"],
        "n_sv": _config["n_sv"],
        "max_iter": 10,  # Only need a few iterations for debugging
        "eval_fold": _config["eval_fold"],
        "min_size": _config["min_size"],
        "test_label": _config["test_label"],
        "exclude_label": _config["exclude_label"],
        "use_gt": _config["use_gt"],
    }

    train_dataset = TrainDataset(data_config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=_config["batch_size"],
        shuffle=True,
        num_workers=_config["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    # Create output directory
    output_dir = "./debug_output"
    os.makedirs(output_dir, exist_ok=True)

    # Debug loop
    for i, sample in enumerate(train_loader):
        if i >= 5:  # Only process 5 samples for debugging
            break

        # Extract and visualize support images
        support_images = sample["support_images"]
        support_masks = sample["support_fg_labels"]

        # Visualize first support image and mask
        img = support_images[0][0].cpu().numpy()
        if img.ndim == 3:
            img = img[0]  # Take first channel if 3D
        mask = support_masks[0][0].cpu().numpy()

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap="gray")
        plt.title("Support Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap="gray")
        plt.title("Support Mask")
        plt.axis("off")

        plt.savefig(f"{output_dir}/support_{i}.png")
        plt.close()

        # Extract and visualize query images
        query_images = sample["query_images"]
        query_labels = sample["query_labels"]

        # Visualize first query image and label
        q_img = query_images[0].cpu().numpy()
        if q_img.ndim == 3:
            q_img = q_img[0]  # Take first channel if 3D
        q_label = query_labels[0].cpu().numpy()

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(q_img, cmap="gray")
        plt.title("Query Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(q_label, cmap="gray")
        plt.title("Query Label")
        plt.axis("off")

        plt.savefig(f"{output_dir}/query_{i}.png")
        plt.close()

        _log.info(f"Saved debug images for sample {i}")

    _log.info("Debugging completed. Images saved in ./debug_output")
