#!/bin/bash

# 创建 Conda 环境
echo "Creating Conda environment..."
conda create --name net python=3.10 -y

# 激活 Conda 环境
echo "Activating Conda environment..."
source activate net

# 安装 CUDA Toolkit
echo "Installing CUDA Toolkit..."
conda install cudatoolkit==11.8 -y

# 安装依赖
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# 安装 PyTorch 和 TorchVision
echo "Installing PyTorch and TorchVision..."
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

echo "Setup complete!"
