# FSMIS via GMRD

![image](https://github.com/zmcheng9/GMRD/blob/main/overview.png)

### Abstract
Automatic medical image segmentation has witnessed significant development with the success of large models on massive datasets. However, acquiring and annotating vast medical image datasets often proves to be impractical due to the time consumption, specialized expertise requirements, and compliance with patient privacy standards, etc. As a result, Few-shot Medical Image Segmentation (FSMIS) has become an increasingly compelling research direction. Conventional FSMIS methods usually learn prototypes from support images and apply nearest-neighbor searching to segment the query images. However, only a single prototype cannot well represent the distribution of each class, thus leading to restricted performance. To address this problem, we propose to Generate Multiple Representative Descriptors (GMRD), which can comprehensively represent the commonality within the corresponding class distribution. In addition, we design a Multiple Affinity Maps based Prediction (MAMP) module to fuse the multiple affinity maps generated by the aforementioned descriptors. Furthermore, to address intra-class variation and enhance the representativeness of descriptors, we introduce two novel losses. Notably, our model is structured as a dual-path design to achieve a balance between foreground and background differences in medical images. Extensive experiments on four publicly available medical image datasets demonstrate that our method outperforms the state-of-the-art methods, and the detailed analysis also verifies the effectiveness of our designed module.

### Dependencies
Please install following essential dependencies:
```
dcm2nii
json5==0.8.5
jupyter==1.0.0
nibabel==2.5.1
numpy==1.22.0
opencv-python==4.5.5.62
Pillow>=8.1.1
sacred==0.8.2
scikit-image==0.18.3
SimpleITK==1.2.3
torch==1.10.2
torchvision=0.11.2
tqdm==4.62.3
```

### Data sets and pre-processing
Download:
1) **CHAOS-MRI**: [Combined Healthy Abdominal Organ Segmentation data set](https://chaos.grand-challenge.org/)
2) **Synapse-CT**: [Multi-Atlas Abdomen Labeling Challenge](https://www.synapse.org/#!Synapse:syn3193805/wiki/218292)
3) **CMR**: [Multi-sequence Cardiac MRI Segmentation data set](https://zmiclab.github.io/projects/mscmrseg19/) (bSSFP fold)

Pre-processing is performed according to [Ouyang et al.](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation/tree/2f2a22b74890cb9ad5e56ac234ea02b9f1c7a535) and we follow the procedure on their github repository.

### Training
1. Compile `./data/supervoxels/felzenszwalb_3d_cy.pyx` with cython (`python ./data/supervoxels/setup.py build_ext --inplace`) and run `./data/supervoxels/generate_supervoxels.py` 
2. Download pre-trained ResNet-101 weights [vanilla version](https://download.pytorch.org/models/resnet101-63fe2227.pth) or [deeplabv3 version](https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth) and put your checkpoints folder, then replace the absolute path in the code `./models/encoder.py`.  
3. Run `./script/train.sh` 

### Inference
Run `./script/evaluate.sh` 

### Citation
```
@ARTICLE{cheng2024few,
  author={Cheng, Ziming and Wang, Shidong and Xin, Tong and Zhou, Tao and Zhang, Haofeng and Shao, Ling},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Few-Shot Medical Image Segmentation via Generating Multiple Representative Descriptors}, 
  year={2024},
  volume={},
  number={},
  pages={1-1}
  doi={10.1109/TMI.2024.3358295}}
```

```
FewShotSeg (nn.Module)
├── encoder: Res101Encoder (特征提取器，可加载预训练权重)
├── mlp1: MLP (将 256 维特征映射到 100 维)
├── mlp2: MLP (将 256 维特征映射到 600 维)
├── decoder1: Decoder (解码前景相似度图)
├── decoder2: Decoder (解码背景相似度图)
└── forward (前向传播)
    ├── 特征提取 (encoder)
    ├── 原型提取
    │   ├── getFeatures (计算掩码平均池化特征)
    │   ├── getPrototype (计算原型)
    │   ├── get_fg_pts (提取前景原型)
    │   └── get_bg_pts (提取背景原型)
    ├── 预测计算
    │   ├── getPred (计算预测)
    │   ├── get_fg_sim (计算前景相似度)
    │   └── get_bg_sim (计算背景相似度)
    └── 损失计算
        ├── alignLoss (计算对齐损失)
        └── align_aux_Loss (计算对齐和辅助损失)
```