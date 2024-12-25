# ?Net

### Abstract
Null

### Dependencies
Please install following essential dependencies:
```
dcm2nii or dcm2niix
other detail in requirements.txt
```

### Data sets and pre-processing
Download:
1) **CHAOS-MRI**: [Combined Healthy Abdominal Organ Segmentation data set](https://chaos.grand-challenge.org/)
2) **Synapse-CT**: [Multi-Atlas Abdomen Labeling Challenge](https://www.synapse.org/#!Synapse:syn3193805/wiki/218292)
3) **CMR**: [Multi-sequence Cardiac MRI Segmentation data set](https://zmiclab.github.io/projects/mscmrseg19/) (bSSFP fold)

Pre-processing is performed according to [Ouyang et al.](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation/tree/2f2a22b74890cb9ad5e56ac234ea02b9f1c7a535) and we follow the procedure on their github repository.
I numbered the script execution order,all you need in ./data

Compile `./data/supervoxels/felzenszwalb_3d_cy.pyx` with cython (`python ./data/supervoxels/setup.py build_ext --inplace`) and run `./data/supervoxels/generate_supervoxels.py` 

### Pre-trained models
Download pre-trained ResNet-101 weights [vanilla version](https://download.pytorch.org/models/resnet101-63fe2227.pth) or [deeplabv3 version](https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth) and put your checkpoints folder, then replace the absolute path in the code `./models/encoder.py`.  


### Training and Inference
Run `main.py` 

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

### Citation
```
@ARTICLE{Luo2024few,
  author={Luo},
  journal={?}, 
  title={?}, 
  year={2024},
  volume={},
  number={},
  pages={1-1}
  doi={10.1109/TMI.2024.3358295}}
```

