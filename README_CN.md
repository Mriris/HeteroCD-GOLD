# HeteroCD-GOLD

基于**引导在线蒸馏学习**的异源遥感图像变化检测框架

[中文文档](README_CN.md) | [English](README.md)

## 项目简介

本项目实现了GOLD模型，这是一个三分支全监督在线蒸馏模型，用于检测光学和SAR遥感图像之间的变化。该模型通过同源光学-光学对（教师分支）的在线知识蒸馏来指导异源光学-SAR变化检测（学生分支），从根本上解决了跨模态特征空间差异问题。

## 主要特点

- **三分支架构**：同源教师分支（光学-光学）和异源学生分支（光学-SAR），共享时间点1光学编码器
- **在线知识蒸馏**：在每次迭代中实时传递高层次变化特征和优质标签信息
- **差异图注意力迁移**：结合显著性图的空间和通道注意力机制，增强变化感知能力
- **动态权重分配**：基于不确定性估计的自适应损失权重调整，平衡变化检测、蒸馏和注意力损失
- **LabelmeCD-AI标注工具**：双图同步显示与AI预标注功能

## 数据集

首个结合光学-光学和光学-SAR时序对的基准数据集：
- **高分二号**高分辨率光学图像
- **高分三号**化合成孔径雷达图像
- **Sentinel-2**多光谱图像

## 环境要求

```bash
# 安装PyTorch
pip3 install torch torchvision torchaudio

# 安装其他依赖
pip install -r requirements.txt
```

## 数据集结构

```
data/
├── train/
│   ├── A/          # 光学图像（时间点1）
│   ├── B/          # SAR图像（时间点2）
│   ├── C/          # 光学图像（时间点2）
│   └── Label/      # 变化标签
└── val/
    ├── A/
    ├── B/
    ├── C/
    └── Label/
```

## 快速开始

### 训练

```bash
python train.py --dataroot ./data --name experiment_name --gpu_ids 0
```

### 测试

```bash
python test.py --dataroot ./data --model_path ./checkpoints/model.pth --phase test
```

### 轻量化模型

```bash
python train.py --dataroot ./data --name lightweight_exp --use_lightweight --gpu_ids 0
```

## 项目结构

```
HeteroCD-GOLD/
├── models/                 # 模型实现
│   ├── GOLD.py            # 主要GOLD模型
│   ├── TripleEUNet.py     # 三分支网络
│   └── loss.py            # 损失函数
├── datasets/              # 数据加载工具
├── options/               # 训练/测试配置
├── train.py               # 训练脚本
├── test.py                # 测试脚本
└── doc/                   # 文档和论文
```

## 引用

如果您在研究中使用了本代码，请引用：

```bibtex
@article{heterocd_gold,
  title={GOLD: Guided Online Learning for Distillation for Heterogeneous Remote Sensing Image Change Detection},
  author={Tingxuan Yan},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  year={2025}
}
```

## 开源协议

本项目基于MIT协议开源。