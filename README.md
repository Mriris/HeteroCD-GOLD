# HeteroCD-GOLD

A heterogeneous remote sensing image change detection framework based on **G**uided **O**nline **L**earning for **D**istillation.

[中文文档](README_CN.md) | [English](README.md)

## Overview

This project implements GOLD, a three-branch fully-supervised online distillation model for detecting changes between optical and SAR remote sensing images. The model uses online knowledge distillation from homogeneous optical-optical pairs (teacher branch) to guide heterogeneous optical-SAR change detection (student branch), fundamentally addressing cross-modal feature space differences.

## Key Features

- **Three-branch Architecture**: Homogeneous teacher branch (optical-optical) and heterogeneous student branch (optical-SAR) with shared temporal-1 optical encoder
- **Online Knowledge Distillation**: Real-time transfer of high-level change features and quality label information during each iteration
- **Difference Map Attention Transfer**: Spatial and channel attention mechanisms with saliency maps for enhanced change perception
- **Dynamic Weight Allocation**: Uncertainty-based adaptive loss weighting for change detection, distillation, and attention losses
- **LabelmeCD-AI Annotation Tool**: Synchronized dual-image display with AI-driven pre-annotations

## Dataset

First benchmark dataset combining optical-optical and optical-SAR time-series pairs:
- **Gaofen-2** high-resolution optical images
- **Gaofen-3** SAR images  
- **Sentinel-2** multispectral images

## Requirements

```bash
# intsall PyTorch
pip3 install torch torchvision torchaudio

# install other dependencies
pip install -r requirements.txt
```

## Dataset Structure

```
data/
├── train/
│   ├── A/          # Optical images (time 1)
│   ├── B/          # SAR images (time 2)
│   ├── C/          # Optical images (time 2)
│   └── Label/      # Change labels
└── val/
    ├── A/
    ├── B/
    ├── C/
    └── Label/
```

## Quick Start

### Training

```bash
python train.py --dataroot ./data --name experiment_name --gpu_ids 0
```

### Testing

```bash
python test.py --dataroot ./data --model_path ./checkpoints/model.pth --phase test
```

### Lightweight Model

```bash
python train.py --dataroot ./data --name lightweight_exp --use_lightweight --gpu_ids 0
```

## Repository Structure

```
HeteroCD-GOLD/
├── models/                 # Model implementations
│   ├── GOLD.py            # Main GOLD model
│   ├── TripleEUNet.py     # Three-branch network
│   └── loss.py            # Loss functions
├── datasets/              # Data loading utilities
├── options/               # Training/testing options
├── train.py               # Training script
├── test.py                # Testing script
└── doc/                   # Documentation and paper
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{heterocd_gold,
  title={GOLD: Guided Online Learning for Distillation for Heterogeneous Remote Sensing Image Change Detection},
  author={Tingxuan Yan},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  year={2025}
}
```

## License

This project is released under the MIT License. 