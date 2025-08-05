# SDE-Diffusion: 先进的连续时间扩散模型

[English](README.md) | [中文](README_zh.md)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

一个全面而灵活的PyTorch框架，用于构建、训练和采样基于随机微分方程（SDE）的**连续时间扩散模型**。此实现提供了包括VP-SDE、VE-SDE和SubVP-SDE在内的最先进方法。

## 🚀 主要特性

### 🎯 **连续时间框架**
- 在连续时间设置中实现扩散模型，提供优于离散时间DDPM的性能。
- 支持多种SDE公式，具有统一的接口。

### 🔄 **双重采样方法**
- **SDE采样器**: 基于逆时SDE的预测-校正采样器，用于高质量生成。
- **ODE采样器**: 使用概率流ODE的确定性采样器，用于快速高效的采样。

### 🏗️ **模块化架构**
- **灵活的调度器系统**: 可扩展的`NoiseScheduler`基类，支持VP-SDE、VE-SDE和SubVP-SDE。
- **模型**: 将U-Net和DiT实现为骨干网络。DiT在CIFAR-10与MNIST上支持以类标签作为条件的条件生成以及无分类器指导采样(CFG)。
- **配置驱动设计**: 基于YAML的配置，用于可复现的实验。

### ⚡ **生产就绪的训练**
- 支持DDP的分布式训练。
- 具有自动梯度缩放的混合精度训练。
- 指数移动平均（EMA）用于稳定训练。
- 全面的检查点和恢复功能。

## 📁 项目结构

```
SDE-Diffusion/
├── configs/                  # YAML配置文件 (VP-SDE, VE-SDE等)
├── data_processing/          # 数据加载与预处理脚本
├── diffusion/                # 核心扩散算法 (调度器, 管线)
├── models/                   # 神经网络架构 (DiT, U-Net)
├── modules/                  # 可复用的神经网络组件
├── scripts/                  # 执行脚本
│   ├── train/                # 各数据集的训练脚本
│   ├── sample_celebahq.py    # 用于生成的采样脚本
│   ├── sample_cifar10.py
├── trainer/                  # 训练流程编排类
├── utils.py                  # 工具函数
├── requirements.txt
├── README.md
└── README_zh.md
```

## 🛠️ 安装

### 先决条件
- Python 3.8或更高版本
- 兼容CUDA的GPU（推荐）
- 8GB+ GPU内存用于训练

### 快速设置

1. **克隆仓库**
```bash
git clone https://github.com/your-username/SDE-Diffusion.git
cd SDE-Diffusion
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **验证安装**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## 🚀 快速入门

### 1. 选择您的配置

从`configs/`目录中选择预配置的设置：

| 配置文件 | 方法 | 最适用于 | 训练时间 |
|-------------|--------|----------|---------------|
| `vpsde_celebahq.yaml` | VP-SDE | 通用，稳定训练 | ~12小时 |
| `vesde_celebahq.yaml` | VE-SDE | 高质量样本 | ~18小时 |
| `subvpsde_config.yaml` | SubVP-SDE | 内存高效 | ~10小时 |

### 2. 开始训练

```bash
python scripts/train/train_celebahq.py --config configs/vpsde_celebahq.yaml
```
或者在这里下载已训练的模型:
[DOWNLOWD the trained model](https://drive.google.com/file/d/1d9ulaepOYhsWFEmpe0XaqxsvXjUa9pRf/view?usp=sharing)

### 3. 生成样本

使用采样脚本进行生成：

```bash
# 在CelebA-HQ上生成无条件样本
python scripts/sample_celebahq.py --config configs/vpsde_celebahq.yaml \
                                --checkpoints.pipeline_path path/to/your/celebahq_model.pth

# 在CIFAR-10上使用CFG进行条件采样
python scripts/sample_cifar10.py --config configs/vpsde_cifar10.yaml \
                               --checkpoints.pipeline_path path/to/your/cifar10_model.pth \
                               --sampling.labels [0,1,2,3,4,5,6,7,8,9] \
                               --sampling.cfg_scale 4.0
```


## 💻 高级用法

### 自定义训练循环

```python
import torch
from utils import instantiate_from_config
from diffusion.DiffusionPipeline import DiffusionPipeline

# 加载配置
config = torch.load('configs/vpsde_celebahq.yaml')
pipeline = instantiate_from_config(config['diffusion_pipeline'])


# 训练步骤
def train_step(batch):
    x0 = batch.to(device)
    loss = pipeline.train_step(x0)
    return loss

# 在这里编写您的训练循环...
```

### 程序化采样

```python
import torch
from utils import instantiate_from_config

# 加载训练好的流程
pipeline = instantiate_from_config(config['diffusion_pipeline'])
checkpoint = torch.load('checkpoint.pth')
pipeline.model.load_state_dict(checkpoint['model_state_dict'])

# 生成样本
samples = pipeline.sample(
    shape=(8, 4, 32, 32),  # (批量, 通道, 高度, 宽度)
    n_steps=1000,          # 采样步数
    sampler_type='sde',    # 'sde' 或 'ode'
    corrector_steps=1      # 用于SDE采样
)
```

## 🔧 配置指南

### 模型架构

```yaml
# DiT (扩散变换器) - 推荐
model:
  target: models.DiT.DiT
  params:
    input_size: 32      # 输入分辨率
    input_ch: 4         # 输入通道 (RGB + alpha 或潜在)
    patch_size: 2       # 用于标记化的补丁大小
    n_ch: 512          # 模型维度
    n_blocks: 12       # 变换器块的数量
    num_heads: 8       # 注意力头数
    pe: "rope"         # 位置编码: "rope" 或 "abs"

# U-Net - 经典选择
model:
  target: models.Unet.Unet
  params:
    # U-Net特定参数
```

## 📊 评估与指标

### 计算FID分数
```bash
python scripts/compute_fid.py --real_path /path/to/real/images \
                             --generated_path /path/to/generated/images
```

### 可视化潜在空间
```bash
python scripts/visualize_latent_space.py --checkpoint path/to/checkpoint.pth
```

## 📚 引用

如果您在研究中使用此代码，请引用原始论文和本代码库：

```bibtex
@misc{sde-diffusion-repo,
  title={SDE-Diffusion: Advanced Continuous-Time Diffusion Models},
  author={Your Name},
  year={2025},
  url={https://github.com/your-username/SDE-Diffusion}
}

@inproceedings{song2021scorebased,
  title={Score-Based Generative Modeling through Stochastic Differential Equations},
  author={Yang Song and Jascha Sohl-Dickstein and Diederik P Kingma and Abhishek Kumar and Stefano Ermon and Ben Poole},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=E_2243q5s7}
}

@inproceedings{peebles2023scalable,
  title={Scalable Diffusion Models with Transformers},
  author={William Peebles and Saining Xie},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={4195--4205},
  year={2023}
}
```

## 📄 许可证

本项目根据MIT许可证授权。

## 🙏 致谢

- 基于[基于分数的生成模型](https://arxiv.org/abs/2011.13456)的理论基础构建
- DiT架构基于[使用变换器的可扩展扩散模型](https://arxiv.org/abs/2212.09748)

---

## 至是，工程已毕，言尽于此。

```
