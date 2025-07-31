# SDE-Diffusion: Advanced Continuous-Time Diffusion Models

[English](README.md) | [中文](README_zh.md)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive and flexible PyTorch framework for building, training, and sampling from **continuous-time diffusion models** based on Stochastic Differential Equations (SDEs). This implementation provides state-of-the-art methods including VP-SDE, VE-SDE, and SubVP-SDE.

## 🚀 Key Features

### 🎯 **Continuous-Time Framework**
- Implements diffusion models in a continuous-time setting, offering superior performance over discrete-time DDPMs.
- Support for multiple SDE formulations with a unified interface.

### 🔄 **Dual Sampling Methods**
- **SDE Sampler**: Predictor-Corrector sampler based on reverse-time SDE for high-quality generation.
- **ODE Sampler**: Deterministic sampler using the Probability Flow ODE for fast and efficient sampling.

### 🏗️ **Modular Architecture**
- **Flexible Scheduler System**: Extensible `NoiseScheduler` base class supporting VP-SDE, VE-SDE, and SubVP-SDE.
- **Models**: Implemented U-Net and DiT as backbone networks. DiT supports conditional generation with class labels and Classifier-Free Guidance (CFG) for CIFAR-10 and MNIST.
- **Config-Driven Design**: YAML-based configuration for reproducible experiments.

### ⚡ **Production-Ready Training**
- Distributed training with DDP support.
- Mixed precision training with automatic gradient scaling.
- Exponential Moving Average (EMA) for stable training.
- Comprehensive checkpointing and resuming.

## 📁 Project Structure

```
SDE-Diffusion/
├── configs/                  # YAML configuration files (VP-SDE, VE-SDE, etc.)
├── data_processing/          # Data loading and preprocessing scripts
├── diffusion/                # Core diffusion algorithms (Schedulers, Pipeline)
├── models/                   # Neural network architectures (DiT, U-Net)
├── modules/                  # Reusable neural network components
├── scripts/                  # Execution scripts
│   ├── train/                # Training scripts for each dataset
│   ├── sample_celebahq.py    # Sampling scripts for generation
│   ├── sample_cifar10.py
├── trainer/                  # Training orchestration classes
├── utils.py                  # Utility functions
├── requirements.txt
├── README.md
└── README_zh.md
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 8GB+ GPU memory for training

### Quick Setup

1. **Clone the repository**
```bash
git clone https://github.com/your-username/SDE-Diffusion.git
cd SDE-Diffusion
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Verify installation**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## 🚀 Quick Start

### 1. Choose Your Configuration

Select from pre-configured setups in the `configs/` directory:

| Config File | Method | Best For | Training Time |
|-------------|--------|----------|---------------|
| `vpsde_celebahq.yaml` | VP-SDE | General purpose, stable training | ~12 hours |
| `vesde_celebahq.yaml` | VE-SDE | High-quality samples | ~18 hours |
| `subvpsde_config.yaml` | SubVP-SDE | Memory efficient | ~10 hours |

### 2. Start Training

```bash
# Train with VP-SDE on CelebA-HQ (recommended for beginners)
python scripts/train/train_celebahq.py --config configs/vpsde_celebahq.yaml
```

### 3. Generate Samples

Use the sampling scripts for generation:

```bash
# Generate unconditional samples on CelebA-HQ
python scripts/sample_celebahq.py --config configs/vpsde_celebahq.yaml \
                                --checkpoints.pipeline_path path/to/your/celebahq_model.pth

# Conditional sampling with CFG on CIFAR-10
python scripts/sample_cifar10.py --config configs/vpsde_cifar10.yaml \
                               --checkpoints.pipeline_path path/to/your/cifar10_model.pth \
                               --sampling.labels [0,1,2,3,4,5,6,7,8,9] \
                               --sampling.cfg_scale 4.0
```


## 💻 Advanced Usage

### Custom Training Loop

```python
import torch
from utils import instantiate_from_config
from diffusion.DiffusionPipeline import DiffusionPipeline

# Load configuration
config = torch.load('configs/vpsde_celebahq.yaml')
pipeline = instantiate_from_config(config['diffusion_pipeline'])


# Training step
def train_step(batch):
    x0 = batch.to(device)
    loss = pipeline.train_step(x0)
    return loss

# Your training loop here...
```

### Programmatic Sampling

```python
import torch
from utils import instantiate_from_config

# Load trained pipeline
pipeline = instantiate_from_config(config['diffusion_pipeline'])
checkpoint = torch.load('checkpoint.pth')
pipeline.model.load_state_dict(checkpoint['model_state_dict'])

# Generate samples
samples = pipeline.sample(
    shape=(8, 4, 32, 32),  # (batch, channels, height, width)
    n_steps=1000,          # Sampling steps
    sampler_type='sde',    # 'sde' or 'ode'
    corrector_steps=1      # For SDE sampling
)
```

## 🔧 Configuration Guide

### Model Architectures

```yaml
# DiT (Diffusion Transformer) - Recommended
model:
  target: models.DiT.DiT
  params:
    input_size: 32      # Input resolution
    input_ch: 4         # Input channels (RGB + alpha or latent)
    patch_size: 2       # Patch size for tokenization
    n_ch: 512          # Model dimension
    n_blocks: 12       # Number of transformer blocks
    num_heads: 8       # Attention heads
    pe: "rope"         # Position encoding: "rope" or "abs"

# U-Net - Classic choice
model:
  target: models.Unet.Unet
  params:
    # U-Net specific parameters
```

## 📊 Evaluation & Metrics

### Compute FID Score
```bash
python scripts/compute_fid.py --real_path /path/to/real/images \
                             --generated_path /path/to/generated/images
```

### Visualize Training Progress
```bash
python scripts/visualize_latent_space.py --checkpoint path/to/checkpoint.pth
```

## 📚 Citation

If you use this code in your research, please cite the original papers and this repository:

```bibtex
@misc{sde-diffusion-repo,
  title={SDE-Diffusion: Advanced Continuous-Time Diffusion Models},
  author={Your Name},
  year={2024},
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

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Built upon the theoretical foundations of [Score-Based Generative Models](https://arxiv.org/abs/2011.13456)
- DiT architecture based on [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)

---

## Thus, the project is completed, and all has been said.
