# SDE-Diffusion: Advanced Continuous-Time Diffusion Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive and flexible PyTorch framework for building, training, and sampling from **continuous-time diffusion models** based on Stochastic Differential Equations (SDEs). This implementation provides state-of-the-art methods including VP-SDE, VE-SDE, Flow Matching, and more.

## 🚀 Key Features

### 🎯 **Continuous-Time Framework**
- Implements diffusion models in continuous-time setting, offering superior performance over discrete-time DDPMs
- Support for multiple SDE formulations with unified interface

### 🔄 **Dual Sampling Methods**
- **SDE Sampler**: Predictor-Corrector sampler based on reverse-time SDE for high-quality generation
- **ODE Sampler**: Deterministic sampler using Probability Flow ODE for fast and efficient sampling

### 🏗️ **Modular Architecture**
- **Flexible Scheduler System**: Extensible `NoiseScheduler` base class supporting VP-SDE, VE-SDE, SubVP-SDE, and Flow Matching
- **Multiple Model Architectures**: DiT (Diffusion Transformer), U-Net, and VAE/VQGAN support
- **Config-Driven Design**: YAML-based configuration for reproducible experiments

### ⚡ **Production-Ready Training**
- Distributed training with DDP support
- Mixed precision training with automatic gradient scaling
- Exponential Moving Average (EMA) for stable training
- Comprehensive checkpointing and resuming

## 📁 Project Structure

```
SDE-Diffusion/
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 utils.py                    # Utility functions and config instantiation
├── 📁 configs/                    # 🔧 YAML configuration files
│   ├── vpsde_config.yaml         # Variance Preserving SDE config
│   ├── vesde_config.yaml         # Variance Exploding SDE config
│   ├── subvpsde_config.yaml      # Sub-Variance Preserving SDE config
│   ├── fm_config.yaml            # Flow Matching config
│   └── default_config_res.yaml   # Default configuration template
├── 📁 diffusion/                  # 🧠 Core diffusion algorithms
│   ├── NoiseSchedulerBase.py     # Abstract scheduler interface
│   ├── VPSDEScheduler.py         # VP-SDE implementation
│   ├── VESDEScheduler.py         # VE-SDE implementation
│   ├── SubVPSDEScheduler.py      # SubVP-SDE implementation
│   ├── FlowMatchingScheduler.py  # Flow Matching implementation
│   ├── DDPMLinearScheduler.py    # Linear DDPM scheduler
│   └── DiffusionPipeline.py      # 🚀 Main training/sampling pipeline
├── 📁 models/                     # 🏗️ Neural network architectures
│   ├── DiT.py                    # Diffusion Transformer (DiT)
│   ├── Unet.py                   # U-Net architecture
│   ├── VAVAE.py                  # Variational Autoencoder
│   └── VQGAN.py                  # Vector Quantized GAN
├── 📁 modules/                    # 🔧 Reusable components
│   ├── autoencoderkl.py          # Autoencoder modules
│   ├── vae_modules.py            # VAE building blocks
│   ├── perceptual_module.py      # Perceptual loss components
│   └── resnet.py                 # ResNet blocks
├── 📁 trainer/                    # 🎯 Training orchestration
│   ├── UnifiedTrainer.py         # Main training loop with DDP support
│   ├── AutoencoderKL_trainer.py  # Autoencoder-specific trainer
│   └── VQGAN_trainer.py          # VQGAN-specific trainer
├── 📁 scripts/                    # 🛠️ Execution scripts
│   ├── sample_unified.py         # Unified sampling script
│   ├── compute_fid.py            # FID evaluation
│   └── train/                    # Training scripts
└── 📁 data/                       # 📊 Data processing utilities
    ├── init_dataset.py           # Dataset initialization
    ├── download_celeba_hq.py     # CelebA-HQ downloader
    └── face_crop.py              # Face preprocessing
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
| `vpsde_config.yaml` | VP-SDE | General purpose, stable training | ~12 hours |
| `vesde_config.yaml` | VE-SDE | High-quality samples | ~18 hours |
| `fm_config.yaml` | Flow Matching | Fast sampling, fewer steps | ~8 hours |
| `subvpsde_config.yaml` | SubVP-SDE | Memory efficient | ~10 hours |

### 2. Start Training

```bash
# Train with VP-SDE (recommended for beginners)
python scripts/train/train_unified.py --config configs/vpsde_config.yaml

# Train with Flow Matching (fastest)
python scripts/train/train_unified.py --config configs/fm_config.yaml
```

### 3. Generate Samples

Use the unified sampling script for easy generation:

```bash
# Generate samples using trained model
python scripts/sample_unified.py --config configs/vpsde_config.yaml \
                                --checkpoint path/to/your/checkpoint.pth \
                                --num_samples 64 \
                                --sampler_type sde

# Fast ODE sampling (fewer steps)
python scripts/sample_unified.py --config configs/fm_config.yaml \
                                --checkpoint path/to/your/checkpoint.pth \
                                --num_samples 64 \
                                --sampler_type ode \
                                --n_steps 100
```

## 💻 Advanced Usage

### Custom Training Loop

```python
import torch
from utils import instantiate_from_config
from diffusion.DiffusionPipeline import DiffusionPipeline

# Load configuration
config = torch.load('configs/vpsde_config.yaml')
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

### Scheduler Comparison

| Scheduler | Pros | Cons | Use Case |
|-----------|------|------|----------|
| **VP-SDE** | Stable, well-studied | Slower sampling | General purpose |
| **VE-SDE** | High quality samples | Memory intensive | Research, high-end |
| **Flow Matching** | Fast sampling | Newer method | Production, speed |
| **SubVP-SDE** | Memory efficient | Less explored | Limited resources |

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

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black . && isort .
```

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@misc{sde-diffusion,
  title={SDE-Diffusion: Advanced Continuous-Time Diffusion Models},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/SDE-Diffusion}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built upon the theoretical foundations of [Score-Based Generative Models](https://arxiv.org/abs/2011.13456)
- Inspired by [Flow Matching](https://arxiv.org/abs/2210.02747) and [Rectified Flow](https://arxiv.org/abs/2209.03003)
- DiT architecture based on [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)

---

**🚀 Ready to generate amazing samples? Start with `configs/vpsde_config.yaml` for your first experiment!**
