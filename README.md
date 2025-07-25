# SDE-Diffusion: A Framework for Continuous-Time Diffusion Models

This project provides a comprehensive and flexible PyTorch framework for building, training, and sampling from continuous-time diffusion models based on Stochastic Differential Equations (SDEs).

## Core Features

- **Continuous-Time Framework**: Implements diffusion models in a continuous-time setting, offering a more powerful alternative to discrete-time DDPMs.
- **SDE/ODE Dual Sampling**: Supports both stochastic (SDE) and deterministic (ODE) sampling methods:
  - **SDE Sampler**: A Predictor-Corrector sampler based on the reverse-time SDE, allowing for high-quality sample generation.
  - **ODE Sampler**: A deterministic sampler based on the corresponding Probability Flow ODE, enabling faster and more efficient sampling.
- **Flexible Scheduler Architecture**: A base `NoiseScheduler` class allows for easy extension to different types of SDEs (e.g., VP-SDE, VE-SDE, subVP-SDE).
- **Config-Driven**: Utilizes YAML configuration files for managing all aspects of the model, scheduler, and training, promoting reproducibility and easy experimentation.
- **Modular Design**: Clean separation of concerns between models (`models/`), schedulers (`diffusion/`), and trainers (`trainer/`).

## Project Structure

```
E:/DL/Diffusion-Model/
├───README.md
├───requirements.txt
├───utils.py
├───configs/         # YAML configuration files for models and training.
├───data/            # Scripts for data loading and preprocessing.
├───diffusion/       # Core diffusion process logic.
│   ├───NoiseSchedulerBase.py  # Abstract base class for all schedulers.
│   ├───VPSDEScheduler.py      # Variance Preserving (VP) SDE implementation.
│   ├───DiffusionPipeline.py   # Main pipeline for training and sampling.
│   └───...
├───models/          # Score-based network architectures (e.g., Unet, DiT).
├───modules/         # Reusable neural network modules.
├───scripts/         # High-level scripts for training, sampling, and evaluation.
└───trainer/         # Trainer classes that manage the training loop.
```

## Dependencies

All required Python packages are listed in the `requirements.txt` file. Install them using pip:

```bash
pip install -r requirements.txt
```

## How to Use

### 1. Configuration

All settings for the model, scheduler, and data are defined in YAML files within the `configs/` directory. You can create new configs or modify existing ones (e.g., `configs/vpsde_config.yaml`) to experiment with different hyperparameters.

### 2. Training

Training is handled by the scripts in `scripts/train/`. To start a training run, you typically point a training script to a configuration file.

**Example command:**

```bash
python scripts/train/train_unified.py --config configs/vpsde_config.yaml
```

### 3. Sampling

After training, you can use the `DiffusionPipeline` to generate samples. The pipeline provides a simple interface for both SDE and ODE sampling.

Below is a Python code snippet demonstrating how to load a pre-trained model and generate an image:

```python
import torch
from utils import instantiate_from_config
from diffusion.DiffusionPipeline import DiffusionPipeline

# 1. Load the configuration from your training run
# Make sure to replace 'path/to/your/config.yaml' with the actual path
config = torch.load('path/to/your/config.yaml')

# 2. Instantiate the pipeline from the config
# This will automatically set up the model and scheduler
pipeline = instantiate_from_config(config.pipeline)

# 3. Load your trained model checkpoint
# Make sure to replace 'path/to/your/checkpoint.pth' with the actual path
checkpoint = torch.load('path/to/your/checkpoint.pth')
pipeline.model.load_state_dict(checkpoint['model_state_dict'])
pipeline.model.eval()

# --- Generate Samples ---

# Define the shape of the desired output
# (batch_size, channels, height, width)
image_shape = (1, 3, 64, 64)
n_steps = 1000 # Number of steps for the reverse process

# Generate a sample using the SDE sampler (Predictor-Corrector)
print("Generating sample with SDE sampler...")
sde_sample = pipeline.sample(
    shape=image_shape, 
    n_steps=n_steps, 
    sampler_type='sde',
    corrector_steps=1 # Optional: number of corrector steps
)

# Generate a sample using the ODE sampler (Deterministic)
# ODE sampling is often faster and may require fewer steps.
print("Generating sample with ODE sampler...")
ode_sample = pipeline.sample(
    shape=image_shape, 
    n_steps=n_steps, 
    sampler_type='ode'
)

# You can now save or display the generated `sde_sample` and `ode_sample` tensors.
```

### 至是，工程已毕，言尽于此。