import torch
import os
from omegaconf import OmegaConf

from utils import instantiate_from_config, load_model_from_checkpoint, show_tensor_image


def main():
    cli_conf = OmegaConf.from_cli()
    default_config_path = cli_conf.get("config", "../configs/vpsde_celebahq.yaml")
    conf = OmegaConf.merge(OmegaConf.load(default_config_path), cli_conf)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    diffusion_pipeline = instantiate_from_config(conf.diffusion_pipeline)
    diffusion_pipeline = load_model_from_checkpoint(conf.checkpoints.pipeline_path, 'dp', device, diffusion_pipeline)

    print(f"[INFO] DiffusionPipeline loaded from {conf.checkpoints.pipeline_path}")

    vae = load_model_from_checkpoint(conf.checkpoints.vae_path, 'autoencoderkl', device)
    print(f"[INFO] VAE loaded from {conf.checkpoints.vae_path}")

    print(f"[INFO] Generating {conf.sampling.shape[0]} samples...")
    with torch.no_grad():
        latents = diffusion_pipeline.sample(**conf.sampling)
        print("[INFO] Decoding latents...")
        images = vae.decode(latents).sample

    # 6. Save images
    for i, image in enumerate(images):
        print(image.shape)
        show_tensor_image(image)


if __name__ == "__main__":
    main()