import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
from omegaconf import OmegaConf
from diffusion.DiffusionPipeline import DiffusionPipeline
from trainer.UnifiedTrainer import UnifiedTrainer
from utils import init_distributed, instantiate_from_config
from data.init_dataset import LatentDataset


def main():
    # 1. Load configuration
    cli_conf = OmegaConf.from_cli()
    default_config_path = cli_conf.get("config", "./configs/vesde_config.yaml")
    yaml_conf = OmegaConf.load(default_config_path)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    device, local_rank = init_distributed()

    diffusion_pipeline = instantiate_from_config(conf.diffusion_pipeline)

    dataset = LatentDataset(conf.data.path)

    trainer = UnifiedTrainer(
        pipeline = diffusion_pipeline,
        dataset = dataset,
        title = conf.training.title,
        lr = conf.training.lr,
        n_epoch = conf.training.epochs,
        batch_size = conf.training.batch_size,
    )

    if conf.checkpoint.load:
        trainer._load_checkpoint(conf.checkpoint.path, local_rank)

    trainer.train(epochs = conf.training.epochs, save_every = conf.training.save_every)


if __name__ == "__main__":
    main()
