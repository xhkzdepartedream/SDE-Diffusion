import torch
from torch_ema import ExponentialMovingAverage
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from diffusion.DiffusionPipeline import DiffusionPipeline
from utils import init_distributed

device, local_rank = init_distributed()


class Trainer:
    """
    A unified, modular trainer for diffusion models, adapted to the project's style.
    It handles the training loop, distributed data_processing parallel setup, gradient scaling,
    and checkpointing in a generic way.
    """

    def __init__(
            self,
            pipeline: DiffusionPipeline,
            dataset: Dataset,
            title: str,
            lr: float,
            n_epoch: int,
            batch_size: int,
    ):
        self.pipeline = pipeline
        self.dataset = dataset
        self.optimizer = torch.optim.AdamW(self.pipeline.model.parameters(), lr = lr)
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.title = title
        self.losses = []
        self.model_conditional = pipeline.model.has_cond if hasattr(pipeline.model, 'has_cond') else False
        self.start_epoch = 1

        self.datasampler = DistributedSampler(dataset)
        self.dataloader = DataLoader(self.dataset, batch_size = self.batch_size, shuffle = False,
                                     num_workers = 32, pin_memory = True, sampler = self.datasampler)
        self.scaler = torch.amp.GradScaler()
        self.device = device
        self.local_rank = local_rank

        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0 = 20,
            T_mult = 1,
            eta_min = 1e-5
        )
        self.pipeline.model = self.pipeline.model.to(self.device)
        self.pipeline.model = DDP(self.pipeline.model, device_ids = [self.local_rank])

        self.ema = ExponentialMovingAverage(self.pipeline.model.parameters(), decay = 0.999)

    def _save_checkpoint(self, epoch: int):
        """Saves a checkpoint of the model and optimizer states."""
        if self.local_rank != 0:
            return

        model_state = self.pipeline.model.state_dict()

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
        }

        torch.save(checkpoint, f"{self.title}_{epoch}.pth")
        print(f"[INFO] DIFFUSION MODEL Checkpoint saved at epoch {epoch}.")

    def _load_checkpoint(self, checkpoint_path: str, rank: int):
        self.checkpoint_path = checkpoint_path
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}  # 把保存时的 0号GPU 映射到当前rank

        checkpoint = torch.load(checkpoint_path, map_location = map_location)

        self.pipeline.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1

        print(f"[INFO][RANK {rank}] Loaded checkpoint from epoch {checkpoint['epoch']}.")

    def train(self, epochs: int, save_every: int = 5):
        if self.model_conditional:
            self._conditional_train(epochs, save_every)
        else:
            self._unconditional_train(epochs, save_every)

    def _conditional_train(self, epochs: int, save_every: int = 20):
        for epoch in range(self.start_epoch, epochs + 1):
            self.datasampler.set_epoch(epoch)
            total_loss = 0
            pbar = tqdm(self.dataloader, desc = f"Epoch {epoch}", disable = (self.local_rank != 0))

            for batch, label in pbar:
                x0 = batch.to(self.device)

                self.optimizer.zero_grad()

                with torch.amp.autocast('cuda'):
                    loss = self.pipeline.train_step(x0, label)

                # Additional NaN detection after loss computation
                if torch.isnan(loss):
                    print(f"NaN loss detected at epoch {epoch}")
                    print(f"Current learning rate: {self.optimizer.param_groups[0]['lr']}")
                    print(f"Scaler scale: {self.scaler.get_scale()}")
                    # Check model parameters for NaN
                    for name, param in self.pipeline.model.named_parameters():
                        if torch.any(torch.isnan(param)):
                            print(f"NaN in parameter: {name}")
                        if param.grad is not None and torch.any(torch.isnan(param.grad)):
                            print(f"NaN in gradient: {name}")
                    raise RuntimeError(f"NaN loss at epoch {epoch}")


                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.pipeline.model.parameters(), max_norm = 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.ema.update()
                self.scheduler.step(epoch + len(self.losses) / len(self.dataloader))
                total_loss += loss.item()

                if self.local_rank == 0:
                    pbar.set_postfix(loss = loss.item())

            self.losses.append(total_loss)
            if self.local_rank == 0:
                print(f"Epoch {epoch} | Loss: {total_loss:.4f}")

            if epoch % save_every == 0:
                self._save_checkpoint(epoch)

        if self.local_rank == 0:
            print("[INFO] Training finished.")

    def _unconditional_train(self, epochs: int, save_every: int = 5):
        for epoch in range(self.start_epoch, epochs + 1):
            self.datasampler.set_epoch(epoch)
            total_loss = 0
            pbar = tqdm(self.dataloader, desc = f"Epoch {epoch}", disable = (self.local_rank != 0))

            for batch in pbar:
                x0 = batch.to(self.device)

                self.optimizer.zero_grad()

                with torch.amp.autocast('cuda'):
                    loss = self.pipeline.train_step(x0)

                # Additional NaN detection after loss computation
                if torch.isnan(loss):
                    print(f"NaN loss detected at epoch {epoch}")
                    print(f"Current learning rate: {self.optimizer.param_groups[0]['lr']}")
                    print(f"Scaler scale: {self.scaler.get_scale()}")
                    # Check model parameters for NaN
                    for name, param in self.pipeline.model.named_parameters():
                        if torch.any(torch.isnan(param)):
                            print(f"NaN in parameter: {name}")
                        if param.grad is not None and torch.any(torch.isnan(param.grad)):
                            print(f"NaN in gradient: {name}")
                    raise RuntimeError(f"NaN loss at epoch {epoch}")

                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.pipeline.model.parameters(), max_norm = 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.ema.update()
                self.scheduler.step(epoch + len(self.losses) / len(self.dataloader))
                total_loss += loss.item()

                if self.local_rank == 0:
                    pbar.set_postfix(loss = loss.item())

            self.losses.append(total_loss)
            if self.local_rank == 0:
                print(f"Epoch {epoch} | Loss: {total_loss:.4f}")

            if epoch % save_every == 0:
                self._save_checkpoint(epoch)

        if self.local_rank == 0:
            print("[INFO] Training finished.")
