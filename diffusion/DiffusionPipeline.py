import torch
import math
from tqdm import tqdm
from typing import Optional
from utils import instantiate_from_config

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DiffusionPipeline:
    """
    A generalized diffusion model that relies on a scheduler object for its
    forward and reverse process logic, adapted for continuous time SDEs.
    """
    def __init__(self, model: dict, scheduler: dict):

        super().__init__()
        self.model = instantiate_from_config(model).to(device)
        self.scheduler = instantiate_from_config(scheduler)
        self.device = device

    def train_step(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None, eps: float = 1e-5) -> torch.Tensor:

        batch_size = x0.shape[0]
        t = torch.rand(batch_size, device = self.device) * (1.0 - eps) + eps

        if noise is None:
            noise = torch.randn_like(x0)

        xt, target = self.scheduler.forward_step(x0, t, noise)
        prediction = self.model(xt, t)

        loss_weight = self.scheduler.get_loss_weight(t)
        loss = torch.mean((loss_weight.view(-1, 1, 1, 1) * (prediction - target) ** 2))

        return loss

    def corrector(self, x: torch.Tensor, t: torch.Tensor, target_snr: float = 0.16,
                  corrector_steps: int = 1) -> torch.Tensor:

        for _ in range(corrector_steps):
            # Predict score
            score = self.model(x, t)

            noise = torch.randn_like(x)

            grad_norm = torch.norm(score.reshape(score.shape[0], -1), dim = -1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim = -1).mean()

            eps = (target_snr * noise_norm / grad_norm) ** 2 * 2
            x_mean = x + eps * score
            x = x_mean + torch.sqrt(2 * eps) * noise

        return x

    @torch.no_grad()
    def sample(self, shape, n_steps: int, eps: float = 1e-5, sampler_type: str = 'sde', **kwargs) -> torch.Tensor:
        shape = tuple(shape)
        target_snr = kwargs.get('target_snr', 0.16)
        corrector_steps = kwargs.get('corrector_steps', 1)
        xt = torch.randn(shape, device = self.device)
        print("Noise stats:", torch.mean(xt, dim = (1, 2, 3)), torch.std(xt, dim = (1, 2, 3)))
        print("Sample noise diff:", torch.sum((xt[0] - xt[1]) ** 2))

        # time_steps = torch.linspace(1.0, eps, n_steps, device = self.device)

        # t_lin = torch.linspace(0, 1, n_steps, device = self.device)
        # time_steps = torch.cos(t_lin * math.pi / 2) ** 2
        # time_steps = time_steps * (1.0 - eps) + eps

        t_lin = torch.linspace(0, 1, n_steps, device = self.device)
        time_steps = (1.0 - t_lin) ** 2 * (1.0 - eps) + eps

        for i in tqdm(range(n_steps)):
            t = time_steps[i]
            vec_t = torch.full((shape[0],), t.item(), device = self.device)
            dt = time_steps[i] - time_steps[i - 1] if i > 0 else time_steps[0] - 1.0
            model_output = self.model(xt, vec_t)

            if sampler_type == 'sde':
                xt = self.scheduler.reverse_sde_step(model_output, vec_t, xt, dt)
                xt = self.corrector(xt, vec_t, target_snr = target_snr, corrector_steps = corrector_steps)
            elif sampler_type == 'ode':
                xt = self.scheduler.reverse_ode_step(model_output, vec_t, xt, dt)
            else:
                raise ValueError(f"Unknown sampler_type: {sampler_type}")

        return xt