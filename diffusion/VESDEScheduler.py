import torch
from .NoiseSchedulerBase import NoiseScheduler


class VESDEScheduler(NoiseScheduler):
    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 378.0, **kwargs):
        super().__init__(**kwargs)
        self.sigma_min = torch.tensor(sigma_min)
        self.sigma_max = torch.tensor(sigma_max)
        self.frac = self.sigma_max / self.sigma_min

    def _get_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def sde(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the drift and diffusion coefficients of the VPSDE."""
        drift = torch.zeros_like(x)
        diffusion = torch.sqrt(2 * torch.log(self.frac)) * (self.frac) ** t * self.sigma_min
        return drift, diffusion

    def marginal_prob(self, x_0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = x_0
        std = torch.sqrt(self._get_sigma_t(t) ** 2 - self.sigma_min ** 2)
        return mean, std
