import torch
from .NoiseSchedulerBase import NoiseScheduler

class SubVPSDEScheduler(NoiseScheduler):
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0, **kwargs):
        super().__init__(**kwargs)
        self.beta_min = beta_min
        self.beta_max = beta_max

    def _get_beta_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def _integrated_beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2

    def sde(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        beta_t = self._get_beta_t(t)
        int_beta = self._integrated_beta(t)
        drift = -0.5 * beta_t.view(-1, 1, 1, 1) * x
        diffusion = torch.sqrt(beta_t * (1 - torch.exp(-2 * int_beta)))
        return drift, diffusion

    def marginal_prob(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        int_beta = self._integrated_beta(t)
        log_mean_coeff = -int_beta
        mean = torch.exp(log_mean_coeff).view(-1, 1, 1, 1) * x0
        std = torch.sqrt(1. - torch.exp(-2. * int_beta)).view(-1, 1, 1, 1)
        return mean, std

