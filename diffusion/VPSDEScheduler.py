import torch
from .NoiseSchedulerBase import NoiseScheduler

class VPSDEScheduler(NoiseScheduler):
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0, **kwargs):
        super().__init__(**kwargs)
        self.beta_min = beta_min
        self.beta_max = beta_max

    def _get_beta_t(self, t: torch.Tensor) -> torch.Tensor:
        """Helper to calculate beta(t)."""
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def sde(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the drift and diffusion coefficients of the VPSDE."""
        beta_t = self._get_beta_t(t)
        drift = -0.5 * beta_t.view(-1, 1, 1, 1) * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x_0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the mean and standard deviation of the marginal probability p(x_t|x_0)."""
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        mean = torch.exp(log_mean_coeff).view(-1, 1, 1, 1) * x_0
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std