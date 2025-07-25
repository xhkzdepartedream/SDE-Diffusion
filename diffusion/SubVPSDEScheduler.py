import torch
from .VPSDEScheduler import VPSDEScheduler

class SubVPSDEScheduler(VPSDEScheduler):
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0, **kwargs):
        super().__init__(beta_min=beta_min, beta_max=beta_max, **kwargs)

    def _get_beta_t(self, t: torch.Tensor) -> torch.Tensor:
        """Helper to calculate beta(t) for sub-VP-SDE."""
        # The beta schedule is the same as VP-SDE
        return super()._get_beta_t(t)

    def sde(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the drift and diffusion coefficients of the sub-VP-SDE."""
        beta_t = self._get_beta_t(t)
        drift = -0.5 * beta_t.view(-1, 1, 1, 1) * x
        # The diffusion is scaled by sqrt(1 - exp(-2 * integral_beta_t))
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        diffusion_scaling = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff)).view(-1, 1, 1, 1)
        diffusion = torch.sqrt(beta_t).view(-1, 1, 1, 1) * diffusion_scaling
        return drift, diffusion

    def marginal_prob(self, x_0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the mean and standard deviation of the marginal probability p(x_t|x_0) for sub-VP-SDE."""
        # The marginal probability is the same as VP-SDE
        return super().marginal_prob(x_0, t)
