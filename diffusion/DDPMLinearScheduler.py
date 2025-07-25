import torch
import torch.nn.functional as F
from .NoiseSchedulerBase import NoiseScheduler
from utils import gather

class DDPMLinearScheduler(NoiseScheduler):
    def __init__(self, n_steps: int, device: str):
        """
        Initializes the DDPM scheduler with a linear beta schedule.

        Args:
            n_steps (int): The total number of diffusion steps.
            device (str): The device to place tensors on ('cuda' or 'cpu').
        """
        super().__init__()
        self.n_steps = n_steps
        self.device = device

        # --- Linear Schedule ---
        self.beta = torch.linspace(1e-4, 0.02, n_steps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        # For reverse process q(x_{t-1} | x_t, x_0)
        alpha_bar_prev = F.pad(self.alpha_bar[:-1], (1, 0), value=1.0)
        self.posterior_variance = self.beta * (1. - alpha_bar_prev) / (1. - self.alpha_bar)
        # For p_sample, when not learning sigma, variance is fixed
        self.sigma2 = self.posterior_variance

    def forward_get_mean_var(self, x_data: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the mean and variance of q(x_t | x_0) for the forward process.
        This corresponds to the original q_xt_x0.
        """
        mean = gather(self.alpha_bar, t).sqrt() * x_data
        var = 1 - gather(self.alpha_bar, t)
        return mean, var

    def forward_step(self, x_data: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs one step of the forward diffusion process (q_sample) and returns the noisy sample
        and the target for the model to predict.

        For standard DDPM, the target is the noise itself.
        """
        if noise is None:
            noise = torch.randn_like(x_data)
        
        mean, var = self.forward_get_mean_var(x_data, t)
        xt = mean + var.sqrt() * noise
        target = noise # The model should predict the noise
        
        return xt, target

    def reverse_sde_step(self, model_output: torch.Tensor, t: torch.Tensor, xt: torch.Tensor, dt: float = None) -> torch.Tensor:
        """
        Performs one step of the reverse diffusion process (p_sample).
        This implementation uses the fixed variance from the DDPM paper.
        The 'dt' parameter is ignored here as we use discrete steps.
        """
        predicted_noise = model_output
        
        alpha_bar = gather(self.alpha_bar, t)
        alpha = gather(self.alpha, t)

        mean_coeff = (1.0 - alpha) / (1.0 - alpha_bar).sqrt()
        mean = (1.0 / alpha.sqrt()) * (xt - mean_coeff * predicted_noise)
        
        # Use the fixed posterior variance
        var = gather(self.sigma2, t)
        
        noise = torch.randn_like(xt)
        # Do not sample noise at the last step (t=0)
        mask = (t > 0).float().view(-1, 1, 1, 1)
        
        return mean + (var.sqrt() * noise * mask)