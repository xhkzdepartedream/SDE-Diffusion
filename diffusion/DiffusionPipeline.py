import torch
import math
from tqdm import tqdm
from typing import Optional
from utils import instantiate_from_config

try:
    from diffusers import DPMSolverMultistepScheduler

    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Warning: diffusers not available. DPM-Solver++ sampling will not work.")

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

    def train_step(self, x0: torch.Tensor, labels: Optional[torch.Tensor] = None,
                   noise: Optional[torch.Tensor] = None, eps: float = 1e-5) -> torch.Tensor:

        batch_size = x0.shape[0]
        t = torch.rand(batch_size, device = self.device) * (1.0 - eps) + eps

        if noise is None:
            noise = torch.randn_like(x0)

        xt, target = self.scheduler.forward_step(x0, t, noise)

        # Pass labels to model if provided (for conditional DiT)
        if labels is not None:
            prediction = self.model(xt, t, labels)
        else:
            prediction = self.model(xt, t)

        # NaN detection before loss computation
        if torch.any(torch.isnan(xt)) or torch.any(torch.isnan(target)) or torch.any(torch.isnan(prediction)):
            print("NaN detected before loss computation.")
            print("t:", t)
            print("g_t:", self.scheduler.sde(torch.zeros_like(t), t)[1])
            print("xt mean/std:", xt.mean().item(), xt.std().item())
            print("target mean/std:", target.mean().item(), target.std().item())
            print("prediction mean/std:", prediction.mean().item(), prediction.std().item())
            raise RuntimeError("NaN during forward pass")

        loss_weight = self.scheduler.get_loss_weight(t)
        loss = torch.mean((loss_weight.view(-1, 1, 1, 1) * (prediction - target) ** 2))

        return loss

    def corrector(self, x: torch.Tensor, t: torch.Tensor, labels: Optional[torch.Tensor] = None,
                  target_snr: float = 0.16, corrector_steps: int = 1) -> torch.Tensor:

        for _ in range(corrector_steps):
            # Get model output and convert to score based on prediction type
            if labels is not None:
                model_output = self.model(x, t, labels)
            else:
                model_output = self.model(x, t)

            # Convert model output to score using scheduler's method
            score = self.scheduler._get_predicted_score(model_output, t, x)

            noise = torch.randn_like(x)

            grad_norm = torch.norm(score.reshape(score.shape[0], -1), dim = -1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim = -1).mean()

            # Add numerical stability
            if grad_norm < 1e-8:
                continue

            eps = (target_snr * noise_norm / grad_norm) ** 2 * 2
            x_mean = x + eps * score
            x = x_mean + torch.sqrt(2 * eps) * noise

        return x

    @torch.no_grad()
    def sample(self, shape, n_steps: int, eps: float = 1e-5, sampler_type: str = 'sde',
               labels: Optional[torch.Tensor] = None, cfg_scale: Optional[float] = None, **kwargs) -> torch.Tensor:
        shape = tuple(shape)
        labels = torch.tensor(labels, dtype = torch.int, device = self.device)
        target_snr = kwargs.get('target_snr', 0.16)
        corrector_steps = kwargs.get('corrector_steps', 1)
        xt = torch.randn(shape, device = self.device)

        time_steps = torch.linspace(1.0, eps, n_steps, device = self.device)

        # t_lin = torch.linspace(0, 1, n_steps, device = self.device)
        # time_steps = torch.cos(t_lin * math.pi / 2) ** 2
        # time_steps = time_steps * (1.0 - eps) + eps

        # t_lin = torch.linspace(0, 1, n_steps, device = self.device)
        # time_steps = (1.0 - t_lin) ** 2 * (1.0 - eps) + eps

        for i in tqdm(range(n_steps)):
            t = time_steps[i]
            vec_t = torch.full((shape[0],), t.item(), device = self.device)
            dt = time_steps[i] - time_steps[i - 1] if i > 0 else time_steps[0] - 1.0

            # Get model output with optional labels and CFG
            if cfg_scale is not None and cfg_scale > 1.0 and labels is not None:
                model_output = self.model(xt, vec_t, labels, cfg_scale = cfg_scale)
            elif labels is not None:
                model_output = self.model(xt, vec_t, labels)
            else:
                model_output = self.model(xt, vec_t)

            if sampler_type == 'sde':
                xt = self.scheduler.reverse_sde_step(model_output, vec_t, xt, dt)
                xt = self.corrector(xt, vec_t, labels, target_snr = target_snr, corrector_steps = corrector_steps)
            elif sampler_type == 'ode':
                xt = self.scheduler.reverse_ode_step(model_output, vec_t, xt, dt)
            elif sampler_type == 'dpmsolver++':
                # Use diffusers DPM-Solver++ for high-quality sampling
                if not DIFFUSERS_AVAILABLE:
                    raise ImportError(
                        "diffusers library is required for DPM-Solver++ sampling. Install with: pip install diffusers")

                return self._sample_with_dpmsolver(shape, n_steps, eps, labels = labels, cfg_scale = cfg_scale,
                                                   **kwargs)
            else:
                raise ValueError(f"Unknown sampler_type: {sampler_type}")

        return xt

    def _sample_with_dpmsolver(self, shape, n_steps: int, eps: float = 1e-5,
                               labels: Optional[torch.Tensor] = None, cfg_scale: Optional[float] = None, **kwargs):
        """
        Sample using diffusers DPM-Solver++ implementation.
        """
        # Create DPM-Solver scheduler
        dpm_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps = 1000,
            beta_start = 0.00085,
            beta_end = 0.012,
            beta_schedule = "scaled_linear",
            solver_order = kwargs.get('solver_order', 2),
            prediction_type = self.scheduler.prediction_type,
            thresholding = kwargs.get('thresholding', False),
            dynamic_thresholding_ratio = kwargs.get('dynamic_thresholding_ratio', 0.995),
            sample_max_value = kwargs.get('sample_max_value', 1.0),
            algorithm_type = kwargs.get('algorithm_type', 'dpmsolver++'),
            solver_type = kwargs.get('solver_type', 'midpoint'),
            lower_order_final = kwargs.get('lower_order_final', True),
            use_karras_sigmas = kwargs.get('use_karras_sigmas', False),
        )

        # Set timesteps
        dpm_scheduler.set_timesteps(n_steps, device = self.device)
        timesteps = dpm_scheduler.timesteps

        # Initialize sample
        sample = torch.randn(shape, device = self.device)

        # Sampling loop
        for i, t in enumerate(tqdm(timesteps, desc = "DPM-Solver++ Sampling")):
            # Expand timestep for batch
            timestep = t.expand(sample.shape[0])

            # Model prediction with optional labels and CFG
            with torch.no_grad():
                if cfg_scale is not None and cfg_scale > 1.0 and labels is not None:
                    model_output = self.model(sample, timestep, labels, cfg_scale = cfg_scale)
                elif labels is not None:
                    model_output = self.model(sample, timestep, labels)
                else:
                    model_output = self.model(sample, timestep)

            # DPM-Solver step
            sample = dpm_scheduler.step(model_output, t, sample).prev_sample

        return sample
