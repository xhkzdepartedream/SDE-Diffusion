import torch
import abc

class NoiseScheduler(abc.ABC):
    """Abstract base class for noise schedulers."""
    def __init__(self, prediction_type: str = 's'):
        self.prediction_type = prediction_type

    @abc.abstractmethod
    def sde(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the drift and diffusion coefficients of the forward SDE."""
        pass

    @abc.abstractmethod
    def marginal_prob(self, x_0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the mean and standard deviation of the marginal probability p(x_t|x_0)."""
        pass

    def forward_step(self, x_data: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) \
            -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x_data)

        mean, std = self.marginal_prob(x_data, t)
        xt = mean + std.view(-1, 1, 1, 1) * noise

        if self.prediction_type == 'eps':
            target = noise
        elif self.prediction_type == 's':
            target = -noise / (std.view(-1, 1, 1, 1) + 1e-8)
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        return xt, target

    def get_loss_weight(self, t: torch.Tensor):
        _, g_t = self.sde(torch.zeros(1, device=t.device), t)
        return g_t.view(-1) ** 2

    def _get_predicted_score(self, model_output: torch.Tensor, t: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
        """Helper function to compute the predicted score from the model output."""
        _, std_t = self.marginal_prob(torch.zeros_like(xt), t)
        std_t = std_t.view(-1, 1, 1, 1)

        if self.prediction_type == 'eps':
            predicted_score = -model_output / (std_t + 1e-8)
        elif self.prediction_type == 's':
            predicted_score = model_output
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")
        return predicted_score

    def reverse_sde_step(self, model_output: torch.Tensor, t: torch.Tensor, xt: torch.Tensor, dt: float) -> torch.Tensor:
        f_xt, g_t = self.sde(xt, t)
        g_t_2 = (g_t ** 2).view(-1, 1, 1, 1)

        predicted_score = self._get_predicted_score(model_output, t, xt)
        drift = f_xt - g_t_2 * predicted_score
        
        noise = torch.randn_like(xt)
        mask = (t > 1e-5).float().view(-1, 1, 1, 1)
        x_prev = xt + drift * dt + torch.sqrt(g_t_2 * abs(dt)) * noise * mask

        return x_prev

    def reverse_ode_step(self, model_output: torch.Tensor, t: torch.Tensor, xt: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Performs one reverse diffusion step using the corresponding ODE (probability flow).
        """
        f_xt, g_t = self.sde(xt, t)
        g_t_2 = (g_t ** 2).view(-1, 1, 1, 1)

        predicted_score = self._get_predicted_score(model_output, t, xt)
        drift = f_xt - 0.5 * g_t_2 * predicted_score

        x_prev = xt + drift * dt

        return x_prev
