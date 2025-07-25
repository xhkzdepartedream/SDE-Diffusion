import torch
from .NoiseSchedulerBase import NoiseScheduler

class FlowMatchingScheduler(NoiseScheduler):
    """
    A scheduler for Flow Matching models, particularly Rectified Flow.

    This scheduler implements the training and inference logic for a model that learns
    the velocity field of a simple, straight-line path between noise and data.
    """

    def __init__(self, **kwargs):
        # Flow matching doesn't have prediction types like eps/v, so we can ignore it.
        super().__init__(**kwargs)

    def sde(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Flow matching is based on ODEs, not SDEs. This method is not applicable."""
        # For compatibility, we can return zeros, but it shouldn't be used.
        return torch.zeros_like(x), torch.zeros_like(t)

    def marginal_prob(self, x_0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The marginal probability for flow matching is not as straightforward as in SDEs.
           The forward_step method directly computes the required xt.
           This method is not directly used in the training loop for flow matching.
        """
        raise NotImplementedError("marginal_prob is not used for FlowMatchingScheduler.")

    def forward_step(self, x_data: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) \
            -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the interpolated sample `xt` and the target velocity `v` for training.

        Args:
            x_data (torch.Tensor): The real data sample (x1).
            t (torch.Tensor): A tensor of time steps in [0, 1].
            noise (torch.Tensor, optional): The noise sample (x0). If None, it's generated.

        Returns:
            xt (torch.Tensor): The interpolated sample at time t.
            target (torch.Tensor): The target velocity field (x1 - x0).
        """
        if noise is None:
            noise = torch.randn_like(x_data)

        # Reshape t for broadcasting: [B] -> [B, 1, 1, 1]
        t_reshaped = t.view(-1, 1, 1, 1)

        # Linear interpolation between noise (x0) and data (x1)
        xt = (1 - t_reshaped) * noise + t_reshaped * x_data

        # The target for the model is the constant velocity field
        target = x_data - noise

        return xt, target

    def get_loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        """
        For standard flow matching (and Rectified Flow), the loss is unweighted.
        Returns a tensor of ones.
        """
        return torch.ones_like(t)

    def reverse_sde_step(self, model_output: torch.Tensor, t: torch.Tensor, xt: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Performs one step of the reverse ODE solve using Euler's method.

        Args:
            model_output (torch.Tensor): The predicted velocity `v(xt, t)` from the model.
            t (torch.Tensor): The current time step (not used in the simplest Euler step, but kept for interface consistency).
            xt (torch.Tensor): The current sample in the reverse process.
            dt (float): The time step size. For flow matching, this is positive during inference (t=0 to t=1).

        Returns:
            torch.Tensor: The sample at the next time step `xt + dt`.
        """
        # The model directly predicts the velocity v_t
        predicted_velocity = model_output

        # Euler method for the ODE: dx/dt = v(x,t)
        # x_{t+dt} = x_t + v(x_t, t) * dt
        # Note: In our sampling loop, dt is negative, so we subtract to move forward in time (0 -> 1)
        x_prev = xt - predicted_velocity * dt

        return x_prev
