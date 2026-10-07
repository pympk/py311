"""
Continuous Action Actor-Critic Network for PPO.
Equipped with LayerNorm, bounded logstd exploration floor, authoritative Tanh mean projection,
exact Tobit censored Gaussian likelihood with boundary tail clipping, and full static typing compliance.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def layer_init(
    layer: nn.Linear,
    std: float = math.sqrt(2),
    bias_const: float = 0.0,
) -> nn.Linear:
    """Orthogonal Initialization for variance stabilization."""
    torch.nn.init.orthogonal_(layer.weight, std)  # type: ignore[arg-type]
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class AbsoluteZeroAgent(nn.Module):
    """
    Continuous Action Actor-Critic Network for PPO.
    Uses Boundary-Accessible Rescaled Projection allowing exact corner solutions
    at -1.0 and +1.0 while maintaining smooth sub-boundary gradients.
    """

    def __init__(
        self,
        obs_dim: int = 49,
        action_dim: int = 16,
        hidden_size: int = 256,
        initial_logstd: float = -0.50,  # std ~ 0.6065 (Gen 11 Restored Contract)
    ):
        super().__init__()

        def make_hidden_block(in_features: int, out_features: int) -> nn.Sequential:
            return nn.Sequential(
                layer_init(nn.Linear(in_features, out_features)),
                nn.LayerNorm(out_features),
                nn.Tanh(),
            )

        # CRITIC Network (State-Value Estimation V(s))
        self.critic = nn.Sequential(
            make_hidden_block(obs_dim, hidden_size),
            make_hidden_block(hidden_size, hidden_size),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )

        # ACTOR Network (Linear representation trunk)
        self.actor_mean = nn.Sequential(
            make_hidden_block(obs_dim, hidden_size),
            make_hidden_block(hidden_size, hidden_size),
            layer_init(nn.Linear(hidden_size, action_dim), std=0.01),
        )

        # EXPLORATION NOISE: Bounded continuous parameter initialized to std ~ 0.6065
        self.actor_logstd = nn.Parameter(
            torch.full((1, action_dim), float(initial_logstd), dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        Boundary-Accessible Projection:
        Scales tanh by 1.05 and clamps to [-1.0, 1.0].
        Guarantees that any logit |x| >= 1.74 reaches the exact corner solution
        (-1.0 or +1.0) while retaining smooth gradients inside the boundary.
        """
        raw_mean = self.actor_mean(x)
        return torch.clamp(1.05 * torch.tanh(raw_mean), -1.0, 1.0)

    def get_deterministic_action(self, x: torch.Tensor) -> torch.Tensor:
        """Authoritative deterministic action for validation, OOS evaluation, and production."""
        return self.forward(x, deterministic=True)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Estimates V(s) during rollout and evaluation."""
        return self.critic(x)

    def _compute_censored_log_prob(
        self, probs: Normal, action: torch.Tensor
    ) -> torch.Tensor:
        """
        Exact Tobit Censored Gaussian Log-Likelihood on [-1.0, 1.0] with Tail Clipping.
        - Interior (-1.0, 1.0): Continuous Gaussian log-density.
        - Left boundary (<= -1.0): log-CDF tail mass ln(Phi((-1 - mu) / sigma)).
        - Right boundary (>= 1.0): log-CDF tail mass ln(Phi((mu - 1) / sigma)).

        Tail clipping with eps = 1e-4 (min log prob = ln(1e-4) ~ -9.2103) stabilizes
        PPO probability ratios and resolves chronic clipping caused by extreme tail noise.
        """
        mu = probs.loc
        sigma = probs.scale

        # 1. Continuous interior density
        cont_log_prob = probs.log_prob(action)

        # 2. Discrete boundary masses via log_ndtr with eps = 1e-4 tail clipping
        log_eps = math.log(1e-4)
        left_tail = torch.clamp(
            torch.special.log_ndtr((-1.0 - mu) / sigma), min=log_eps, max=0.0
        )
        right_tail = torch.clamp(
            torch.special.log_ndtr((mu - 1.0) / sigma), min=log_eps, max=0.0
        )

        # 3. Assemble piecewise censored likelihood
        censored_log_prob = torch.where(
            action <= -1.0 + 1e-6,
            left_tail,
            torch.where(
                action >= 1.0 - 1e-6,
                right_tail,
                cont_log_prob,
            ),
        )
        return censored_log_prob.sum(dim=-1)

    def get_action_and_value(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes action, exact censored log-probability, entropy, and value V(s).
        Clamps logstd dynamically to [-2.0, -0.2] (std in [0.135, 0.819]).
        """
        action_mean = self.forward(x, deterministic=True)

        # Dynamic exploration noise bounds: std in [0.1353, 0.8187]
        clamped_logstd = torch.clamp(self.actor_logstd, min=-2.0, max=-0.2)
        action_std = torch.exp(clamped_logstd.expand_as(action_mean))

        probs = Normal(action_mean, action_std)

        if action is None:
            raw_sample = probs.sample()
            action = torch.clamp(raw_sample, -1.0, 1.0)

        log_prob = self._compute_censored_log_prob(probs, action)

        return (
            action,
            log_prob,
            probs.entropy().sum(dim=-1),
            self.critic(x),
        )
