import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from typing import Optional


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal Initialization for variance stabilization."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class AbsoluteZeroAgent(nn.Module):
    """
    Continuous Action Actor-Critic Network for PPO.
    Equipped with LayerNorm, bounded logstd exploration floor, and linear mean projection.
    """

    def __init__(
        self,
        obs_dim: int = 46,
        action_dim: int = 16,
        hidden_size: int = 256,
        initial_logstd: float = -0.5,
    ):
        super().__init__()

        def make_hidden_block(in_features, out_features):
            return nn.Sequential(
                layer_init(nn.Linear(in_features, out_features)),
                nn.LayerNorm(out_features),
                nn.Tanh(),
            )

        # CRITIC Network (State-Value Estimation)
        self.critic = nn.Sequential(
            make_hidden_block(obs_dim, hidden_size),
            make_hidden_block(hidden_size, hidden_size),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )

        # ACTOR Network (Linear projection without squashing saturation)
        self.actor_mean = nn.Sequential(
            make_hidden_block(obs_dim, hidden_size),
            make_hidden_block(hidden_size, hidden_size),
            layer_init(nn.Linear(hidden_size, action_dim), std=0.01),
        )

        # EXPLORATION NOISE: State-independent parameter with safety clamping in forward
        self.actor_logstd = nn.Parameter(
            torch.full((1, action_dim), float(initial_logstd), dtype=torch.float32)
        )

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Estimates V(s) during rollout and evaluation."""
        return self.critic(x)

    def get_action_and_value(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ):
        """
        Computes Gaussian action distribution with bounded logstd.
        """
        raw_mean = self.actor_mean(x)
        # Smoothly bound policy mean to [-1.0, 1.0]
        action_mean = torch.tanh(raw_mean)

        # Enforce entropy floor: logstd in [-2.0, 0.2] -> std in [0.135, 1.22]
        clamped_logstd = torch.clamp(self.actor_logstd, min=-2.0, max=0.2)
        action_std = torch.exp(clamped_logstd.expand_as(action_mean))

        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )
