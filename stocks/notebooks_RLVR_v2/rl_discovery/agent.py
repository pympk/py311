import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    [DEEP MODULE] Orthogonal Initialization.
    Maintains variance of activations across layers to prevent vanishing gradients.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class AbsoluteZeroAgent(nn.Module):
    """
    Continuous Action Actor-Critic Network for PPO.
    """

    def __init__(self, obs_dim: int = 33, action_dim: int = 13):
        super().__init__()

        # CRITIC: Estimates the expected veritable reward (Value) of the current state
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),  # std=1.0 for the final value layer
        )

        # ACTOR: Decides what action to take (Policy)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(
                nn.Linear(128, action_dim), std=0.01
            ),  # std=0.01 so initial actions are near 0
        )

        # State-Independent exploration noise.
        # Initialized to 0, so exp(0) = 1.0 standard deviation.
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x):
        """Used during rollout to calculate advantages."""
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """
        Calculates the policy distribution, samples an action, and calculates its probability.
        """
        action_mean = self.actor_mean(x)  # # (batch, 13)
        action_logstd = self.actor_logstd.expand_as(action_mean)  # # (batch, 13)
        action_std = torch.exp(action_logstd)  # # (batch, 13)

        # Create a Normal distribution for our 13 continuous dimensions
        probs = Normal(action_mean, action_std)  # # 13 independent Gaussians

        # If we aren't evaluating an old action, sample a new one
        if action is None:
            action = probs.sample()

        # PPO requires the log probability of the action and the entropy (for the exploration bonus)
        # .sum(1) collapses the 13 dimension probabilities into a single scalar per batch item
        return (
            action,  # the sampled (or provided) action
            probs.log_prob(action).sum(
                1
            ),  # log-probability of that action, summed over 13 dims
            probs.entropy().sum(1),  # entropy of the distribution, summed over 13 dims
            self.critic(x),  # state-value estimate
        )
