import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Orthogonal Initialization.
    Maintains variance of activations across layers to prevent vanishing/exploding gradients.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class AbsoluteZeroAgent(nn.Module):
    """
    Continuous Action Actor-Critic Network for PPO.
    Upgraded with Layer Normalization and wider capacity [256, 256].
    """

    def __init__(self, obs_dim: int = 33, action_dim: int = 13, hidden_size: int = 256):
        super().__init__()

        # ---------------------------------------------------------------------
        # HELPER: Construct a normalized hidden block
        # Pattern: Linear -> LayerNorm -> Activation
        # ---------------------------------------------------------------------
        def make_hidden_block(in_features, out_features):
            return nn.Sequential(
                layer_init(nn.Linear(in_features, out_features)),
                nn.LayerNorm(out_features),  # Stabilizes financial variance
                nn.Tanh(),  # Tanh is standard/stable for PPO
            )

        # ---------------------------------------------------------------------
        # CRITIC: Independent Network
        # Estimates the expected veritable reward (Value) of the current state.
        # ---------------------------------------------------------------------
        self.critic = nn.Sequential(
            make_hidden_block(obs_dim, hidden_size),
            make_hidden_block(hidden_size, hidden_size),
            # Final Value layer uses std=1.0
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )

        # ---------------------------------------------------------------------
        # ACTOR: Independent Network
        # Decides what action to take (Policy Mean).
        # ---------------------------------------------------------------------
        self.actor_mean = nn.Sequential(
            make_hidden_block(obs_dim, hidden_size),
            make_hidden_block(hidden_size, hidden_size),
            # Final Action layer uses std=0.01 to ensure initial actions are ~0
            layer_init(nn.Linear(hidden_size, action_dim), std=0.01),
        )

        # ---------------------------------------------------------------------
        # EXPLORATION NOISE (Entropy)
        # State-Independent log standard deviation.
        # Initialized to 0, so exp(0) = 1.0 standard deviation.
        # ---------------------------------------------------------------------
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x):
        """Used during rollout to calculate advantages."""
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """
        Calculates the policy distribution, samples an action, and calculates its probability.
        """
        # 1. Get Action Mean from the Actor network
        action_mean = self.actor_mean(x)  # (batch, action_dim)

        # 2. Get Action StdDev from the independent parameter
        action_logstd = self.actor_logstd.expand_as(action_mean)  # (batch, action_dim)
        action_std = torch.exp(action_logstd)  # (batch, action_dim)

        # 3. Create a Normal distribution for our continuous dimensions
        probs = Normal(action_mean, action_std)  # Independent Gaussians

        # 4. If we aren't evaluating an old action, sample a new one
        if action is None:
            action = probs.sample()

        # PPO requires the log probability of the action and the entropy (for the exploration bonus)
        # .sum(1) collapses the dimension probabilities into a single scalar per batch item
        return (
            action,  # the sampled (or provided) action
            probs.log_prob(action).sum(
                1
            ),  # log-probability of that action, summed over dims
            probs.entropy().sum(1),  # entropy of the distribution, summed over dims
            self.critic(x),  # state-value estimate
        )
