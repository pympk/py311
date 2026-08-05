import torch
import numpy as np
from rl_discovery.agent import AbsoluteZeroAgent
from rl_discovery.trainer import RolloutBuffer, PPOTrainer


def test_buffer_advantage_calculation():
    """
    [GUARD] Verifies GAE math for the vectorized buffer.
    If value expected 0, but reward is 1, advantage must be positive.
    """
    num_steps = 3
    num_envs = 1
    obs_dim = 35
    action_dim = 14

    # Initialize with num_envs=1 for simple math verification
    buffer = RolloutBuffer(
        num_steps=num_steps, num_envs=num_envs, obs_dim=obs_dim, action_dim=action_dim
    )

    # Fill mock data (Now requiring the batch dimension)
    for i in range(num_steps):
        buffer.add(
            obs=np.zeros((num_envs, obs_dim)),
            action=torch.zeros((num_envs, action_dim)),
            logprob=torch.tensor([-1.0]),
            reward=np.array([1.0]),  # Positive reward
            value=torch.tensor([[0.0]]),  # Critic expected 0
            done=np.array([False]),
        )

    next_value = torch.tensor([[0.0]])
    next_done = torch.tensor([False], dtype=torch.float32)

    buffer.compute_advantages(
        next_value, next_done=next_done, gamma=0.99, gae_lambda=0.95
    )

    # Advantages should be positive because reward (1) > value (0)
    assert (
        buffer.advantages > 0
    ).all(), "Positive rewards vs 0-value should yield positive advantages."
    assert buffer.returns.shape == (
        num_steps,
        num_envs,
    ), "Returns tensor shape mismatch."


def test_ppo_trainer_update():
    """
    [GUARD] Verifies the optimizer steps, alters network weights, and
    returns correct diagnostic telemetry keys.
    """
    obs_dim = 35
    action_dim = 14
    num_envs = 1

    # Match agent to the new dimensions
    agent = AbsoluteZeroAgent(obs_dim=obs_dim, action_dim=action_dim)
    trainer = PPOTrainer(agent, lr=1e-3)

    # Store old parameters to check for changes
    old_actor_weight = next(agent.actor_mean.parameters()).clone().detach()
    old_critic_weight = next(agent.critic.parameters()).clone().detach()

    # Create fake populated buffer
    num_steps = 64
    buffer = RolloutBuffer(
        num_steps=num_steps, num_envs=num_envs, obs_dim=obs_dim, action_dim=action_dim
    )

    for i in range(num_steps):
        buffer.add(
            obs=np.random.randn(num_envs, obs_dim),
            action=torch.randn(num_envs, action_dim),
            logprob=torch.tensor([-1.0]),
            reward=np.random.randn(num_envs),
            value=torch.tensor(np.random.randn(num_envs, 1)),
            done=np.random.choice([True, False], size=num_envs),
        )

    buffer.compute_advantages(
        torch.tensor([[0.0]]), torch.tensor([False], dtype=torch.float32)
    )

    # Run PPO Update
    diagnostics = trainer.update(buffer, update_epochs=1, mini_batch_size=32)

    # 1. Verify weights changed
    assert not torch.equal(
        old_actor_weight, next(agent.actor_mean.parameters())
    ), "Actor weights did not update."
    assert not torch.equal(
        old_critic_weight, next(agent.critic.parameters())
    ), "Critic weights did not update."

    # 2. Verify diagnostics
    expected_keys = {
        "policy_loss",
        "value_loss",
        "entropy",
        "total_loss",
        "approx_kl",
        "clip_fraction",
        "explained_variance",
    }

    for key in expected_keys:
        assert key in diagnostics, f"Diagnostic key '{key}' was missing."
        assert isinstance(diagnostics[key], (float, np.floating)) or np.isnan(
            diagnostics[key]
        )
