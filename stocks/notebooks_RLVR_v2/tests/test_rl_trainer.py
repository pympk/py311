import numpy as np
import torch
from rl_discovery.agent import AbsoluteZeroAgent
from rl_discovery.trainer import PPOTrainer, RolloutBuffer


def test_buffer_advantage_calculation():
    """
    [GUARD] Verifies GAE math for the vectorized buffer with explicit
    separation of terminations and truncations.
    """
    num_steps = 3
    num_envs = 1
    obs_dim = 35
    action_dim = 14

    buffer = RolloutBuffer(
        num_steps=num_steps, num_envs=num_envs, obs_dim=obs_dim, action_dim=action_dim
    )

    for _ in range(num_steps):
        buffer.add(
            obs=np.zeros((num_envs, obs_dim)),
            action=torch.zeros((num_envs, action_dim)),
            logprob=torch.tensor([-1.0]),
            reward=np.array([1.0]),
            value=torch.tensor([[0.0]]),
            terminations=np.zeros(num_envs),
            truncations=np.zeros(num_envs),
        )

    next_value = torch.tensor([[0.0]])
    next_termination = torch.tensor([0.0], dtype=torch.float32)

    buffer.compute_advantages(
        next_value, next_termination=next_termination, gamma=0.99, gae_lambda=0.95
    )

    # Advantages should be positive because reward (1.0) > value (0.0)
    assert (
        buffer.advantages > 0
    ).all(), "Positive rewards vs 0-value should yield positive advantages."
    assert buffer.returns.shape == (
        num_steps,
        num_envs,
    ), "Returns tensor shape mismatch."


def test_ppo_trainer_update():
    """
    [GUARD] Verifies optimizer updates actor and critic parameters and
    returns correct diagnostic telemetry keys under the termination/truncation protocol.
    """
    obs_dim = 35
    action_dim = 14
    num_envs = 1

    agent = AbsoluteZeroAgent(obs_dim=obs_dim, action_dim=action_dim)
    trainer = PPOTrainer(agent, lr=1e-3)

    old_actor_weight = next(agent.actor_mean.parameters()).clone().detach()
    old_critic_weight = next(agent.critic.parameters()).clone().detach()

    num_steps = 64
    buffer = RolloutBuffer(
        num_steps=num_steps, num_envs=num_envs, obs_dim=obs_dim, action_dim=action_dim
    )

    for _ in range(num_steps):
        buffer.add(
            obs=np.random.randn(num_envs, obs_dim),
            action=torch.randn(num_envs, action_dim),
            logprob=torch.tensor([-1.0]),
            reward=np.random.randn(num_envs),
            value=torch.tensor(np.random.randn(num_envs, 1)),
            terminations=np.zeros(num_envs),
            truncations=np.zeros(num_envs),
        )

    buffer.compute_advantages(
        torch.tensor([[0.0]]), next_termination=torch.tensor([0.0], dtype=torch.float32)
    )

    diagnostics = trainer.update(buffer, update_epochs=1, mini_batch_size=32)

    # 1. Verify weights updated
    assert not torch.equal(
        old_actor_weight, next(agent.actor_mean.parameters())
    ), "Actor weights did not update."
    assert not torch.equal(
        old_critic_weight, next(agent.critic.parameters())
    ), "Critic weights did not update."

    # 2. Verify diagnostics payload
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
