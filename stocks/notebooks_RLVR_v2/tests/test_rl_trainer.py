import torch
import numpy as np

from rl_discovery.agent import AbsoluteZeroAgent
from rl_discovery.trainer import RolloutBuffer, PPOTrainer


def test_buffer_advantage_calculation():
    """
    [GUARD] Verifies GAE math.
    If value expected 0, but reward is 1, advantage must be positive.
    """
    num_steps = 3
    buffer = RolloutBuffer(num_steps=num_steps)

    # Fill mock data
    for i in range(num_steps):
        buffer.add(
            obs=np.zeros(33),
            action=torch.zeros(13),
            logprob=torch.tensor(-1.0),
            reward=1.0,  # Consistently positive reward
            value=torch.tensor(0.0),  # Critic pessimistically expected 0
            done=False,
        )

    next_value = torch.tensor(0.0)
    buffer.compute_advantages(next_value, next_done=False, gamma=0.99, gae_lambda=0.95)

    assert (
        buffer.advantages > 0
    ).all(), "Positive rewards vs 0-value should yield positive advantages."
    assert buffer.returns.shape == (num_steps,), "Returns tensor shape mismatch."


def test_ppo_trainer_update():
    """
    [GUARD] Verifies the optimizer steps, alters network weights, and
    returns correct diagnostic telemetry keys and types.
    """
    agent = AbsoluteZeroAgent()
    trainer = PPOTrainer(agent, lr=1e-3)

    # Store old parameters to check for changes
    old_actor_weight = next(agent.actor_mean.parameters()).clone().detach()
    old_critic_weight = next(agent.critic.parameters()).clone().detach()

    # Create fake populated buffer
    num_steps = 64
    buffer = RolloutBuffer(num_steps=num_steps)
    for i in range(num_steps):
        buffer.add(
            obs=np.random.randn(33),
            action=torch.randn(13),
            logprob=torch.tensor(-1.0),
            reward=np.random.randn(),
            value=torch.tensor(np.random.randn()),
            done=False,
        )

    buffer.compute_advantages(torch.tensor(0.0), False)

    # Run PPO Update and capture diagnostic payload
    diagnostics = trainer.update(buffer, update_epochs=1, mini_batch_size=32)

    # 1. Verify weights changed (Agent actually learned)
    assert not torch.equal(
        old_actor_weight, next(agent.actor_mean.parameters())
    ), "Actor weights did not update."
    assert not torch.equal(
        old_critic_weight, next(agent.critic.parameters())
    ), "Critic weights did not update."

    # 2. Verify diagnostics dictionary output structure and types
    assert isinstance(diagnostics, dict), "Trainer update did not return a dictionary."

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
        # Check that keys are present
        assert (
            key in diagnostics
        ), f"Diagnostic key '{key}' was missing from trainer output."

        # Check that outputs are numerical (accepts standard python floats or numpy floating points)
        val = diagnostics[key]
        assert isinstance(val, (float, np.floating)) or np.isnan(
            val
        ), f"Expected numeric float for diagnostic key '{key}', got {type(val)}: {val}"
