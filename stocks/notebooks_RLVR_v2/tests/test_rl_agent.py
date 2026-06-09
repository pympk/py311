import torch

from rl_discovery.agent import AbsoluteZeroAgent


def test_agent_tensor_shapes():
    """
    [GUARD] Verifies the Actor-Critic network outputs correctly shaped tensors.
    A mismatch here causes catastrophic silent broadcasting errors in PPO.
    """
    batch_size = 4
    obs_dim = 33
    action_dim = 13

    # 1. Setup Mock Batch
    # PyTorch expects tensors of shape (Batch, Features)
    mock_obs = torch.randn(batch_size, obs_dim)

    # 2. Initialize Agent
    agent = AbsoluteZeroAgent(obs_dim=obs_dim, action_dim=action_dim)

    # 3. Test Value Output (Critic)
    values = agent.get_value(mock_obs)
    assert values.shape == (
        batch_size,
        1,
    ), f"Critic shape mismatch: {values.shape} != {(batch_size, 1)}"

    # 4. Test Action Sampling (Actor)
    action, log_prob, entropy, value = agent.get_action_and_value(mock_obs)

    assert action.shape == (
        batch_size,
        action_dim,
    ), f"Action shape mismatch: {action.shape} != {(batch_size, action_dim)}"
    assert log_prob.shape == (
        batch_size,
    ), f"Log-prob shape mismatch: {log_prob.shape} != {(batch_size,)}"
    assert entropy.shape == (
        batch_size,
    ), f"Entropy shape mismatch: {entropy.shape} != {(batch_size,)}"
    assert value.shape == (
        batch_size,
        1,
    ), f"Value shape mismatch in joint forward pass: {value.shape}"


def test_agent_gradient_flow():
    """
    [GUARD] Ensures the loss can backpropagate through the network.
    """
    agent = AbsoluteZeroAgent()
    mock_obs = torch.randn(1, 33)

    action, log_prob, _, value = agent.get_action_and_value(mock_obs)

    # Create a dummy loss: maximize value and maximize log_prob
    loss = value.mean() + log_prob.mean()
    loss.backward()

    # Check if gradients populated in the critic
    has_critic_grad = any(p.grad is not None for p in agent.critic.parameters())
    has_actor_grad = any(p.grad is not None for p in agent.actor_mean.parameters())

    assert has_critic_grad, "Gradients failed to flow through the Critic network."
    assert has_actor_grad, "Gradients failed to flow through the Actor network."
