import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .agent import AbsoluteZeroAgent


class RolloutBuffer:
    """
    [DEEP MODULE] Stores sequential experiences and computes Generalized Advantage Estimation (GAE).
    Pre-allocates tensors for memory efficiency.
    """

    def __init__(
        self,
        num_steps: int,
        obs_dim: int = 33,
        action_dim: int = 13,
        device: torch.device = torch.device("cpu"),
    ):
        self.num_steps = num_steps
        self.device = device

        # Pre-allocate memory (Batch, Dim)
        self.obs = torch.zeros((num_steps, obs_dim), dtype=torch.float32).to(device)
        self.actions = torch.zeros((num_steps, action_dim), dtype=torch.float32).to(
            device
        )
        self.logprobs = torch.zeros((num_steps,), dtype=torch.float32).to(device)
        self.rewards = torch.zeros((num_steps,), dtype=torch.float32).to(device)
        self.values = torch.zeros((num_steps,), dtype=torch.float32).to(device)
        self.dones = torch.zeros((num_steps,), dtype=torch.float32).to(device)

        self.step = 0

    def add(
        self,
        obs: np.ndarray,
        action: torch.Tensor,
        logprob: torch.Tensor,
        reward: float,
        value: torch.Tensor,
        done: bool,
    ):
        """[GUARD] Traps buffer overflow."""
        if self.step >= self.num_steps:
            raise IndexError(
                "RolloutBuffer is full. Call compute_advantages() and reset."
            )

        self.obs[self.step] = torch.tensor(obs, dtype=torch.float32).to(self.device)
        self.actions[self.step] = action
        self.logprobs[self.step] = logprob
        self.rewards[self.step] = reward
        self.values[self.step] = value.flatten()
        self.dones[self.step] = done
        self.step += 1

    def compute_advantages(
        self,
        next_value: torch.Tensor,
        next_done: bool,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        Calculates GAE.
        Advantage = Return - Baseline (Value).
        Positive Advantage = Action was better than expected.
        """
        self.advantages = torch.zeros_like(self.rewards).to(self.device)
        lastgaelam = 0

        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                nextnonterminal = 1.0 - int(next_done)
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - self.dones[t + 1]
                nextvalues = self.values[t + 1]

            delta = (
                self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
            )
            self.advantages[t] = lastgaelam = (
                delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            )

        self.returns = self.advantages + self.values


class PPOTrainer:
    """
    Executes the Clipped Surrogate Objective update over the Experience Buffer.
    """

    def __init__(
        self,
        agent: AbsoluteZeroAgent,
        lr: float = 3e-4,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
    ):
        self.agent = agent
        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr, eps=1e-5)

        # Hyperparameters
        self.clip_coef = clip_coef  # Prevents updating policy too much
        self.ent_coef = ent_coef  # Encourages exploration
        self.vf_coef = vf_coef  # Critic loss scaling
        self.max_grad_norm = max_grad_norm

    def update(
        self, buffer: RolloutBuffer, update_epochs: int = 4, mini_batch_size: int = 64
    ) -> dict:
        """
        Executes PPO update step and returns key diagnostic metrics.
        """
        b_obs = buffer.obs
        b_actions = buffer.actions
        b_logprobs = buffer.logprobs
        b_advantages = buffer.advantages
        b_returns = buffer.returns

        # Batch normalization of advantages (critical for stable learning)
        b_advantages = (b_advantages - b_advantages.mean()) / (
            b_advantages.std() + 1e-8
        )

        buffer_size = buffer.num_steps
        b_inds = np.arange(buffer_size)

        # Metric storage
        pg_losses = []
        v_losses = []
        entropy_losses = []
        total_losses = []
        clip_fractions = []
        approx_kls = []

        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, buffer_size, mini_batch_size):
                end = start + mini_batch_size
                mb_inds = b_inds[start:end]

                # 1. Get new probabilities for the old actions using updated network
                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )

                # 2. Policy Loss (Clipped Surrogate)
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # Approximate KL divergence estimate for stability checking [2]
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean().item()
                    approx_kls.append(approx_kl)

                mb_advantages = b_advantages[mb_inds]
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Calculate clip fraction
                with torch.no_grad():
                    clip_frac = (
                        ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                    )
                    clip_fractions.append(clip_frac)

                # 3. Value Loss (MSE between prediction and actual return)
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # 4. Entropy Loss (Maximize entropy to encourage exploration)
                entropy_loss = entropy.mean()

                # 5. Total Loss
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                # Save raw step losses
                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropy_losses.append(entropy_loss.item())
                total_losses.append(loss.item())

                # 6. Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

        # Compute Explained Variance of Value Predictions
        y_pred = buffer.values.cpu().numpy()
        y_true = b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = (
            np.nan if var_y == 0 else 1.0 - (np.var(y_true - y_pred) / var_y)
        )

        # Aggregate metrics over all update epochs and steps
        diagnostics = {
            "policy_loss": np.mean(pg_losses),
            "value_loss": np.mean(v_losses),
            "entropy": np.mean(entropy_losses),
            "total_loss": np.mean(total_losses),
            "approx_kl": np.mean(approx_kls),
            "clip_fraction": np.mean(clip_fractions),
            "explained_variance": float(explained_var),
        }
        return diagnostics
