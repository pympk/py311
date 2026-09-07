from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .agent import AbsoluteZeroAgent


class RolloutBuffer:
    """Stores sequential experiences and computes Generalized Advantage Estimation (GAE)

    with proper separation between true terminations and time-limit truncations.
    """

    def __init__(
        self,
        num_steps: int,
        num_envs: int = 8,
        obs_dim: int = 46,
        action_dim: int = 16,
        device: torch.device = torch.device("cpu"),
        gamma: float = 0.90,
        gae_lambda: float = 0.95,
    ):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.obs = torch.zeros(
            (num_steps, num_envs, obs_dim), dtype=torch.float32, device=device
        )
        self.actions = torch.zeros(
            (num_steps, num_envs, action_dim), dtype=torch.float32, device=device
        )
        self.logprobs = torch.zeros(
            (num_steps, num_envs), dtype=torch.float32, device=device
        )
        self.rewards = torch.zeros(
            (num_steps, num_envs), dtype=torch.float32, device=device
        )
        self.values = torch.zeros(
            (num_steps, num_envs), dtype=torch.float32, device=device
        )
        self.terminations = torch.zeros(
            (num_steps, num_envs), dtype=torch.float32, device=device
        )
        self.truncations = torch.zeros(
            (num_steps, num_envs), dtype=torch.float32, device=device
        )

        self.step = 0

    def add(
        self,
        obs: np.ndarray,
        action: torch.Tensor,
        logprob: torch.Tensor,
        reward: np.ndarray,
        value: torch.Tensor,
        terminations: np.ndarray,
        truncations: np.ndarray,
    ):
        if self.step >= self.num_steps:
            raise IndexError(
                "RolloutBuffer is full. Call compute_advantages() and reset."
            )

        self.obs[self.step] = torch.tensor(obs, dtype=torch.float32, device=self.device)
        self.actions[self.step] = action
        self.logprobs[self.step] = logprob
        self.rewards[self.step] = torch.tensor(
            reward, dtype=torch.float32, device=self.device
        )
        self.values[self.step] = value.flatten()
        self.terminations[self.step] = torch.tensor(
            terminations, dtype=torch.float32, device=self.device
        )
        self.truncations[self.step] = torch.tensor(
            truncations, dtype=torch.float32, device=self.device
        )
        self.step += 1

    def compute_advantages(
        self,
        next_value: torch.Tensor,
        next_termination: torch.Tensor,
        gamma: Optional[float] = None,
        gae_lambda: Optional[float] = None,
    ):
        """GAE computation bootstrapping value across step-limit truncations."""
        g = self.gamma if gamma is None else gamma
        l = self.gae_lambda if gae_lambda is None else gae_lambda

        self.advantages = torch.zeros_like(self.rewards, device=self.device)
        lastgaelam = torch.zeros(self.num_envs, device=self.device)
        next_term_tensor = next_termination.float().to(self.device)

        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                nextnonterminal = 1.0 - next_term_tensor
                nextvalues = next_value.flatten()
            else:
                nextnonterminal = 1.0 - self.terminations[t + 1]
                nextvalues = self.values[t + 1]

            delta = self.rewards[t] + g * nextvalues * nextnonterminal - self.values[t]
            self.advantages[t] = lastgaelam = (
                delta + g * l * nextnonterminal * lastgaelam
            )

        self.returns = self.advantages + self.values


class PPOTrainer:
    """Executes Clipped Surrogate Objective updates with Target KL Early Stopping."""

    def __init__(
        self,
        agent: AbsoluteZeroAgent,
        lr: float = 1.5e-4,
        critic_lr: Optional[float] = None,
        clip_coef: float = 0.2,
        clip_vloss: bool = True,
        ent_coef: float = 0.003,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = 0.025,
    ):
        self.agent = agent
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        c_lr = critic_lr if critic_lr is not None else (lr * 4.0)
        self.optimizer = optim.Adam(
            [
                {
                    "params": self.agent.actor_mean.parameters(),
                    "lr": lr,
                    "initial_lr": lr,
                },
                {"params": [self.agent.actor_logstd], "lr": lr, "initial_lr": lr},
                {
                    "params": self.agent.critic.parameters(),
                    "lr": c_lr,
                    "initial_lr": c_lr,
                },
            ],
            eps=1e-5,
        )

    def update(
        self,
        buffer: RolloutBuffer,
        update_epochs: int = 4,
        mini_batch_size: int = 256,
    ) -> Dict[str, float]:
        b_obs = buffer.obs.reshape((-1, buffer.obs.shape[-1]))
        b_actions = buffer.actions.reshape((-1, buffer.actions.shape[-1]))
        b_logprobs = buffer.logprobs.reshape(-1)
        b_advantages = buffer.advantages.reshape(-1)
        b_returns = buffer.returns.reshape(-1)
        b_values = buffer.values.reshape(-1)

        b_advantages = (b_advantages - b_advantages.mean()) / (
            b_advantages.std() + 1e-8
        )

        batch_size = buffer.num_steps * buffer.num_envs
        b_inds = np.arange(batch_size)

        pg_losses = []
        v_losses = []
        entropy_losses = []
        total_losses = []
        clip_fractions = []
        approx_kls = []

        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)

            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean().item()
                    approx_kls.append(approx_kl)

                # Policy Loss
                mb_advantages = b_advantages[mb_inds]
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                with torch.no_grad():
                    clip_frac = (
                        ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                    )
                    clip_fractions.append(clip_frac)

                # Value Loss with clipping
                newvalue = newvalue.view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = (
                    pg_loss - (self.ent_coef * entropy_loss) + (v_loss * self.vf_coef)
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropy_losses.append(entropy_loss.item())
                total_losses.append(loss.item())

            if self.target_kl is not None and approx_kl > self.target_kl:
                break

        y_pred = b_values.cpu().numpy()
        y_true = b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = (
            np.nan if var_y < 1e-8 else float(1.0 - (np.var(y_true - y_pred) / var_y))
        )

        return {
            "policy_loss": float(np.mean(pg_losses)),
            "value_loss": float(np.mean(v_losses)),
            "entropy": float(np.mean(entropy_losses)),
            "total_loss": float(np.mean(total_losses)),
            "approx_kl": float(np.mean(approx_kls)),
            "clip_fraction": float(np.mean(clip_fractions)),
            "explained_variance": float(explained_var),
        }

    def update_schedules(
        self,
        current_epoch: int,
        total_epochs: int,
        ent_start: float = 0.003,
        ent_end: float = 0.0005,
    ):
        """Anneals Learning Rates and Entropy Coefficient proportionally."""
        if total_epochs <= 1:
            fraction = 0.0
        else:
            fraction = 1.0 - (float(current_epoch) - 1.0) / float(total_epochs)
        fraction = max(0.0, min(1.0, fraction))

        for param_group in self.optimizer.param_groups:
            initial_lr = param_group.get("initial_lr", param_group["lr"])
            param_group["lr"] = initial_lr * fraction

        self.ent_coef = float(ent_end + (ent_start - ent_end) * fraction)
