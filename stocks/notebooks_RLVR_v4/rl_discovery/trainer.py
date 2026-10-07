from typing import Dict, List, Optional
import numpy as np
import torch
import torch.distributions.kl as dist_kl
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
    """Executes Clipped Surrogate Objective updates with Target KL Early Stopping
    and optional Walk-Forward Policy KL Anchor Regularization.
    """

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
        anchor_agent: Optional[AbsoluteZeroAgent] = None,
        kl_anchor_coef: float = 0.0,
        seed: Optional[int] = None,
    ):
        self.agent = agent
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.anchor_agent = anchor_agent
        self.kl_anchor_coef = float(kl_anchor_coef)
        self.rng = np.random.default_rng(seed)

        if self.anchor_agent is not None:
            self.anchor_agent.eval()
            for param in self.anchor_agent.parameters():
                param.requires_grad = False
            try:
                agent_device = next(self.agent.parameters()).device
                self.anchor_agent.to(agent_device)
            except (StopIteration, RuntimeError):
                pass

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
        stratified_sampling: bool = False,
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

        if stratified_sampling:
            if buffer.num_envs < 2 or buffer.num_envs % 2 != 0:
                raise ValueError(
                    f"stratified_sampling requires an even number of environments (num_envs >= 2), got {buffer.num_envs}."
                )
            if mini_batch_size % 2 != 0:
                raise ValueError(
                    f"mini_batch_size must be even for 50/50 stratified sampling, got {mini_batch_size}."
                )
            half_envs = buffer.num_envs // 2
            all_inds = np.arange(batch_size)
            env_ids = all_inds % buffer.num_envs
            hist_inds_base = all_inds[env_ids < half_envs]
            rec_inds_base = all_inds[env_ids >= half_envs]
        else:
            b_inds = np.arange(batch_size)

        pg_losses: List[float] = []
        v_losses: List[float] = []
        entropy_losses: List[float] = []
        total_losses: List[float] = []
        clip_fractions: List[float] = []
        approx_kls: List[float] = []
        anchor_kls: List[float] = []

        for epoch in range(update_epochs):
            epoch_kls = []

            if stratified_sampling:
                shuffled_hist = self.rng.permutation(hist_inds_base)
                shuffled_rec = self.rng.permutation(rec_inds_base)
                half_mb = mini_batch_size // 2

                mini_batches = []
                for start in range(0, len(shuffled_hist), half_mb):
                    end = start + half_mb
                    h_chunk = shuffled_hist[start:end]
                    r_chunk = shuffled_rec[start:end]
                    mini_batches.append(np.concatenate([h_chunk, r_chunk]))
            else:
                shuffled_inds = self.rng.permutation(b_inds)
                mini_batches = [
                    shuffled_inds[start : start + mini_batch_size]
                    for start in range(0, batch_size, mini_batch_size)
                ]

            for mb_inds in mini_batches:
                mb_obs = b_obs[mb_inds]
                mb_actions = b_actions[mb_inds]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                    mb_obs, mb_actions
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = torch.clamp(logratio, -20.0, 20.0).exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean().item()
                    approx_kls.append(approx_kl)
                    epoch_kls.append(approx_kl)

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

                # Walk-Forward Behavioral Anchor Regularization (Analytical Gaussian KL)
                if self.anchor_agent is not None and self.kl_anchor_coef > 0.0:
                    with torch.no_grad():
                        anchor_dist = self.anchor_agent.get_distribution(mb_obs)
                    curr_dist = self.agent.get_distribution(mb_obs)
                    # Sum analytical KL over action dimensions, average over mini-batch
                    kl_div = (
                        dist_kl.kl_divergence(anchor_dist, curr_dist).sum(dim=-1).mean()
                    )
                    anchor_kl_val = kl_div.item()
                    anchor_loss = self.kl_anchor_coef * kl_div
                else:
                    anchor_kl_val = 0.0
                    anchor_loss = 0.0

                loss = (
                    pg_loss
                    - (self.ent_coef * entropy_loss)
                    + (v_loss * self.vf_coef)
                    + anchor_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropy_losses.append(entropy_loss.item())
                total_losses.append(loss.item())
                anchor_kls.append(anchor_kl_val)

            if (
                self.target_kl is not None
                and len(epoch_kls) > 0
                and np.mean(epoch_kls) > self.target_kl
            ):
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
            "anchor_kl": float(np.mean(anchor_kls)) if anchor_kls else 0.0,
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
