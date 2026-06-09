import torch
import numpy as np
import pandas as pd
from typing import Dict, Any
from .agent import AbsoluteZeroAgent


class AgentEvaluator:
    """
    [DEEP MODULE] Executes Deterministic Out-Of-Sample (OOS) Rollouts.
    Removes exploration noise to measure the true learned policy.
    """

    @staticmethod
    def evaluate(
        agent: AbsoluteZeroAgent, env, device: torch.device = torch.device("cpu")
    ) -> Dict[str, Any]:

        # 1. Engage Inference Mode (Locks Batch Norms, Dropouts, and allows manual sampling)
        agent.eval()

        obs, info = env.reset()

        rewards = []
        equity_curve = [1.0]
        dates = [info.get("date", pd.Timestamp.min)]

        done = False

        # 2. Disable Gradient Tracking (Saves memory and speeds up computation)
        with torch.no_grad():
            while not done:
                # Prepare tensor
                obs_tensor = (
                    torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                )

                # 3. Deterministic Execution: We bypass the Normal distribution sampling
                # and directly take the raw 'actor_mean' from the network.
                action_mean = agent.actor_mean(obs_tensor)
                action = action_mean.cpu().numpy()[0]

                # 4. Step Environment
                obs, reward, done, _, info = env.step(action)

                rewards.append(reward)
                # Reverse the log-return back to standard compounding
                equity_curve.append(equity_curve[-1] * np.exp(reward))
                dates.append(info.get("date", pd.Timestamp.min))

        # 5. Restore Training Mode
        agent.train()

        # 6. Calculate Standard Quant Metrics
        returns = pd.Series(rewards)

        # Annualize Sharpe based on the environment's holding period
        holding_period = getattr(env.unwrapped, "holding_period", 5)
        annualization_factor = np.sqrt(252 / holding_period)

        # FIX: Epsilon threshold for floating point stability
        std_dev = returns.std()
        sharpe = (
            (returns.mean() / std_dev) * annualization_factor if std_dev > 1e-8 else 0.0
        )

        return {
            "total_return": equity_curve[-1] - 1.0,
            "sharpe_ratio": float(sharpe),
            "equity_curve": equity_curve,
            "dates": dates,
            "steps": len(rewards),
        }
