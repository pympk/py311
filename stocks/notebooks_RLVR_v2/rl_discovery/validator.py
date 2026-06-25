import torch
import numpy as np
import pandas as pd

from typing import Dict, Any


class AgentEvaluator:
    """
    Deterministic OOS Evaluator with Realistic Overlapping Math and Deep Trade Blotter.
    """

    @staticmethod
    def evaluate(
        agent,
        env,
        device=torch.device("cpu"),
        detailed_log: bool = False,
        metadata: dict | None = None,
    ):
        agent.eval()
        obs, info = env.reset()

        rewards = []
        equity_curve = [1.0]
        dates = [info.get("date", pd.Timestamp.min)]

        trade_blotter = []
        done = False

        # Extract holding period dynamically from the underlying environment
        holding_period = getattr(env.unwrapped.env, "holding_period", 5)

        with torch.no_grad():
            while not done:
                obs_tensor = (
                    torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                )

                action_mean = agent.actor_mean(obs_tensor)
                action = action_mean.cpu().numpy()[0]
                predicted_value = float(agent.get_value(obs_tensor).cpu().numpy()[0][0])

                next_obs, reward, done, _, info = env.step(action)

                if detailed_log:
                    trade_blotter.append(
                        {
                            "decision_date": (
                                info.get("date").strftime("%Y-%m-%d")
                                if info.get("date")
                                else None
                            ),
                            "buy_date": (
                                info.get("buy_date").strftime("%Y-%m-%d")
                                if info.get("buy_date")
                                else None
                            ),
                            "sell_date": (
                                info.get("sell_date").strftime("%Y-%m-%d")
                                if info.get("sell_date")
                                else None
                            ),
                            "universe_size": info.get("universe_size"),
                            "max_score": info.get("max_score"),
                            "min_score": info.get("min_score"),
                            "observation": obs.copy().tolist(),  # <--- Added 33-Dim Observation Here
                            "raw_actions": action.copy().tolist(),
                            "decoded_offset": info.get("offset"),
                            "decoded_width": info.get("width"),
                            "top_3_tickers": info.get("top_3"),
                            "chosen_tickers": info.get("tickers"),
                            "predicted_reward": predicted_value,
                            "actual_reward": float(reward),
                        }
                    )

                rewards.append(reward)

                # REALISTIC OVERLAPPING EQUITY MATH
                sleeve_return = np.exp(reward) - 1.0
                portfolio_impact = sleeve_return / holding_period
                equity_curve.append(equity_curve[-1] * (1.0 + portfolio_impact))

                dates.append(info.get("date", pd.Timestamp.min))
                obs = next_obs

        agent.train()

        returns = pd.Series(rewards)
        annualization_factor = np.sqrt(252)  # Daily stepped
        daily_portfolio_returns = (np.exp(returns) - 1.0) / holding_period
        std_dev = daily_portfolio_returns.std()

        sharpe = (
            ((daily_portfolio_returns.mean() / std_dev) * annualization_factor)
            if std_dev > 1e-8
            else 0.0
        )

        results = {
            "total_return": equity_curve[-1] - 1.0,
            "sharpe_ratio": float(sharpe),
            "equity_curve": equity_curve,
            "dates": dates,
            "steps": len(rewards),
        }

        if detailed_log:
            results["blotter"] = trade_blotter
            results["metadata"] = metadata or {}

        return results


#
