from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import torch


class AgentEvaluator:
    """
    Deterministic OOS Evaluator with Mark-to-Market (MTM) FIFO Accounting.
    """

    @staticmethod
    def evaluate(
        agent,
        env,
        device: torch.device = torch.device("cpu"),
        detailed_log: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        agent.eval()
        obs, info = env.reset()

        net_simple_returns_list: List[float] = []
        bm_simple_returns_list: List[float] = []
        equity_curve: List[float] = [1.0]
        bm_equity_curve: List[float] = [1.0]
        alpha_equity_curve: List[float] = [1.0]
        dates: List[Any] = [info.get("date", pd.Timestamp.min)]

        trade_blotter: List[Dict[str, Any]] = []
        done = False

        with torch.no_grad():
            while not done:
                obs_tensor = (
                    torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                )

                action_mean = agent.actor_mean(obs_tensor)
                action = torch.clamp(action_mean, -1.0, 1.0).cpu().numpy()[0]
                predicted_value = float(agent.get_value(obs_tensor).cpu().numpy()[0][0])
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)

                net_daily_simple_ret = float(info["net_daily_simple_ret"])
                bm_daily_simple_ret = float(info["bm_daily_simple_ret"])
                alpha_daily_simple_ret = float(info["alpha_daily_simple_ret"])
                penalized_alpha_daily_simple_ret = float(
                    info["penalized_alpha_daily_simple_ret"]
                )

                date_val = info.get("date")
                date_str = (
                    date_val.strftime("%Y-%m-%d")
                    if hasattr(date_val, "strftime")
                    else str(date_val) if date_val is not None else None
                )
                buy_date_val = info.get("buy_date")
                buy_date_str = (
                    buy_date_val.strftime("%Y-%m-%d")
                    if hasattr(buy_date_val, "strftime")
                    else str(buy_date_val) if buy_date_val is not None else None
                )
                sell_date_val = info.get("sell_date")
                sell_date_str = (
                    sell_date_val.strftime("%Y-%m-%d")
                    if hasattr(sell_date_val, "strftime")
                    else str(sell_date_val) if sell_date_val is not None else None
                )

                if detailed_log:
                    trade_blotter.append(
                        {
                            "date": date_str,
                            "decision_date": date_str,
                            "buy_date": buy_date_str,
                            "sell_date": sell_date_str,
                            "universe_size": info.get("universe_size"),
                            "max_score": info.get("max_score"),
                            "min_score": info.get("min_score"),
                            "observation": obs.copy().tolist(),
                            "raw_actions": action.copy().tolist(),
                            "offset": info.get("offset"),
                            "width": info.get("width"),
                            "top_3": info.get("top_3"),
                            "tickers": info.get("tickers"),
                            "weight_active": info.get("weight_active"),
                            "weight_benchmark": info.get("weight_benchmark"),
                            "weight_cash": info.get("weight_cash"),
                            "equity_exposure": info.get("equity_exposure"),
                            "active_tilt": info.get("active_tilt"),
                            "predicted_reward": predicted_value,
                            "rl_reward_received": float(reward),
                            "gross_stock_daily_simple_ret": info.get(
                                "gross_stock_daily_simple_ret"
                            ),
                            "bm_daily_simple_ret": bm_daily_simple_ret,
                            "cash_daily_simple_ret": info.get(
                                "cash_daily_simple_ret", 0.0
                            ),
                            "slippage_daily_simple_loss": info.get(
                                "slippage_daily_simple_loss", 0.0
                            ),
                            "gross_daily_simple_ret": info.get(
                                "gross_daily_simple_ret"
                            ),
                            "net_daily_simple_ret": net_daily_simple_ret,
                            "alpha_daily_simple_ret": alpha_daily_simple_ret,
                            "penalized_alpha_daily_simple_ret": penalized_alpha_daily_simple_ret,
                            "agent_equity": equity_curve[-1]
                            * (1.0 + net_daily_simple_ret),
                            "bm_equity": bm_equity_curve[-1]
                            * (1.0 + bm_daily_simple_ret),
                            "alpha_equity": alpha_equity_curve[-1]
                            * (1.0 + penalized_alpha_daily_simple_ret),
                        }
                    )

                net_simple_returns_list.append(net_daily_simple_ret)
                bm_simple_returns_list.append(bm_daily_simple_ret)

                equity_curve.append(equity_curve[-1] * (1.0 + net_daily_simple_ret))
                bm_equity_curve.append(
                    bm_equity_curve[-1] * (1.0 + bm_daily_simple_ret)
                )
                alpha_equity_curve.append(
                    alpha_equity_curve[-1] * (1.0 + penalized_alpha_daily_simple_ret)
                )

                dates.append(info.get("date", pd.Timestamp.min))
                obs = next_obs

        agent.train()

        port_arr = np.asarray(net_simple_returns_list, dtype=np.float64)
        bm_arr = np.asarray(bm_simple_returns_list, dtype=np.float64)
        eq_arr = np.asarray(equity_curve, dtype=np.float64)
        bm_eq_arr = np.asarray(bm_equity_curve, dtype=np.float64)

        total_return = float(eq_arr[-1] - 1.0) if eq_arr.size > 0 else 0.0
        bm_total_return = float(bm_eq_arr[-1] - 1.0) if bm_eq_arr.size > 0 else 0.0
        excess_return = float(total_return - bm_total_return)

        # 1. Raw Portfolio Sharpe Ratio
        mean_ret = float(np.mean(port_arr)) if port_arr.size > 0 else 0.0
        std_dev = float(np.std(port_arr, ddof=1)) if port_arr.size > 1 else 0.0
        sharpe = float((mean_ret / std_dev) * np.sqrt(252.0)) if std_dev > 1e-8 else 0.0

        # 2. Information Ratio (Alpha Spread Sharpe)
        active_spread = port_arr - bm_arr
        active_mean = float(np.mean(active_spread)) if active_spread.size > 0 else 0.0
        active_std = (
            float(np.std(active_spread, ddof=1)) if active_spread.size > 1 else 0.0
        )
        info_ratio = (
            float((active_mean / active_std) * np.sqrt(252.0))
            if active_std > 1e-8
            else 0.0
        )

        # 3. Sortino Ratio (on Alpha Spread)
        downside_alpha = active_spread[active_spread < 0.0]
        downside_alpha_std = (
            float(np.std(downside_alpha, ddof=1)) if downside_alpha.size > 1 else 0.0
        )
        sortino = (
            float((active_mean / downside_alpha_std) * np.sqrt(252.0))
            if downside_alpha_std > 1e-8
            else (info_ratio if info_ratio > 0.0 else 0.0)
        )

        # 4. Maximum Drawdown
        running_max = np.maximum.accumulate(eq_arr)
        drawdowns = (eq_arr - running_max) / np.maximum(running_max, 1e-8)
        max_drawdown = float(np.min(drawdowns)) if drawdowns.size > 0 else 0.0

        # 5. Beta to Benchmark
        bm_var = float(np.var(bm_arr, ddof=1)) if bm_arr.size > 1 else 0.0
        if bm_var > 1e-8 and port_arr.size > 1:
            cov_mat = np.cov(port_arr, bm_arr)
            beta = float(cov_mat[0, 1] / bm_var)
        else:
            beta = 1.0

        results: Dict[str, Any] = {
            "total_return": total_return,
            "bm_total_return": bm_total_return,
            "excess_return": excess_return,
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": max_drawdown,
            "information_ratio": float(info_ratio),
            "beta": float(beta),
            "equity_curve": equity_curve,
            "bm_equity_curve": bm_equity_curve,
            "alpha_equity_curve": alpha_equity_curve,
            "dates": dates,
            "steps": len(net_simple_returns_list),
        }

        if detailed_log:
            results["blotter"] = trade_blotter
            results["metadata"] = metadata or {}

        return results
