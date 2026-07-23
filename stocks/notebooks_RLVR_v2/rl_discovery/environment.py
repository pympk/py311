import pandas as pd
import numpy as np
from typing import cast
from core.logic import AlphaLogic, SelectionLogic
from core.settings import TradingConfig


class DiscoveryEnv:
    """STATEFUL: The 'Arena' for the Agent."""

    def __init__(
        self,
        feature_cube: pd.DataFrame,
        reward_matrix: pd.DataFrame,
        calendar: pd.DatetimeIndex,
        macro_df: pd.DataFrame,  # NEW: Required for Market Return tracking
        config: TradingConfig | None = None,  # Simple union syntax
    ):
        self.cube = feature_cube
        self.reward_matrix = reward_matrix
        self.calendar = calendar
        self.macro_df = macro_df
        self.config = config or TradingConfig()
        self.holding_period = self.config.holding_period
        self.reset()

    def reset(self, start_date=None):
        if start_date:
            idx = self.calendar.get_loc(start_date)
            self.current_date_idx = cast(int, idx)
        else:
            self.current_date_idx = 0

        # NEW: Track both Absolute Return & Alpha Outperformance
        self.equity_curve = [1.0]
        self.alpha_equity_curve = [1.0]
        return self._get_observation()

    def _get_observation(self):
        date = self.calendar[self.current_date_idx]
        try:
            ensemble = self.cube.xs(date, level="Date")
        except KeyError:
            ensemble = pd.DataFrame()

        return {"ensemble": ensemble, "date": date}

    def step(self, action: np.ndarray):
        date = self.calendar[self.current_date_idx]
        obs_dict = self._get_observation()
        ensemble = obs_dict["ensemble"]

        # 1. Delegate Ticker Selection
        selected_tickers, top_3, offset, width, max_s, min_s = (
            SelectionLogic.apply_action(
                ensemble,
                action,
                self.config.rank_max_offset,
                self.config.rank_max_width,
            )
        )

        # 2. Extract Raw Truth Reward
        log_reward = AlphaLogic.calculate_veritable_reward(
            self.reward_matrix, date, selected_tickers
        )

        # 3. Apply Slippage, Constraints, & Alpha Math
        raw_sleeve_return = np.exp(log_reward) - 1.0
        slippage_applied = 0.0

        if len(selected_tickers) > 0:
            slippage_applied = self.config.slippage_rate
            raw_sleeve_return -= slippage_applied

        mkt_return = (
            self.macro_df.loc[date, "Mkt_Ret"] if date in self.macro_df.index else 0.0
        )
        alpha = raw_sleeve_return - mkt_return

        # Penalize underperformance aggressively for the RL Agent
        penalized_alpha = alpha * self.config.downside_penalty if alpha < 0 else alpha

        # 4. Update Internal State curves
        # Math scales it down by holding period logic per capital deployment
        portfolio_impact = raw_sleeve_return / self.holding_period
        alpha_impact = alpha / self.holding_period

        self.equity_curve.append(self.equity_curve[-1] * (1.0 + portfolio_impact))
        self.alpha_equity_curve.append(
            self.alpha_equity_curve[-1] * (1.0 + alpha_impact)
        )

        # Store the decision index BEFORE we increment it
        decision_idx = self.current_date_idx

        self.current_date_idx += 1

        # We need enough room for T+1 (Buy) and T+1+HP (Sell)
        done = self.current_date_idx >= (len(self.calendar) - self.holding_period - 1)

        # 5. Temporal Alignment & BLOTTER Update
        # Match the Oracle: Buy is T+1, Sell is T+1+HP
        buy_date = self.calendar[decision_idx + 1]
        sell_date = self.calendar[decision_idx + 1 + self.holding_period]

        info = {
            "date": date,
            "buy_date": buy_date,
            "sell_date": sell_date,
            "tickers": selected_tickers,
            "top_3": top_3,
            "universe_size": len(ensemble),
            "offset": offset,
            "width": width,
            "max_score": max_s,
            "min_score": min_s,
            # BLOTTER METRICS
            "raw_log_reward": log_reward,
            "actual_return": raw_sleeve_return,
            "mkt_return": mkt_return,
            "alpha": alpha,
            "penalized_alpha": penalized_alpha,
            "slippage_applied": slippage_applied,
            # NEW: Passing pre-calculated impacts and curves
            "portfolio_impact": portfolio_impact,
            "alpha_impact": alpha_impact,
            "agent_equity": self.equity_curve[-1],
            "alpha_equity": self.alpha_equity_curve[-1],
        }

        # The RL Engine receives penalized_alpha to optimize
        return self._get_observation(), penalized_alpha, done, info
