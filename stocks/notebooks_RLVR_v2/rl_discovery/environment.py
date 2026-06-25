import pandas as pd
import numpy as np

from typing import cast

from core.logic import AlphaLogic, SelectionLogic


class DiscoveryEnv:
    """STATEFUL: The 'Arena' for the Agent."""

    def __init__(
        self,
        feature_cube: pd.DataFrame,
        reward_matrix: pd.DataFrame,
        calendar: pd.DatetimeIndex,
        holding_period: int = 5,
    ):
        self.cube = feature_cube
        self.reward_matrix = reward_matrix
        self.calendar = calendar
        self.holding_period = holding_period
        self.reset()

    def reset(self, start_date=None):
        if start_date:
            # Use cast to force Pylance to treat this as an int
            idx = self.calendar.get_loc(start_date)
            self.current_date_idx = cast(int, idx)
        else:
            # Start at the beginning of the provided calendar slice
            self.current_date_idx = 0

        self.equity_curve = [1.0]
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

        # 1. Delegate Ticker Selection (Unpacking the new V6 Tuple)
        selected_tickers, top_3, offset, width, max_s, min_s = (
            SelectionLogic.apply_action(ensemble, action)
        )

        # 2. Delegate Reward Calculation
        reward = AlphaLogic.calculate_veritable_reward(
            self.reward_matrix, date, selected_tickers
        )

        # 3. Update internal state (REALISTIC OVERLAPPING MATH)
        # Agent gets raw reward for learning, but equity assumes 1/5th capital deployment
        sleeve_return = np.exp(reward) - 1.0
        portfolio_impact = sleeve_return / self.holding_period
        self.equity_curve.append(self.equity_curve[-1] * (1.0 + portfolio_impact))

        # DAILY STEPPING
        self.current_date_idx += 1

        done = self.current_date_idx >= (len(self.calendar) - self.holding_period - 1)

        # 4. Temporal Alignment for Debugging
        buy_date = self.calendar[self.current_date_idx]
        sell_date = self.calendar[self.current_date_idx + self.holding_period]

        info = {
            "date": date,
            "buy_date": buy_date,
            "sell_date": sell_date,
            "tickers": selected_tickers,
            "top_3": top_3,
            "reward": reward,
            "universe_size": len(ensemble),
            "offset": offset,
            "width": width,
            "max_score": max_s,
            "min_score": min_s,
        }
        return self._get_observation(), reward, done, info


#
