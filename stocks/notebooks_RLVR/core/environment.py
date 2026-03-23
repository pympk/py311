import numpy as np
import pandas as pd
import numpy as np

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
            self.current_date_idx = self.calendar.get_loc(start_date)
        else:
            # Start far enough in so we have cache data
            self.current_date_idx = 252
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

        # 1. Delegate Ticker Selection (Stateless)
        selected_tickers = SelectionLogic.apply_action(ensemble, action)

        # 2. Delegate Reward Calculation (Stateless)
        reward = AlphaLogic.calculate_veritable_reward(
            self.reward_matrix, date, selected_tickers
        )

        # 3. Update internal state
        self.equity_curve.append(self.equity_curve[-1] * np.exp(reward))
        self.current_date_idx += self.holding_period

        done = self.current_date_idx >= (len(self.calendar) - self.holding_period - 1)

        info = {"date": date, "tickers": selected_tickers, "reward": reward}
        return self._get_observation(), reward, done, info


#
