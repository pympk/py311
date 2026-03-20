import numpy as np
import pandas as pd
import numpy as np
from typing import Dict, Any

from core.engine import AlphaCache
from core.logic import AlphaLogic, SelectionLogic


# class DiscoveryEnv:
#     """
#     The Absolute Zero Discovery Environment.
#     Wraps AlphaEngine into a time-stepping episode for RL Agent discovery.
#     """

#     def __init__(
#         self,
#         engine: Any,
#         lookbacks: List[int] = [21, 63, 252],
#         holding_period: int = 5,
#         rank_max_offset: int = 50,
#     ):
#         self.engine = engine
#         self.lookbacks = lookbacks
#         self.holding_period = holding_period
#         self.rank_max_offset = rank_max_offset

#         # Precompute the reward matrix once for the whole episode
#         self.engine.precompute_reward_matrix(holding_period)

#         # Setup Timeline
#         self.calendar = self.engine.trading_calendar
#         self.reset()

#     def reset(self, start_date: pd.Timestamp = None) -> Dict[str, np.ndarray]:
#         """Resets the environment to a specific or random start date."""
#         if start_date is None:
#             # Pick a random date that allows for a full lookback and holding period
#             idx = np.random.randint(max(self.lookbacks), len(self.calendar) - 20)
#             self.current_date_idx = idx
#         else:
#             self.current_date_idx = self.calendar.get_loc(start_date)

#         self.equity_curve = [1.0]
#         self.history = []
#         return self._get_observation()

#     def _get_observation(self) -> Dict[str, np.ndarray]:
#         """Gets the 'Vision' for the Agent: Ensemble + Macro."""
#         date = self.calendar[self.current_date_idx]

#         # 1. Ensemble Vision (3D flattened to 2D for the agent)
#         ensemble = self.engine.compute_alpha_ensemble(date, self.lookbacks)

#         # 2. Macro Context (Regime Awareness)
#         context = self.engine.compute_context_vector(date)

#         return {
#             "ensemble": ensemble,  # Used by env to calculate scores
#             "context": context.values,  # Used by Agent as input
#             "date": date,
#         }

#     def step(
#         self, action: np.ndarray
#     ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
#         """
#         Executes one discovery cycle.
#         Action Vector: [Weight_1...Weight_N, Rank_Offset, Rank_Width]
#         """
#         obs_dict = self._get_observation()
#         ensemble = obs_dict["ensemble"]
#         date = obs_dict["date"]

#         # 1. DECODE ACTION
#         # Action is assumed to be normalized [-1, 1] from the Neural Network
#         # Weights for metrics: [Lookbacks * Metrics]
#         n_weights = ensemble.shape[1]
#         weights = action[:n_weights]

#         # Rank selection logic (Your Idea #1)
#         # Map normalized action to discrete rank range
#         raw_offset = action[-2]  # e.g., 0.2
#         raw_width = action[-1]  # e.g., 0.5

#         rank_offset = int(np.interp(raw_offset, [-1, 1], [0, self.rank_max_offset]))
#         rank_width = int(np.interp(raw_width, [-1, 1], [1, 20]))

#         # 2. SYNTHESIZE DISCOVERY SCORE
#         # Vectorized Matrix Multiplication: [Tickers x Features] @ [Features]
#         discovery_scores = ensemble.values @ weights
#         discovery_series = pd.Series(discovery_scores, index=ensemble.index)

#         # 3. SELECT TICKERS (Rank Offset & Width)
#         sorted_tickers = discovery_series.sort_values(ascending=False)
#         selected = sorted_tickers.iloc[
#             rank_offset : rank_offset + rank_width
#         ].index.tolist()

#         # 4. FETCH VERITABLE REWARD
#         # This is the Log-Return from Step 4
#         reward = self.engine.get_batch_reward(date, selected)

#         # 5. UPDATE STATE
#         self.equity_curve.append(self.equity_curve[-1] * np.exp(reward))
#         self.current_date_idx += self.holding_period

#         done = self.current_date_idx >= (len(self.calendar) - self.holding_period - 1)

#         # Log for auditing
#         info = {
#             "date": date,
#             "tickers": selected,
#             "rank_offset": rank_offset,
#             "rank_width": rank_width,
#             "step_reward": reward,
#         }
#         self.history.append(info)

#         return self._get_observation(), reward, done, info


# class DiscoveryEnv:
#     def __init__(
#         self,
#         engine,
#         cache: AlphaCache,
#         holding_period: int = 5,
#         rank_max_offset: int = 50,
#     ):
#         self.engine = engine
#         self.cache = cache
#         self.holding_period = holding_period
#         self.rank_max_offset = rank_max_offset
#         self.calendar = self.engine.trading_calendar

#         # Precompute the reward matrix in the engine (as we did before)
#         self.engine.precompute_reward_matrix(holding_period)

#     def reset(self, start_date: pd.Timestamp = None) -> Dict[str, Any]:
#         if start_date is None:
#             # Ensure we start where cache has data (min lookback)
#             safe_start = max(self.cache.lookbacks)
#             self.current_date_idx = np.random.randint(
#                 safe_start, len(self.calendar) - 20
#             )
#         else:
#             self.current_date_idx = self.calendar.get_loc(start_date)

#         self.equity_curve = [1.0]
#         return self._get_observation()

#     def _get_observation(self) -> Dict[str, Any]:
#         date = self.calendar[self.current_date_idx]

#         # LOOKUP vs CALCULATION
#         ensemble = self.cache.get_vision(date)
#         context = self.engine.compute_context_vector(date)

#         return {
#             "ensemble": ensemble,
#             "context": context.values,
#             "date": date,
#         }

#     def step(self, action: np.ndarray):
#         obs_dict = self._get_observation()
#         ensemble = obs_dict["ensemble"]
#         date = obs_dict["date"]

#         if ensemble.empty:  # Safety for data gaps
#             self.current_date_idx += 1
#             return self._get_observation(), 0.0, False, {}

#         # Vectorized Action decoding
#         n_features = ensemble.shape[1]
#         weights = action[:n_features]

#         # Rank Logic
#         rank_offset = int(np.interp(action[-2], [-1, 1], [0, self.rank_max_offset]))
#         rank_width = int(np.interp(action[-1], [-1, 1], [1, 20]))

#         # High-Speed Scoring: Matrix Multiplication
#         scores = ensemble.values @ weights

#         # Fast Sorting using NumPy argsort (faster than pandas sort_values)
#         idx_top = np.argsort(scores)[::-1][rank_offset : rank_offset + rank_width]
#         selected_tickers = ensemble.index[idx_top].tolist()

#         # Veritable Reward Lookup
#         reward = self.engine.get_batch_reward(date, selected_tickers)

#         self.equity_curve.append(self.equity_curve[-1] * np.exp(reward))
#         self.current_date_idx += self.holding_period

#         done = self.current_date_idx >= (len(self.calendar) - self.holding_period - 1)

#         return self._get_observation(), reward, done, {"date": date}


# class DiscoveryEnv:
#     def __init__(self, engine, cache, holding_period=5):
#         self.engine = engine
#         self.cache = cache
#         self.holding_period = holding_period
#         self.calendar = self.engine.trading_calendar
#         self.engine.precompute_reward_matrix(holding_period)
#         self.reset()

#     def reset(self, start_date=None):
#         if start_date is None:
#             # Random start within valid cache range
#             self.current_date_idx = np.random.randint(252, len(self.calendar) - 20)
#         else:
#             self.current_date_idx = self.calendar.get_loc(start_date)
#         self.equity_curve = [1.0]
#         return self._get_observation()

#     def _get_observation(self):
#         date = self.calendar[self.current_date_idx]
#         ensemble = self.cache.get_vision(date)
#         context = self.engine.compute_context_vector(date)
#         return {"ensemble": ensemble, "context": context.values, "date": date}

#     def step(self, action: np.ndarray):
#         obs_dict = self._get_observation()
#         ensemble = obs_dict["ensemble"]
#         date = obs_dict["date"]

#         # 1. Action Decoding (Weights + Rank Params)
#         n_features = ensemble.shape[1]
#         weights = action[:n_features]

#         # Rank Logic: Offset 0-50, Width 1-10 (YOUR CONSTRAINT)
#         rank_offset = int(np.interp(action[-2], [-1, 1], [0, 50]))
#         rank_width = int(np.interp(action[-1], [-1, 1], [1, 10]))

#         # 2. Ticker Selection
#         if not ensemble.empty:
#             # Dot product: [Tickers x 33] @ [33] = [Tickers]
#             scores = ensemble.values @ weights
#             # Faster than pandas: use numpy to get indices of top scores
#             idx_top = np.argsort(scores)[::-1][rank_offset : rank_offset + rank_width]
#             selected_tickers = ensemble.index[idx_top].tolist()
#         else:
#             selected_tickers = []

#         # 3. Veritable Reward (Cash/Ghost Fix)
#         if not selected_tickers:
#             reward = 0.0  # Return is 0% if staying in cash (relative to cash)
#         else:
#             reward = self.engine.get_batch_reward(date, selected_tickers)

#         # 4. State Update
#         self.equity_curve.append(self.equity_curve[-1] * np.exp(reward))
#         self.current_date_idx += self.holding_period
#         done = self.current_date_idx >= (len(self.calendar) - self.holding_period - 1)

#         # 5. THE AUDIT TRAIL (Fixes the KeyError)
#         info = {
#             "date": date,
#             "tickers": selected_tickers,
#             "ticker_count": len(selected_tickers),
#             "reward": reward,
#             "rank_offset": rank_offset,
#             "rank_width": rank_width,
#         }

#         return self._get_observation(), reward, done, info


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
