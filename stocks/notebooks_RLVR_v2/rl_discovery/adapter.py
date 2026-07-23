import numpy as np
import pandas as pd
import gymnasium as gym
from typing import Dict, Any, Tuple


# ---> NEW: Running Observation Scaler # <---
class ObservationScaler:
    def __init__(self, shape=(33,), clip_max=5.0):
        # Dynamically accepts the incoming shape tuple (e.g. 35)
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = 1e-4

    def transform(self, x: np.ndarray, update: bool = True) -> np.ndarray:
        if update:
            # Welford's online algorithm to update mean and variance on the fly
            self.count += 1
            delta = x - self.mean
            self.mean += delta / self.count
            delta2 = x - self.mean
            self.var += delta * delta2

        # Calculate standard deviation safely
        variance = self.var / self.count
        std = np.sqrt(variance) + 1e-8

        # Z-score standardization
        scaled_x = (x - self.mean) / std

        # ---> NEW: Inverse Hyperbolic Sine (asinh) Transformation
        # Acts linearly near 0, but logarithmically for large outliers.
        # A 10-sigma outlier becomes ~3.0, keeping the NN safe while preserving direction.
        return np.arcsinh(scaled_x)

    def load_state(self, other_scaler):
        """Syncs knowledge from the training environment to validation/test environments"""
        self.mean = other_scaler.mean.copy()
        self.var = other_scaler.var.copy()
        self.count = other_scaler.count


class ObservationAdapter:
    """
    [DEEP MODULE] Translates Pandas DataFrames into RL-safe PyTorch-compatible tensors.
    Hides all NaN handling, type casting, and scaling logic from the RL algorithm.
    """

    @staticmethod
    def process(
        ensemble: pd.DataFrame, macro_row: pd.Series, expected_strats: int
    ) -> np.ndarray:
        # 1. Micro/Strategy Cross-Sectional Stats (Dynamic Match check)
        if not ensemble.empty and ensemble.shape[1] == expected_strats:
            # We use ddof=0 to avoid NaN if there is only 1 ticker
            strat_mean = ensemble.mean(axis=0).fillna(0.0).values
            strat_std = ensemble.std(axis=0, ddof=0).fillna(0.0).values
        else:
            # Edge Case: Universe is completely empty today, or shape mismatch
            strat_mean = np.zeros(expected_strats)
            strat_std = np.zeros(expected_strats)

        # 2. Macro Context
        macro_vals = macro_row.fillna(0.0).values

        # 3. Assemble and Cast
        # MENTOR NOTE: We MUST cast to np.float32. PyTorch defaults to float32.
        # If we pass Pandas' default float64, PyTorch will throw a runtime type mismatch error.

        # obs = np.concatenate([strat_mean, strat_std, macro_vals]).astype(np.float32)
        obs = np.concatenate(
            [np.asarray(strat_mean), np.asarray(strat_std), np.asarray(macro_vals)]
        ).astype(np.float32)

        # Guardrail: Prevent Neural Network explosion from rogue Infs
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs


class RLVRGymEnv(gym.Env):
    """
    [WRAPPER] Bridges the Absolute Zero engine with Standard RL libraries.
    """

    def __init__(self, discovery_env, macro_df: pd.DataFrame):
        super().__init__()
        self.env = discovery_env
        self.macro_df = macro_df

        # ---> DYNAMIC SPACE DETECTION <---
        self.num_features = self.env.cube.shape[1]  # Resolves to 12
        self.num_macro = len(self.macro_df.columns)  # Resolves to 11
        self.obs_dim = 2 * self.num_features + self.num_macro  # 12*2 + 11 = 35

        # Define dynamically scaled spaces
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_features + 2,), dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # Initialize the scaler matching the exact observation dimension (35)
        self.scaler = ObservationScaler(shape=(self.obs_dim,))
        self.is_training = True

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        obs_dict = self.env.reset()

        # ---> NEW: Scale the observation
        raw_obs = self._build_obs(obs_dict)
        scaled_obs = self.scaler.transform(raw_obs, update=self.is_training)
        return scaled_obs, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # MENTOR NOTE: CleanRL uses the step signature (obs, reward, terminated, truncated, info)
        obs_dict, reward, done, info = self.env.step(action)

        # ---> NEW: Scale the observation # <---
        raw_obs = self._build_obs(obs_dict)
        scaled_obs = self.scaler.transform(raw_obs, update=self.is_training)
        return scaled_obs, float(reward), done, False, info

    def _build_obs(self, obs_dict: Dict[str, Any]) -> np.ndarray:
        date = obs_dict["date"]
        ensemble = obs_dict["ensemble"]

        # Robust fetch of the Macro row
        if date in self.macro_df.index:
            macro_row = self.macro_df.loc[date]
        else:
            macro_row = pd.Series(0.0, index=self.macro_df.columns)

        # Pass self.num_features dynamically to avoid static fallback of zeros
        return ObservationAdapter.process(
            ensemble, macro_row, expected_strats=self.num_features
        )


#
