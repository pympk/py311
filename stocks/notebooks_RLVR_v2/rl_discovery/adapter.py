import numpy as np
import pandas as pd
import gymnasium as gym
from typing import Dict, Any, Tuple


class ObservationAdapter:
    """
    [DEEP MODULE] Translates Pandas DataFrames into RL-safe PyTorch-compatible tensors.
    Hides all NaN handling, type casting, and scaling logic from the RL algorithm.
    """

    @staticmethod
    def process(
        ensemble: pd.DataFrame, macro_row: pd.Series, expected_strats: int = 11
    ) -> np.ndarray:
        # 1. Micro/Strategy Cross-Sectional Stats
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
    [WRAPPER] Bridges the Absolute Zero engine with Standard RL libraries (CleanRL/SB3).
    """

    def __init__(self, discovery_env, macro_df: pd.DataFrame):
        super().__init__()
        self.env = discovery_env
        self.macro_df = macro_df

        # Define strict bounded spaces for the Agent
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(13,), dtype=np.float32
        )

        # 11 Means + 11 Stds + 11 Macro columns = 33
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(33,), dtype=np.float32
        )

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        obs_dict = self.env.reset()
        return self._build_obs(obs_dict), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # MENTOR NOTE: CleanRL uses the step signature (obs, reward, terminated, truncated, info)
        obs_dict, reward, done, info = self.env.step(action)
        return self._build_obs(obs_dict), float(reward), done, False, info

    def _build_obs(self, obs_dict: Dict[str, Any]) -> np.ndarray:
        date = obs_dict["date"]
        ensemble = obs_dict["ensemble"]

        # Robust fetch of the Macro row
        if date in self.macro_df.index:
            macro_row = self.macro_df.loc[date]
        else:
            macro_row = pd.Series(0.0, index=self.macro_df.columns)

        return ObservationAdapter.process(ensemble, macro_row)
