from typing import Any, Dict, Optional, Tuple, Union
import gymnasium as gym
import numpy as np
import pandas as pd


class ObservationScaler:
    def __init__(self, shape: Tuple[int, ...] = (46,), clip_max: float = 5.0):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = 1e-4

    def transform(self, x: np.ndarray, update: bool = True) -> np.ndarray:
        if update:
            self.count += 1
            delta = x - self.mean
            self.mean += delta / self.count
            delta2 = x - self.mean
            self.var += delta * delta2

        variance = self.var / self.count
        std = np.sqrt(variance) + 1e-8
        scaled_x = (x - self.mean) / std
        return np.arcsinh(scaled_x)

    def load_state(
        self, other_scaler: Union[Dict[str, Any], "ObservationScaler"]
    ) -> None:
        if isinstance(other_scaler, dict):
            self.mean = other_scaler["mean"].copy()
            self.var = other_scaler["var"].copy()
            self.count = other_scaler["count"]
        else:
            self.mean = other_scaler.mean.copy()
            self.var = other_scaler.var.copy()
            self.count = other_scaler.count


class ObservationAdapter:
    """
    Translates DataFrames and Series into RL-safe PyTorch-compatible tensors.
    """

    @staticmethod
    def process(
        ensemble: pd.DataFrame,
        macro_row: pd.Series,
        expected_strats: int,
        bm_row: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> np.ndarray:
        # 1. Micro/Strategy Cross-Sectional Stats
        if not ensemble.empty and ensemble.shape[1] == expected_strats:
            strat_mean = ensemble.mean(axis=0).fillna(0.0).to_numpy()
            strat_std = ensemble.std(axis=0, ddof=0).fillna(0.0).to_numpy()
        else:
            strat_mean = np.zeros(expected_strats, dtype=np.float32)
            strat_std = np.zeros(expected_strats, dtype=np.float32)

        # 2. Benchmark Strategy Vector Alignment
        if bm_row is not None:
            if isinstance(bm_row, pd.Series):
                bm_vals = bm_row.fillna(0.0).to_numpy()
            else:
                bm_vals = np.asarray(bm_row)
            if len(bm_vals) != expected_strats:
                bm_vals = np.zeros(expected_strats, dtype=np.float32)
        else:
            bm_vals = np.zeros(expected_strats, dtype=np.float32)

        # 3. Macro Context
        macro_vals = macro_row.fillna(0.0).to_numpy()

        # 4. Assemble and Cast (Mean [N], Std [N], Benchmark [N], Macro [M])
        obs = np.concatenate(
            [
                np.asarray(strat_mean, dtype=np.float32),
                np.asarray(strat_std, dtype=np.float32),
                np.asarray(bm_vals, dtype=np.float32),
                np.asarray(macro_vals, dtype=np.float32),
            ]
        ).astype(np.float32)

        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)


class RLVRGymEnv(gym.Env):
    """
    Gym wrapper for DiscoveryEnv.
    """

    def __init__(self, discovery_env, macro_df: pd.DataFrame):
        super().__init__()
        self.env = discovery_env
        self.macro_df = macro_df

        self.num_features = self.env.cube.shape[1]
        self.num_macro = len(self.macro_df.columns)
        self.obs_dim = 3 * self.num_features + self.num_macro

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_features + 4,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        self.scaler = ObservationScaler(shape=(self.obs_dim,))
        self.is_training = True

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        obs_dict = self.env.reset(seed=seed)
        raw_obs = self._build_obs(obs_dict)
        scaled_obs = self.scaler.transform(raw_obs, update=self.is_training)
        info = {"date": obs_dict.get("date", pd.Timestamp.min)}
        return scaled_obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        raw_obs = self._build_obs(obs_dict)
        scaled_obs = self.scaler.transform(raw_obs, update=self.is_training)
        return scaled_obs, float(reward), terminated, truncated, info

    def _build_obs(self, obs_dict: Dict[str, Any]) -> np.ndarray:
        ensemble = obs_dict["ensemble"]

        if "macro_row" in obs_dict:
            macro_row = obs_dict["macro_row"]
        else:
            date = obs_dict["date"]
            if date in self.macro_df.index:
                macro_row = self.macro_df.loc[date]
            else:
                macro_row = pd.Series(0.0, index=self.macro_df.columns)

        bm_row = obs_dict.get("bm_row", None)

        return ObservationAdapter.process(
            ensemble=ensemble,
            macro_row=macro_row,
            expected_strats=self.num_features,
            bm_row=bm_row,
        )
