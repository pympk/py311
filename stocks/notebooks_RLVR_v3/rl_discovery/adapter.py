"""
Observation adapters, scalers, Gym wrappers, and Stratified Vectorized Environment Factories
for continuous action Actor-Critic PPO.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import gymnasium as gym
import numpy as np
import pandas as pd

from core.settings import TradingConfig
from rl_discovery.environment import DiscoveryEnv


class ObservationScaler:
    """
    Welford online running mean and variance tracker with inverse hyperbolic sine scaling.
    """

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
            self.count = float(other_scaler["count"])
        else:
            self.mean = other_scaler.mean.copy()
            self.var = other_scaler.var.copy()
            self.count = float(other_scaler.count)


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
    Gym wrapper for DiscoveryEnv with integrated state scaling and telemetry isolation.
    """

    def __init__(self, discovery_env: DiscoveryEnv, macro_df: pd.DataFrame):
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


def make_stratified_train_envs(
    feature_cube: pd.DataFrame,
    simple_ret_matrix: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    macro_df: pd.DataFrame,
    config: TradingConfig,
    num_envs: int = 8,
    episode_steps: int = 512,
    hist_cutoff_idx: Optional[int] = None,
    initial_scaler_state: Optional[Union[Dict[str, Any], ObservationScaler]] = None,
) -> gym.vector.SyncVectorEnv:
    """
    Constructs a vectorized SyncVectorEnv enforcing Generation 17 Stratified Replay bounds:
    - Envs 0 .. (num_envs // 2 - 1): historical window bounds (0, hist_cutoff_idx).
    - Envs (num_envs // 2) .. (num_envs - 1): recent window bounds (hist_cutoff_idx, len(cal) - 1).
    - If hist_cutoff_idx is None: uniform unstratified sampling across full calendar.
    """
    if hist_cutoff_idx is not None:
        if num_envs < 2 or num_envs % 2 != 0:
            raise ValueError(
                f"num_envs must be an even integer >= 2 for stratified sampling, got {num_envs}."
            )
        if hist_cutoff_idx <= 0 or hist_cutoff_idx >= len(calendar):
            raise ValueError(
                f"hist_cutoff_idx={hist_cutoff_idx} is out of bounds for calendar of length {len(calendar)}."
            )

    half_envs = num_envs // 2

    def make_env_thunk(rank: int) -> Callable[[], RLVRGymEnv]:
        def thunk() -> RLVRGymEnv:
            if hist_cutoff_idx is not None:
                bounds = (
                    (0, hist_cutoff_idx)
                    if rank < half_envs
                    else (hist_cutoff_idx, len(calendar) - 1)
                )
            else:
                bounds = None

            discovery_env = DiscoveryEnv(
                feature_cube=feature_cube,
                simple_ret_matrix=simple_ret_matrix,
                calendar=calendar,
                macro_df=macro_df,
                config=config,
                randomize_start=True,
                episode_steps=episode_steps,
                start_idx_bounds=bounds,
            )
            gym_env = RLVRGymEnv(discovery_env, macro_df)
            gym_env.is_training = True
            if initial_scaler_state is not None:
                gym_env.scaler.load_state(initial_scaler_state)
            return gym_env

        return thunk

    return gym.vector.SyncVectorEnv([make_env_thunk(i) for i in range(num_envs)])


def make_eval_env(
    feature_cube: pd.DataFrame,
    simple_ret_matrix: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    macro_df: pd.DataFrame,
    config: TradingConfig,
    scaler_state: Optional[Union[Dict[str, Any], ObservationScaler]] = None,
) -> RLVRGymEnv:
    """
    Constructs a deterministic validation / test RLVRGymEnv enforcing:
    1. episode_steps = 0 (unbounded, calendar-terminating execution).
    2. is_training = False (strictly freezes observation scaler to eliminate distributional leakage).
    """
    discovery_env = DiscoveryEnv(
        feature_cube=feature_cube,
        simple_ret_matrix=simple_ret_matrix,
        calendar=calendar,
        macro_df=macro_df,
        config=config,
        randomize_start=False,
        episode_steps=0,
    )
    gym_env = RLVRGymEnv(discovery_env, macro_df)
    gym_env.is_training = False
    if scaler_state is not None:
        gym_env.scaler.load_state(scaler_state)
    return gym_env
