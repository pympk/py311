import pytest
import numpy as np
import pandas as pd
from rl_discovery.adapter import ObservationAdapter, RLVRGymEnv


def test_observation_adapter_integrity():
    """Verifies that Pandas structures flatten into exact 49D float32 Tensors safely."""
    strat_cols = [f"Strat_{i}" for i in range(12)]
    ensemble = pd.DataFrame(np.random.randn(3, 12), columns=strat_cols)
    ensemble.iloc[0, 0] = np.nan

    bm_row = pd.Series(np.random.randn(12), index=strat_cols)

    macro_cols = [
        "Mkt_Ret",
        "Mkt_Ret_Z",
        "Macro_Trend",
        "Macro_Trend_Z",
        "Yield_Curve_10Y2Y_Z",
        "Macro_Trend_Vel_Z",
        "Macro_Trend_Mom",
        "Macro_Vix_Z",
        "Macro_Vix_Ratio",
        "Mkt_Vol_63d_Z",
        "Breadth_Above_SMA50",
        "Breadth_Mom_Spread_21d",
        "Breadth_CS_Dispersion_21d",
    ]
    macro_row = pd.Series(np.random.randn(13), index=macro_cols)

    obs = ObservationAdapter.process(
        ensemble, macro_row, expected_strats=12, bm_row=bm_row
    )

    # 12 Mean + 12 Std + 12 Benchmark + 13 Macro = 49
    assert obs.shape == (49,), f"Shape Mismatch: Expected (49,), got {obs.shape}"
    assert obs.dtype == np.float32, f"Type Mismatch: Expected float32, got {obs.dtype}"
    assert not np.isnan(
        obs
    ).any(), "Adapter leaked a NaN into the neural network input!"


class MockDiscoveryEnv:
    """Stubs out DiscoveryEnv with Gymnasium 5-tuple step protocol for 49D observations."""

    def __init__(self):
        self.cube = pd.DataFrame(np.zeros((1, 12)))

    def reset(self, seed=None, **kwargs):
        return {
            "date": pd.Timestamp("2024-01-01"),
            "ensemble": pd.DataFrame(np.random.randn(2, 12)),
            "macro_row": pd.Series(np.zeros(13)),  # 13 macro/breadth dimensions
            "bm_row": pd.Series(np.zeros(12)),
        }

    def step(self, action):
        terminated = False
        truncated = False
        info = {
            "date": pd.Timestamp("2024-01-02"),
            "net_daily_simple_ret": 0.05,
            "bm_daily_simple_ret": 0.0,
            "alpha_daily_simple_ret": 0.05,
            "penalized_alpha_daily_simple_ret": 0.05,
        }
        obs = {
            "date": pd.Timestamp("2024-01-02"),
            "ensemble": pd.DataFrame(np.random.randn(2, 12)),
            "macro_row": pd.Series(np.zeros(13)),  # 13 macro/breadth dimensions
            "bm_row": pd.Series(np.zeros(12)),
        }
        return obs, 0.05, terminated, truncated, info


def test_gym_wrapper_compliance():
    """Verifies the Env complies with Gymnasium specs and handles 49D spaces correctly."""
    mock_macro = pd.DataFrame(
        np.random.randn(2, 13), index=pd.to_datetime(["2024-01-01", "2024-01-02"])
    )

    env = RLVRGymEnv(MockDiscoveryEnv(), mock_macro)

    obs, info = env.reset()
    assert obs.shape == (49,), f"Expected reset obs shape (49,), got {obs.shape}"
    assert env.observation_space.contains(
        obs
    ), "Reset obs does not fit Observation Space"

    action = env.action_space.sample()

    next_obs, reward, terminated, truncated, step_info = env.step(action)
    assert next_obs.shape == (
        49,
    ), f"Expected step obs shape (49,), got {next_obs.shape}"
    assert env.observation_space.contains(
        next_obs
    ), "Step obs does not fit Observation Space"
    assert isinstance(reward, float), "Reward must be a float"
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(step_info, dict)
