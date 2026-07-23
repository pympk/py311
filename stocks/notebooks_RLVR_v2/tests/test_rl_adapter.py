import pytest
import numpy as np
import pandas as pd
from rl_discovery.adapter import ObservationAdapter, RLVRGymEnv


def test_observation_adapter_integrity():
    """Verifies that Pandas structures flatten into exact 35D float32 Tensors safely."""

    # Mock 12-column Strategy Ensemble (3 Tickers)
    strat_cols = [f"Strat_{i}" for i in range(12)]
    ensemble = pd.DataFrame(np.random.randn(3, 12), columns=strat_cols)
    ensemble.iloc[0, 0] = np.nan  # Inject NaN to test safety

    # Mock 11-column Macro DataFrame row
    macro_cols = [
        "Mkt_Ret",
        "Macro_Trend",
        "High_Yield_Spread",
        "Yield_Curve_10Y2Y",
        "High_Yield_Spread_Z",
        "Yield_Curve_10Y2Y_Z",
        "Macro_Trend_Vel",
        "Macro_Trend_Vel_Z",
        "Macro_Trend_Mom",
        "Macro_Vix_Z",
        "Macro_Vix_Ratio",
    ]
    macro_row = pd.Series(np.random.randn(11), index=macro_cols)

    # Process
    obs = ObservationAdapter.process(ensemble, macro_row, expected_strats=12)

    # Assertions
    assert obs.shape == (35,), f"Shape Mismatch: Expected (35,), got {obs.shape}"
    assert obs.dtype == np.float32, f"Type Mismatch: Expected float32, got {obs.dtype}"
    assert not np.isnan(
        obs
    ).any(), "Adapter leaked a NaN into the neural network input!"


class MockDiscoveryEnv:
    """Stubs out the complex AlphaLogic environment for wrapper testing."""

    def __init__(self):
        # Dynamic Space Detection requires a .cube attribute with 12 features
        self.cube = pd.DataFrame(np.zeros((1, 12)))

    def reset(self):
        return {
            "date": pd.Timestamp("2024-01-01"),
            "ensemble": pd.DataFrame(np.random.randn(2, 12)),
        }

    def step(self, action):
        return (
            {
                "date": pd.Timestamp("2024-01-02"),
                "ensemble": pd.DataFrame(np.random.randn(2, 12)),
            },
            0.05,
            False,
            {},
        )


def test_gym_wrapper_compliance():
    """Verifies the Env complies with Gymnasium specs and handles spaces correctly."""
    mock_macro = pd.DataFrame(
        np.random.randn(2, 11), index=pd.to_datetime(["2024-01-01", "2024-01-02"])
    )

    env = RLVRGymEnv(MockDiscoveryEnv(), mock_macro)

    obs, info = env.reset()
    assert env.observation_space.contains(
        obs
    ), "Reset obs does not fit Observation Space"

    # Generate random valid action
    action = env.action_space.sample()

    next_obs, reward, terminated, truncated, step_info = env.step(action)
    assert env.observation_space.contains(
        next_obs
    ), "Step obs does not fit Observation Space"
    assert isinstance(reward, float), "Reward must be a float"
