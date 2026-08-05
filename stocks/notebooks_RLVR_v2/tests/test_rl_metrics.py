import pytest
import pandas as pd
import numpy as np
from core.settings import TradingConfig
from core.logic import SelectionLogic
from rl_discovery.environment import DiscoveryEnv


@pytest.fixture
def dummy_environment_data():
    """Generates fake cross-sectional data, rewards, and macro indices."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]

    # 1. Feature Cube (Updated to 12 features)
    idx = pd.MultiIndex.from_product([tickers, dates], names=["Ticker", "Date"])
    cube = pd.DataFrame(np.random.randn(len(idx), 12), index=idx)

    # 2. Reward Matrix (Flat 1% return for every stock)
    reward_matrix = pd.DataFrame(0.01, index=dates, columns=tickers)

    # NEW: The environment now pulls the benchmark return directly from the reward matrix.
    reward_matrix["SPY"] = 0.005

    # 3. Macro Market Trend (Flat 0.5% return for SPY benchmark)
    macro_df = pd.DataFrame({"Mkt_Ret": [0.005] * 10}, index=dates)

    return cube, reward_matrix, macro_df, dates


def test_selection_logic_zero_width(dummy_environment_data):
    """Ensures Portfolio Constraints allow the agent to buy 0 stocks."""
    cube, _, _, dates = dummy_environment_data
    ensemble = cube.xs(dates[0], level="Date")
    config = TradingConfig(rank_max_width=10)

    # Simulate agent requesting the absolute minimum bounds (all -1.0)
    # ---> FIXED: Dynamically calculate action size based on the dummy cube
    num_features = ensemble.shape[1]
    action_size = num_features + 2
    action = np.full(action_size, -1.0)

    selected, _, _, width, _, _ = SelectionLogic.apply_action(
        ensemble,
        action,
        rank_max_offset_percentile=1.0,
        rank_max_width=config.rank_max_width,
    )

    # ---> FIXED: Update the parameter name to the new percentile logic
    selected, _, _, width, _, _ = SelectionLogic.apply_action(
        ensemble,
        action,
        rank_max_offset_percentile=1.0,
        rank_max_width=config.rank_max_width,
    )

    assert width == 0
    assert len(selected) == 0


def test_environment_alpha_and_slippage(dummy_environment_data):
    """Proves Slippage deduction and Positive Alpha computation are strictly correct."""
    cube, reward_matrix, macro_df, dates = dummy_environment_data
    config = TradingConfig(slippage_rate=0.0010, downside_penalty=2.0, holding_period=1)

    env = DiscoveryEnv(cube, reward_matrix, dates, macro_df, config)
    env.reset()

    # Action that aggressively asks for the max width (+1.0 bound for width)
    # Dynamically match the cube's feature count + 2 rank params
    num_features = cube.shape[1]
    action = np.full(num_features + 2, 1.0)

    # FIX: Force the offset to 0 so we don't skip all 5 stocks in our dummy universe!
    action[-2] = -1.0
    _, reward, _, info = env.step(action)

    assert len(info["tickers"]) == 5  # Bought all 5 available tickers
    assert info["slippage_applied"] == 0.0010

    # EXPECTED MATH:
    # reward_matrix = 0.01 per stock -> agent raw_sleeve_return = 0.01
    # slippage = 0.0010 -> actual_return = 0.0090
    # benchmark Mkt_Ret = 0.005 -> alpha = (0.0090 - 0.005) = +0.0040
    # Because Alpha > 0, penalized_alpha = 0.0040

    assert np.isclose(info["actual_return"], 0.0090)
    assert np.isclose(info["alpha"], 0.0040)
    assert np.isclose(info["penalized_alpha"], 0.0040)
    assert np.isclose(reward, 0.0040)

    # Verifying out-performance curves update correctly
    assert env.equity_curve[-1] == 1.0 + 0.0090
    assert env.alpha_equity_curve[-1] == 1.0 + 0.0040


def test_selection_logic_dynamic_boundaries():
    """
    Tests that continuous actions [-1, 1] correctly interpolate into
    offsets and widths based on rank_max_offset_percentile and rank_max_width.
    """
    # Create a dummy universe of exactly 200 tickers with 2 features
    tickers = [f"TICKER_{i}" for i in range(200)]
    ensemble = pd.DataFrame(
        np.random.randn(200, 2), index=tickers, columns=["Feature_1", "Feature_2"]
    )

    num_features = ensemble.shape[1]

    # Custom bounds for testing
    test_percentile = 0.5  # Max offset should be 50% of 200 = 100
    test_width = 20  # Max width should be 20

    # CASE 1: Absolute Minimum Action [-1.0]
    # Agent wants the lowest possible offset (0) and width (0)
    action_min = np.full(num_features + 2, -1.0)
    _, _, offset, width, _, _ = SelectionLogic.apply_action(
        ensemble, action_min, test_percentile, test_width
    )
    assert offset == 0, "Min action should map to 0 offset"
    assert width == 0, "Min action should map to 0 width"

    # CASE 2: Absolute Maximum Action [+1.0]
    # Agent wants the highest possible offset (100) and width (20)
    action_max = np.full(num_features + 2, 1.0)
    _, _, offset, width, _, _ = SelectionLogic.apply_action(
        ensemble, action_max, test_percentile, test_width
    )
    assert offset == 100, f"Max action should map to {int(200 * 0.5)} offset"
    assert width == 20, "Max action should map to max width"

    # CASE 3: Neutral Action [0.0]
    # Agent wants exactly the middle offset (50) and middle width (10)
    action_mid = np.full(num_features + 2, 0.0)
    _, _, offset, width, _, _ = SelectionLogic.apply_action(
        ensemble, action_mid, test_percentile, test_width
    )
    assert (
        offset == 50
    ), "Neutral action should map to exactly 50% of max allowed offset"
    assert width == 10, "Neutral action should map to exactly 50% of max allowed width"


def test_environment_downside_penalty_and_cash(dummy_environment_data):
    """Proves Downside Penalty triggers properly & Cash retreat avoids slippage."""
    cube, reward_matrix, macro_df, dates = dummy_environment_data
    config = TradingConfig(slippage_rate=0.0010, downside_penalty=2.0, holding_period=1)

    env = DiscoveryEnv(cube, reward_matrix, dates, macro_df, config)
    env.reset()

    # Agent detects a crash and forces width to 0
    # Dynamically match the cube's feature count + 2 rank params (12 + 2 = 14)
    num_features = cube.shape[1]
    action = np.full(num_features + 2, -1.0)

    _, reward, _, info = env.step(action)

    assert len(info["tickers"]) == 0
    assert info["slippage_applied"] == 0.0  # Kept hands in pockets, no fee

    # EXPECTED MATH:
    # 0 stocks bought -> actual_return = 0.0
    # benchmark Mkt_Ret = +0.005 -> alpha = (0.0 - 0.005) = -0.005 (Underperformance vs holding SPY)
    # Because Alpha < 0, penalized_alpha = -0.005 * 2.0 = -0.010

    assert np.isclose(info["actual_return"], 0.0)
    assert np.isclose(info["alpha"], -0.005)
    assert np.isclose(info["penalized_alpha"], -0.010)
    assert np.isclose(reward, -0.010)  # Verify agent receives the penalty

    # The tracker shouldn't log penalty, only absolute tracking
    assert env.equity_curve[-1] == 1.0
    assert env.alpha_equity_curve[-1] == 1.0 + (-0.005)


#
