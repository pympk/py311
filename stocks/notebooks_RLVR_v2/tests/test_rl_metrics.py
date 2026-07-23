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

    # 1. Feature Cube
    idx = pd.MultiIndex.from_product([tickers, dates], names=["Ticker", "Date"])
    cube = pd.DataFrame(np.random.randn(len(idx), 11), index=idx)

    # 2. Reward Matrix (Flat 1% return for every stock)
    reward_matrix = pd.DataFrame(0.01, index=dates, columns=tickers)

    # 3. Macro Market Trend (Flat 0.5% return for SPY benchmark)
    macro_df = pd.DataFrame({"Mkt_Ret": [0.005] * 10}, index=dates)

    return cube, reward_matrix, macro_df, dates


def test_selection_logic_zero_width(dummy_environment_data):
    """Ensures Portfolio Constraints allow the agent to buy 0 stocks."""
    cube, _, _, dates = dummy_environment_data
    ensemble = cube.xs(dates[0], level="Date")
    config = TradingConfig(rank_max_width=10)

    # Simulate agent requesting the absolute minimum bounds (all -1.0)
    action = np.full(13, -1.0)
    selected, _, _, width, _, _ = SelectionLogic.apply_action(
        ensemble, action, rank_max_offset=500, rank_max_width=config.rank_max_width
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
    action = np.full(13, 1.0)

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


def test_environment_downside_penalty_and_cash(dummy_environment_data):
    """Proves Downside Penalty triggers properly & Cash retreat avoids slippage."""
    cube, reward_matrix, macro_df, dates = dummy_environment_data
    config = TradingConfig(slippage_rate=0.0010, downside_penalty=2.0, holding_period=1)

    env = DiscoveryEnv(cube, reward_matrix, dates, macro_df, config)
    env.reset()

    # Agent detects a crash and forces width to 0
    action = np.full(13, -1.0)
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
