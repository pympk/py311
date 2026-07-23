import pytest
import pandas as pd
import numpy as np
from core.settings import TradingConfig
from strategy.registry import get_strategy_registry
from data_pipeline.builder import MicroFeaturePipeline, MacroFeaturePipeline


@pytest.fixture
def fake_ohlcv_data():
    """Generates fake multi-index OHLCV data to test pipelines."""
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    tickers = ["AAPL", "SPY"]

    idx = pd.MultiIndex.from_product([tickers, dates], names=["Ticker", "Date"])

    # Generate prices that artificially trend upward then mean-revert
    prices = np.linspace(100, 150, 100).tolist() * 2
    volumes = np.random.randint(1_000_000, 5_000_000, 200).tolist()

    df = pd.DataFrame(
        {
            "Adj Close": prices,
            "Adj High": np.array(prices) * 1.01,
            "Adj Low": np.array(prices) * 0.99,
            "Volume": volumes,
        },
        index=idx,
    )
    return df


def test_rsi_bomb_scaling(fake_ohlcv_data):
    """Proves RSI is strictly bounded between -1.0 and 1.0"""
    config = TradingConfig()
    macro_df = pd.DataFrame(
        {"Mkt_Ret": [0.0] * 100},
        index=fake_ohlcv_data.index.get_level_values("Date").unique(),
    )

    micro_df = MicroFeaturePipeline.process(fake_ohlcv_data, macro_df, config)

    max_rsi = micro_df["RSI"].max()
    min_rsi = micro_df["RSI"].min()

    assert max_rsi <= 1.0, f"RSI exploded above 1.0: {max_rsi}"
    assert min_rsi >= -1.0, f"RSI exploded below -1.0: {min_rsi}"


def test_macro_dimension_strictness(fake_ohlcv_data):
    """Proves the Macro Pipeline drops raw noise and returns EXACTLY 11 columns."""
    config = TradingConfig(benchmark_ticker="SPY")

    macro_df = MacroFeaturePipeline.process(
        fake_ohlcv_data, df_indices=None, df_fed=None, config=config
    )

    assert (
        macro_df.shape[1] == 11
    ), f"Macro dimensions shattered! Expected 11, got {macro_df.shape[1]}"
    assert (
        "High_Yield_Spread" not in macro_df.columns
    ), "Raw, unscaled macro noise leaked into features!"


def test_slope_dead_node_fix():
    """Proves the new Slope blueprints no longer force cross-sectional means to exactly 0.0"""
    config = TradingConfig()
    registry = get_strategy_registry(config)

    slope_p_blueprint = registry["Slope_P_5_Z"]
    slope_v_blueprint = registry["Slope_V_5_Z"]

    # Simulate a fake observation cross-section for a single day
    fake_obs = pd.DataFrame(
        {
            "slope_p_5_z": [2.0, 1.5, -1.0, 0.0],  # Stock prices are booming
            "slope_v_5_z": [
                -1.0,
                -2.0,
                -0.5,
                0.0,
            ],  # But volume is completely drying up
        }
    )

    p_scores = slope_p_blueprint.formula(fake_obs)
    v_scores = slope_v_blueprint.formula(fake_obs)

    p_mean = p_scores.mean()
    v_mean = v_scores.mean()

    # In the old registry, the cross-sectional mean of ANY distribution would perfectly
    # evaluate to 0.0 every single day due to on-the-fly z-scoring.
    # Now, because we pre-scale temporally, a day of heavy divergence will actually
    # register a non-zero mean, allowing the Neural Network to detect market-wide exhaustion!
    assert not np.isclose(
        p_mean, 0.0
    ), "The Slope_P_5_Z blueprint forced cross-sectional mean to 0.0!"
    assert not np.isclose(
        v_mean, 0.0
    ), "The Slope_V_5_Z blueprint forced cross-sectional mean to 0.0!"

    assert p_mean > 0, "Price slope mean failed. Should be positive."
    assert v_mean < 0, "Volume slope mean failed. Should be negative."
