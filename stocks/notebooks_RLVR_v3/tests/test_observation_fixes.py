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
    """Proves the Macro Pipeline drops raw noise and returns EXACTLY 10 columns."""
    config = TradingConfig(benchmark_ticker="SPY")

    macro_df = MacroFeaturePipeline.process(
        fake_ohlcv_data, df_indices=None, df_fed=None, config=config
    )

    assert (
        macro_df.shape[1] == 10
    ), f"Macro dimensions shattered! Expected 10, got {macro_df.shape[1]}"
    assert (
        "High_Yield_Spread" not in macro_df.columns
    ), "Raw, unscaled macro noise leaked into features!"
    assert (
        "High_Yield_Spread_Z" not in macro_df.columns
    ), "High_Yield_Spread_Z was not properly removed!"


def test_intermediate_factor_blueprints():
    """Proves the 3 intermediate replacement blueprints calculate correctly with valid polarity."""
    from types import SimpleNamespace

    config = TradingConfig()
    registry = get_strategy_registry(config)

    assert "Momentum (63d)" in registry
    assert "Momentum (126d)" in registry
    assert "Residual Low-Vol (63d)" in registry

    fake_obs = SimpleNamespace(
        mom_63=pd.Series([0.15, 0.05, -0.10], index=["AAPL", "MSFT", "GOOG"]),
        mom_126=pd.Series([0.30, 0.10, -0.20], index=["AAPL", "MSFT", "GOOG"]),
        ivol_63=pd.Series([0.012, 0.018, 0.035], index=["AAPL", "MSFT", "GOOG"]),
    )

    m63_scores = registry["Momentum (63d)"](fake_obs)
    m126_scores = registry["Momentum (126d)"](fake_obs)
    rivol_scores = registry["Residual Low-Vol (63d)"](fake_obs)

    assert len(m63_scores) == 3
    assert len(m126_scores) == 3
    assert len(rivol_scores) == 3

    # Polarity Tripwire: Lowest IVol (0.012 for AAPL) must yield HIGHEST residual low-vol score (-0.012 > -0.035)
    assert (
        rivol_scores["AAPL"] > rivol_scores["GOOG"]
    ), "Residual Low-Vol polarity inverted!"
    assert m63_scores["AAPL"] > m63_scores["GOOG"]
    assert m126_scores["AAPL"] > m126_scores["GOOG"]
