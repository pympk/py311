import pytest
import pandas as pd
import numpy as np
from types import SimpleNamespace
from core.settings import TradingConfig
from strategy.registry import get_strategy_registry
from data_pipeline.builder import MicroFeaturePipeline, MacroFeaturePipeline


@pytest.fixture
def fake_ohlcv_data():
    """Generates fake multi-index OHLCV data to test pipelines."""
    config = TradingConfig()
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    tickers = ["AAPL", config.benchmark_ticker]

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
    """Proves the Macro Pipeline drops raw noise and returns EXACTLY 10 stationary columns (AI_CONTEXT invariant: 46 dims)."""
    config = TradingConfig()

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

    expected_10_cols = [
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
    ]
    for col in expected_10_cols:
        assert (
            col in macro_df.columns
        ), f"Missing required stationary macro feature: {col}"
        assert not macro_df[col].isna().any(), f"Macro feature {col} contains NaNs!"


def test_stationary_macro_feature_bounds(fake_ohlcv_data):
    """Verifies that all 10 stationary macro features are bounded and finite."""
    config = TradingConfig()
    macro_df = MacroFeaturePipeline.process(
        fake_ohlcv_data, df_indices=None, df_fed=None, config=config
    )

    # Z-scores must be strictly bounded by feature_zscore_clip
    z_score_cols = [
        "Mkt_Ret_Z",
        "Macro_Trend_Z",
        "Yield_Curve_10Y2Y_Z",
        "Macro_Trend_Vel_Z",
        "Macro_Vix_Z",
        "Mkt_Vol_63d_Z",
    ]
    for col in z_score_cols:
        assert (
            macro_df[col].abs().max() <= config.feature_zscore_clip + 1e-6
        ), f"{col} exceeded z-score clip bound!"

    # Ratio feature must be bounded by feature_ratio_clip
    assert (
        macro_df["Macro_Vix_Ratio"].abs().max() <= config.feature_ratio_clip + 1e-6
    ), "Macro_Vix_Ratio exceeded clip bound!"


def test_intermediate_factor_blueprints():
    """Proves the Gen 16 orthogonal replacement factor blueprints calculate correctly with valid polarity."""
    config = TradingConfig()
    registry = get_strategy_registry(config)

    # The 4 Gen 16 orthogonal replacements + 2 intermediate trend anchors
    assert "Efficiency Ratio (ER_63)" in registry
    assert "Momentum (126d)" in registry
    assert "Residual Low-Vol (63d)" in registry
    assert "Residual Momentum (126d)" in registry
    assert "Range Position (52w High)" in registry
    assert "Downside Beta (-Beta_Down_63)" in registry

    fake_obs = SimpleNamespace(
        er_63=pd.Series([0.85, 0.45, 0.15], index=["AAPL", "MSFT", "GOOG"]),
        mom_126=pd.Series([0.30, 0.10, -0.20], index=["AAPL", "MSFT", "GOOG"]),
        ivol_63=pd.Series([0.012, 0.018, 0.035], index=["AAPL", "MSFT", "GOOG"]),
        res_mom_126=pd.Series([1.80, 0.20, -1.20], index=["AAPL", "MSFT", "GOOG"]),
        range_pos_52w=pd.Series([0.98, 0.85, 0.65], index=["AAPL", "MSFT", "GOOG"]),
        beta_down_63=pd.Series([0.75, 1.05, 1.45], index=["AAPL", "MSFT", "GOOG"]),
    )

    er_scores = registry["Efficiency Ratio (ER_63)"](fake_obs)
    m126_scores = registry["Momentum (126d)"](fake_obs)
    rivol_scores = registry["Residual Low-Vol (63d)"](fake_obs)
    res_mom_scores = registry["Residual Momentum (126d)"](fake_obs)
    range_scores = registry["Range Position (52w High)"](fake_obs)
    beta_down_scores = registry["Downside Beta (-Beta_Down_63)"](fake_obs)

    for scores in [
        er_scores,
        m126_scores,
        rivol_scores,
        res_mom_scores,
        range_scores,
        beta_down_scores,
    ]:
        assert len(scores) == 3

    # Polarity Tripwires:
    # 1. Residual Low-Vol: Lowest IVol (0.012 for AAPL) must yield HIGHEST score (-0.012 > -0.035)
    assert (
        rivol_scores["AAPL"] > rivol_scores["GOOG"]
    ), "Residual Low-Vol polarity inverted!"
    # 2. Downside Beta: Lowest crash beta (0.75 for AAPL) must yield HIGHEST score (-0.75 > -1.45)
    assert (
        beta_down_scores["AAPL"] > beta_down_scores["GOOG"]
    ), "Downside Beta polarity inverted!"
    # 3. Efficiency Ratio: Monotonic positive trend
    assert er_scores["AAPL"] > er_scores["GOOG"]
    # 4. Momentum 126d: Positive trend
    assert m126_scores["AAPL"] > m126_scores["GOOG"]
    # 5. Residual Momentum 126d: Positive trend
    assert res_mom_scores["AAPL"] > res_mom_scores["GOOG"]
    # 6. Range Position: Nearest 52w high
    assert range_scores["AAPL"] > range_scores["GOOG"]
