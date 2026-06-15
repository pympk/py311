import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from core.settings import TradingConfig
from data_pipeline.screener import UniverseScreener
from data_pipeline.cache import AlphaCache
from core.contracts import MetricBlueprint


@pytest.fixture
def mock_config():
    """
    Constructs a minimal TradingConfig with dummy thresholds so our
    mock tickers pass the UniverseScreener.
    """
    config = MagicMock(spec=TradingConfig)

    # Configure mock thresholds to ensure our mock tickers always pass gating
    thresholds = MagicMock()
    thresholds.min_median_dollar_volume = 100000.0
    thresholds.min_liquidity_percentile = None
    thresholds.max_stale_pct = 0.05
    thresholds.max_same_vol_count = 5
    config.thresholds = thresholds

    return config


@pytest.fixture
def test_data():
    """
    Creates a minimal, deterministic set of DataFrames mimicking the database schemas.
    """
    # 10 business days for testing lookbacks
    dates = pd.date_range("2026-01-01", periods=10, freq="B")
    tickers = ["AAA", "BBB"]

    # 1. df_close: Wide-format (Dates x Tickers)
    df_close = pd.DataFrame(100.0, index=dates, columns=tickers)
    # Give 'AAA' an upward trend and 'BBB' a downward trend to check feature changes
    df_close.loc[dates[1], "AAA"] = 101.0
    df_close.loc[dates[2], "AAA"] = 102.0
    df_close.loc[dates[1], "BBB"] = 99.0
    df_close.loc[dates[2], "BBB"] = 98.0

    # 2. features_df: MultiIndex (Ticker, Date) structure expected by screener
    idx = pd.MultiIndex.from_product([tickers, dates], names=["Ticker", "Date"])
    features_df = pd.DataFrame(index=idx)

    # Standard columns required by UniverseScreener.filter_universe
    features_df["RollMedDollarVol"] = 500000.0
    features_df["RollingStalePct"] = 0.01
    features_df["RollingSameVolCount"] = 0.0

    # Columns required by UniverseScreener.build_observation
    features_df["ATRP"] = 0.02
    features_df["TRP"] = 0.01
    features_df["ATR"] = 1.5
    features_df["RSI"] = 50.0
    features_df["Consistency"] = 0.8
    features_df["Mom_21"] = 1.2
    features_df["IR_63"] = 0.5
    features_df["Beta_63"] = 1.0
    features_df["DD_21"] = -0.05
    features_df["AutoCorr_15"] = 0.1
    features_df["Range_Pos_20"] = 0.6
    features_df["Slope_P_5"] = 0.05
    features_df["Slope_V_5"] = 0.02
    features_df["Convexity"] = 0.01

    # Insert distinctive values on a targeted date to trace outputs directly
    features_df.loc[("AAA", dates[2]), "Mom_21"] = 1.5
    features_df.loc[("BBB", dates[2]), "Mom_21"] = -0.5
    features_df.loc[("AAA", dates[2]), "RSI"] = 70.0
    features_df.loc[("BBB", dates[2]), "RSI"] = 30.0

    # 3. macro_df: Index (Date)
    macro_df = pd.DataFrame(index=dates)
    macro_df["Macro_Trend"] = 1.0
    macro_df["Macro_Trend_Vel"] = 0.1
    macro_df["Macro_Vix_Z"] = -0.5
    macro_df["Macro_Vix_Ratio"] = 0.95

    # 4. trading_calendar
    trading_calendar = pd.DatetimeIndex(dates)

    return {
        "df_close": df_close,
        "features_df": features_df,
        "macro_df": macro_df,
        "trading_calendar": trading_calendar,
    }


@pytest.fixture
def screener(test_data, mock_config):
    """
    Constructs a clean screener with dummy inputs.
    """
    return UniverseScreener(
        df_close=test_data["df_close"],
        features_df=test_data["features_df"],
        macro_df=test_data["macro_df"],
        trading_calendar=test_data["trading_calendar"],
        config=mock_config,
    )


@pytest.fixture
def mock_strategy_registry():
    """
    Defines a simple, deterministic set of MetricBlueprint objects
    acting as the patched registry.
    """
    blueprint_rsi = MetricBlueprint(
        name="Mock RSI",
        category="Test",
        regime="Test",
        description="Extract RSI directly from observation",
        agent_hint="Test hint",
        intervention_trigger=f"Test intervention trigger",
        formula=lambda obs: obs.rsi,
    )

    blueprint_mom = MetricBlueprint(
        name="Mock Momentum",
        category="Test",
        regime="Test",
        description="Extract Mom_21 directly from observation",
        agent_hint="Test hint",
        intervention_trigger=f"Test intervention trigger",
        formula=lambda obs: obs.mom_21,
    )

    return {
        "Mock RSI": blueprint_rsi,
        "Mock Momentum": blueprint_mom,
    }


def test_compute_alpha_ensemble(
    screener, mock_config, mock_strategy_registry, test_data
):
    """
    Verifies that compute_alpha_ensemble extracts indices, structures columns,
    and returns correct outputs for a given lookback period.
    """
    cache = AlphaCache(screener=screener, config=mock_config, lookbacks=[2])

    # Select the third trading day ('2026-01-05') to allow a lookback of 2 days
    target_date = test_data["trading_calendar"][2]

    # Patch get_strategy_registry in the cache module namespace to use our deterministic dummy blueprints
    with patch(
        "data_pipeline.cache.get_strategy_registry", return_value=mock_strategy_registry
    ):
        ensemble = cache.compute_alpha_ensemble(target_date, lookback_periods=[2])

    assert not ensemble.empty, "Ensemble calculation returned an empty DataFrame"

    # Validate columns are constructed precisely in "{lookback}d_{strategy_name}" format
    expected_cols = ["2d_Mock RSI", "2d_Mock Momentum"]
    assert list(ensemble.columns) == expected_cols

    # Validate rows match our screened candidates index
    assert list(ensemble.index) == ["AAA", "BBB"]

    # Validate specific values to ensure they match our pre-calculated mock dataset on that target date
    assert ensemble.loc["AAA", "2d_Mock RSI"] == 70.0
    assert ensemble.loc["AAA", "2d_Mock Momentum"] == 1.5
    assert ensemble.loc["BBB", "2d_Mock RSI"] == 30.0
    assert ensemble.loc["BBB", "2d_Mock Momentum"] == -0.5


def test_alpha_cache_build_and_vision(
    screener, mock_config, mock_strategy_registry, test_data
):
    """
    Verifies that the complete 'build' method parses the calendar, creates the multi-indexed
    feature cube structure, and retrieves a targeted date slice cleanly using get_vision.
    """
    cache = AlphaCache(screener=screener, config=mock_config, lookbacks=[2])

    # Start building from index 2 onwards
    start_date_str = str(test_data["trading_calendar"][2].date())

    with patch(
        "data_pipeline.cache.get_strategy_registry", return_value=mock_strategy_registry
    ):
        cache.build(start_date=start_date_str)

    # Verify the final cube properties
    assert not cache.feature_cube.empty, "Built feature cube should not be empty"
    assert cache.feature_cube.index.names == [
        "Date",
        "Ticker",
    ], "Feature cube must use ['Date', 'Ticker'] multi-index"

    # Verify slice extraction using get_vision
    target_date = test_data["trading_calendar"][2]
    vision_df = cache.get_vision(target_date)

    assert (
        not vision_df.empty
    ), "get_vision should return valid data for calculated dates"
    assert list(vision_df.index) == ["AAA", "BBB"]
    assert "2d_Mock RSI" in vision_df.columns
    assert vision_df.loc["AAA", "2d_Mock RSI"] == 70.0


def test_alpha_cache_out_of_bounds_lookback(
    screener, mock_config, mock_strategy_registry, test_data
):
    """
    Verifies that when lookback calculations result in a negative index (offset out of bounds),
    the catch block gracefully handles it and continues rather than raising unhandled exceptions.
    """
    # Lookback is larger than the entire available history prior to the first date
    cache = AlphaCache(screener=screener, config=mock_config, lookbacks=[5])

    # Select the very first date in the calendar
    target_date = test_data["trading_calendar"][0]

    with patch(
        "data_pipeline.cache.get_strategy_registry", return_value=mock_strategy_registry
    ):
        ensemble = cache.compute_alpha_ensemble(target_date, lookback_periods=[5])

    # Should skip lookbacks that hit array boundary errors, yielding an empty dataframe
    assert ensemble.empty


def test_alpha_cache_empty_universe(
    screener, mock_config, mock_strategy_registry, test_data
):
    """
    Verifies that if the screener returns zero candidates, compute_alpha_ensemble returns
    an empty DataFrame early instead of processing further logic.
    """
    cache = AlphaCache(screener=screener, config=mock_config, lookbacks=[2])
    target_date = test_data["trading_calendar"][2]

    # Patch screener logic to return an empty array of candidates
    with patch.object(screener, "filter_universe", return_value=[]):
        ensemble = cache.compute_alpha_ensemble(target_date, lookback_periods=[2])

    assert ensemble.empty


def test_alpha_cache_get_vision_missing_date(screener, mock_config):
    """
    Verifies that get_vision returns an empty DataFrame if queried with a date
    not contained inside the cache index, avoiding uncaught KeyErrors.
    """
    cache = AlphaCache(screener=screener, config=mock_config, lookbacks=[2])

    missing_date = pd.Timestamp("2026-12-31")
    vision_df = cache.get_vision(missing_date)

    assert vision_df.empty
