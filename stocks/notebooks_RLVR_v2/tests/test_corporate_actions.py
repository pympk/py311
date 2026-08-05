import pytest
import pandas as pd
import numpy as np

from core.settings import TradingConfig
from data_pipeline.builder import QualityFilterPipeline
from data_pipeline.screener import UniverseScreener
from walk_forward.engine import AlphaEngine

from core.logic import AlphaLogic
from core.quant import QuantUtils

# Test pipeline safely handles delistings, mergers, and buyouts without poisoning the RL agent!


@pytest.fixture
def mock_config():
    # Setup config matching our intended behavior
    config = TradingConfig()
    config.handle_zeros_as_nan = True
    config.max_data_gap_ffill = 1
    config.nan_price_replacement = 0.0

    # Shrink windows for easier testing
    config.quality_window = 10
    config.quality_min_periods = 1
    return config


@pytest.fixture
def bld_mock_ohlcv():
    """
    Creates a mock OHLCV dataset simulating the BLD merger scenario.
    Days 1-3: Normal trading (~$350).
    Days 4-6: Delisted/Merged ($0.00 price, 0 volume).
    """
    dates = pd.date_range(start="2026-06-25", periods=6, freq="B")

    # BLD_MOCK dies after day 3
    bld_close = [350.0, 354.0, 354.53, 0.0, 0.0, 0.0]
    bld_vol = [1e6, 1.5e6, 3.1e6, 0.0, 0.0, 0.0]

    # HEALTHY survives
    hlt_close = [100.0, 101.0, 102.0, 101.5, 103.0, 104.0]
    hlt_vol = [5e5, 5e5, 6e5, 5.5e5, 6e5, 6e5]

    df = pd.DataFrame(
        {
            "Ticker": ["BLD_MOCK"] * 6 + ["HEALTHY"] * 6,
            "Date": list(dates) + list(dates),
            "Adj Close": bld_close + hlt_close,
            "Adj High": [c * 1.01 for c in bld_close]
            + [
                c * 1.01 for c in hlt_close
            ],  # Different High/Low/Close for Stale checking
            "Adj Low": [c * 0.99 for c in bld_close] + [c * 0.99 for c in hlt_close],
            "Volume": bld_vol + hlt_vol,
        }
    )

    # Ensure MultiIndex matches production data structure
    df = df.set_index(["Ticker", "Date"]).sort_index()
    return df


def test_pipeline_detects_delisting(bld_mock_ohlcv, mock_config):
    """
    [GUARD] Verifies QualityFilterPipeline creates short-term kill-switches.
    """
    # Run the pipeline
    quality_df = QualityFilterPipeline.process(bld_mock_ohlcv, mock_config)

    bld_features = quality_df.xs("BLD_MOCK", level="Ticker")

    # Day 3 (2026-06-29): Should be perfectly healthy
    assert bld_features.iloc[2]["IsZeroPrice"] == 0
    assert bld_features.iloc[2]["RecentStaleDays"] == 0

    # Day 4 (2026-06-30): The day it goes to 0.00
    assert bld_features.iloc[3]["IsZeroPrice"] == 1
    assert bld_features.iloc[3]["RecentStaleDays"] == 1

    # Day 6 (2026-07-02): Stale for 3 days
    assert bld_features.iloc[5]["IsZeroPrice"] == 1
    assert bld_features.iloc[5]["RecentStaleDays"] == 3


def test_screener_rejects_delisted_ticker(bld_mock_ohlcv, mock_config):
    """
    [GUARD] Verifies UniverseScreener drops the dead stock immediately,
    even if its 252-day RollMedDollarVol is still huge.
    """
    # 1. Build Mock Features
    quality_df = QualityFilterPipeline.process(bld_mock_ohlcv, mock_config)

    # We will artificially inject a HIGH RollMedDollarVol to prove the
    # new kill-switches override the lagging volume indicator.
    quality_df["RollMedDollarVol"] = 10_000_000

    # Add dummy columns needed by Screener
    quality_df["RollingStalePct"] = 0.01  # Passes long-term stale check
    quality_df["RollingSameVolCount"] = 0

    df_close_wide = bld_mock_ohlcv["Adj Close"].unstack(level=0)
    screener = UniverseScreener(
        df_close=df_close_wide,  # Provide the wide prices for the new kill-switch
        features_df=quality_df,
        macro_df=pd.DataFrame(),
        trading_calendar=pd.DatetimeIndex([]),
        config=mock_config,
    )

    dates = bld_mock_ohlcv.index.get_level_values("Date").unique().sort_values()

    # Day 3: BLD is alive
    survivors_day_3 = screener.filter_universe(dates[2], mock_config.thresholds)
    assert "BLD_MOCK" in survivors_day_3, "Healthy stock should pass."
    assert "HEALTHY" in survivors_day_3

    # Day 4: BLD dies (Price = 0.00) -> IsZeroPrice triggers
    survivors_day_4 = screener.filter_universe(dates[3], mock_config.thresholds)
    assert (
        "BLD_MOCK" not in survivors_day_4
    ), "Failed to drop $0.00 stock via IsZeroPrice!"
    assert "HEALTHY" in survivors_day_4

    # Day 6: BLD still dead -> RecentStaleDays triggers
    survivors_day_6 = screener.filter_universe(dates[5], mock_config.thresholds)
    assert (
        "BLD_MOCK" not in survivors_day_6
    ), "Failed to drop dead stock via RecentStaleDays!"


def test_engine_ffills_delisted_prices(bld_mock_ohlcv, mock_config):
    """
    [GUARD] Verifies AlphaEngine forward-fills $0.00 prices infinitely so the RL Agent
    calculates a realistic 0% return instead of an artificial -100% crash penalty.
    """
    # Extract just the closing prices into a wide format (like the Engine expects)
    df_close_wide = bld_mock_ohlcv["Adj Close"].unstack(level=0)

    # Create a dummy engine just to test _prepare_data
    engine = AlphaEngine(
        df_ohlcv=pd.DataFrame(),
        features_df=pd.DataFrame(),
        macro_df=pd.DataFrame(),
        config=mock_config,
        df_close_wide=df_close_wide,
        df_atrp_wide=pd.DataFrame(),  # Dummy
        df_trp_wide=pd.DataFrame(),  # Dummy
    )

    # Inspect the cleaned price matrix inside the engine
    cleaned_prices = engine.df_close

    # Check Day 3 (last active day)
    assert np.isclose(float(cleaned_prices["BLD_MOCK"].iloc[2]), 354.53)

    # Check Day 6 (was originally 0.00 in the mock data)
    # It MUST be ffill'ed to 354.53, NOT 0.00!
    assert np.isclose(
        float(cleaned_prices["BLD_MOCK"].iloc[5]), 354.53
    ), "Engine failed to infinitely forward-fill the delisted price! RL will see -100% return."


def test_portfolio_math_with_bad_and_dead_tickers():
    """
    Tests that QuantUtils and AlphaLogic gracefully handle:
    1. Bad tickers with all NaN prices in date window ('GTLS')
    2. Dead/missing tickers not in market parquet ('BLD')

    Verifies that initial equity curve stays at exactly 1.00 and
    both reward engines produce identical returns.
    """
    # 5 business days: 09-21 (Thu), 09-22 (Fri), 09-25 (Mon), 09-26 (Tue), 09-27 (Wed)
    dates = pd.date_range("2023-09-21", "2023-09-27", freq="B")

    # 1. Construct test price matrix (5 elements per column)
    df_prices = pd.DataFrame(
        {
            "LH": [198.523, 198.000, 197.500, 197.000, 196.703],  # -0.9168%
            "SPTS": [25.552, 25.560, 25.570, 25.580, 25.588],  # +0.1397%
            "STLD": [95.461, 97.000, 99.000, 101.000, 102.015],  # +6.8656%
            "GTLS": [np.nan, np.nan, np.nan, np.nan, np.nan],  # Bad ticker
        },
        index=dates,
    )

    # User selected 5 tickers (1 bad, 1 dead)
    chosen_tickers = ["LH", "SPTS", "STLD", "GTLS", "BLD"]
    initial_weights = pd.Series(1.0 / len(chosen_tickers), index=chosen_tickers)

    # Empty dummy ATRP / TRP matrices for compute_portfolio_stats
    dummy_atrp = pd.DataFrame(0.01, index=dates, columns=df_prices.columns)
    dummy_trp = pd.DataFrame(0.01, index=dates, columns=df_prices.columns)

    # 2. Test QuantUtils.compute_portfolio_stats
    equity_curve, _, _, _ = QuantUtils.compute_portfolio_stats(
        df_prices, dummy_atrp, dummy_trp, initial_weights
    )

    # ASSERTION 1: Initial equity curve MUST start at exactly 1.00
    assert (
        abs(equity_curve.iloc[0] - 1.0) < 1e-6
    ), f"Initial equity curve failed! Expected 1.00, got {equity_curve.iloc[0]:.6f}"

    # Expected return of the 3 active tickers: (-0.009168 + 0.001397 + 0.068656) / 3 = 0.020295
    quant_return = float(equity_curve.iloc[-1]) - 1.0

    # 3. Test AlphaLogic.calculate_veritable_reward with reward_matrix containing NaNs
    reward_series = (df_prices.iloc[-1] / df_prices.iloc[0]) - 1.0
    reward_matrix = pd.DataFrame([reward_series], index=[dates[0]])

    log_reward = AlphaLogic.calculate_veritable_reward(
        reward_matrix, dates[0], chosen_tickers
    )
    oracle_return = np.expm1(log_reward)

    # ASSERTION 2: Auditor return and Oracle return must match within 1 basis point
    diff = abs(quant_return - oracle_return)
    assert diff < 1e-4, (
        f"Divergence detected between QuantUtils ({quant_return:.6f}) "
        f"and AlphaLogic ({oracle_return:.6f})! Diff: {diff:.6f}"
    )

    print(
        f"\n✅ PASSED: Quant Return: {quant_return:.6f} | Oracle Return: {oracle_return:.6f}"
    )
