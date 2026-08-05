import pytest
import pickle
import numpy as np
import pandas as pd
import re

from core.paths import OUTPUT_DIR
from core.settings import TradingConfig
from strategy.registry import get_strategy_registry

# =====================================================================
# FIXTURES
# =====================================================================


@pytest.fixture(scope="module")
def trading_config():
    """Loads the core configuration used during the environment run."""
    return TradingConfig()


@pytest.fixture(scope="module")
def blotter_df():
    """Finds the most recent OOS RL run and loads the raw blotter dataframe."""
    if not OUTPUT_DIR.exists():
        pytest.skip("Output directory does not exist.")

    pkl_files = list(OUTPUT_DIR.glob("oos_results_*.pkl"))
    if not pkl_files:
        pytest.skip("No OOS results pickle files found.")

    # Get the latest file by modification time
    latest_pkl = max(pkl_files, key=lambda p: p.stat().st_mtime)

    # ---> NEW: Extract penalty from filename
    match = re.search(r"_pen_([0-9.]+)_", latest_pkl.name)
    penalty = float(match.group(1)) if match else 1.0

    print(
        f"\n🚀 [INFO] Loaded OOS File: {latest_pkl.name} (Extracted Penalty: {penalty})"
    )

    with open(latest_pkl, "rb") as f:
        results = pickle.load(f)

    df = pd.DataFrame(results["blotter"])

    # Ensure datetime types for temporal tests
    df["decision_date"] = pd.to_datetime(df["decision_date"])
    df["buy_date"] = pd.to_datetime(df["buy_date"])
    df["sell_date"] = pd.to_datetime(df["sell_date"])

    # ---> NEW: Attach the extracted penalty to the dataframe's metadata
    df.attrs["downside_penalty"] = penalty

    return df


# =====================================================================
# TESTS
# =====================================================================


def test_temporal_sanity(blotter_df):
    """
    Domain A: Proves there is no temporal leakage in the trading lifecycle.
    Decision happens first, execution next, and settlement (sell) last.
    """
    first_row = blotter_df.iloc[0]
    assert (
        first_row["decision_date"] < first_row["buy_date"]
    ), "First row: Execution before Decision!"
    assert first_row["buy_date"] < first_row["sell_date"], "First row: Sell before Buy!"

    last_row = blotter_df.iloc[-1]
    assert (
        last_row["decision_date"] < last_row["buy_date"]
    ), "Last row: Execution before Decision!"
    assert last_row["buy_date"] < last_row["sell_date"], "Last row: Sell before Buy!"


def test_action_dimensionality(blotter_df, trading_config):
    """
    Domain B: Proves that the anonymous ML float vector (`raw_actions`)
    has the mathematically correct number of dimensions to drive the engine.
    (Strategies + Rank Offset + Rank Width)
    """
    registry = get_strategy_registry(trading_config)
    expected_dims = len(registry) + 2  # Strategy weights + 2 sizing dimensions

    for idx in [0, -1]:
        row = blotter_df.iloc[idx]
        raw_vector = np.array(row["raw_actions"])

        assert len(raw_vector) == expected_dims, (
            f"Action vector dimensionality mismatch in row {idx}. "
            f"Expected {expected_dims}, got {len(raw_vector)}."
        )


def test_intra_row_arithmetic(blotter_df, trading_config):
    """
    Domain C: Recalculates the core Environment step() logic from scratch
    to prove returns, slippage, and penalties were applied correctly.
    """
    hp = trading_config.holding_period
    # ---> NEW: Use the penalty extracted from the filename
    downside_penalty = blotter_df.attrs.get(
        "downside_penalty", trading_config.downside_penalty
    )

    for idx in [0, -1]:
        row = blotter_df.iloc[idx]

        # 1. Expected Return Math
        expected_raw_return = np.exp(row["raw_log_reward"]) - 1.0

        # Slippage is only applied if tickers were actually selected
        expected_slippage = (
            trading_config.slippage_rate if len(row["chosen_tickers"]) > 0 else 0.0
        )
        expected_actual_return = expected_raw_return - expected_slippage

        assert np.isclose(
            expected_actual_return, row["actual_return"], atol=1e-6
        ), f"Return Math drift in row {idx}"

        # 2. Expected Alpha Math
        expected_alpha = expected_actual_return - row["mkt_return"]
        assert np.isclose(
            expected_alpha, row["alpha"], atol=1e-6
        ), f"Alpha Math drift in row {idx}"

        # 3. Expected Penalty Math
        expected_penalized_alpha = (
            expected_alpha * downside_penalty if expected_alpha < 0 else expected_alpha
        )
        assert np.isclose(
            expected_penalized_alpha, row["penalized_alpha"], atol=1e-6
        ), f"Penalty Math drift in row {idx}"

        # 4. Expected Impact Math
        expected_portfolio_impact = expected_actual_return / hp
        assert np.isclose(
            expected_portfolio_impact, row["portfolio_impact"], atol=1e-6
        ), f"Fractional Impact Math drift in row {idx}"


def test_first_row_state_initialization(blotter_df):
    """
    Domain D: Verifies the compounding engine starts clean on step 1.
    """
    first_row = blotter_df.iloc[0]

    # Equity should exactly equal 1.0 * (1 + impact)
    expected_initial_equity = 1.0 * (1.0 + first_row["portfolio_impact"])

    assert np.isclose(
        first_row["agent_equity"], expected_initial_equity, atol=1e-6
    ), f"First row equity initialization is corrupted. Expected {expected_initial_equity}, got {first_row['agent_equity']}"


def test_cumulative_equity_provenance(blotter_df):
    """
    Domain E: The Ultimate Provenance Test.
    Manually reconstructs the entire equity curve using Pandas cumprod()
    on the recorded fractional impacts, proving the final recorded portfolio
    balance is mathematically authentic.
    """
    # Recalculate full cumulative equity curve strictly from fractional impacts
    manual_equity_curve = (1.0 + blotter_df["portfolio_impact"]).cumprod()

    # Get the system's final reported numbers
    final_recorded_equity = blotter_df["agent_equity"].iloc[-1]
    final_manual_equity = manual_equity_curve.iloc[-1]

    # Max acceptable float drift after ~1000 multiplicative operations is slightly larger (1e-4)
    assert np.isclose(
        final_manual_equity, final_recorded_equity, atol=1e-4
    ), f"Phantom Compounding Detected! Manual Final: {final_manual_equity}, Recorded Final: {final_recorded_equity}"
