"""
Accounting Provenance & Arithmetic Audit Test Suite.
Verifies temporal integrity, action dimensionality, intra-row return & slippage math,
equity compounding, and geometric alpha wealth ratios against the authoritative
constituent baseline anchors in output/canonical_anchors/.
"""

import numpy as np
import pandas as pd
import pytest

from core.paths import OUTPUT_DIR
from core.settings import TradingConfig
from strategy.registry import get_strategy_registry

# =====================================================================
# FIXTURES
# =====================================================================


@pytest.fixture(scope="module")
def blotter_df():
    """
    Loads the authoritative ground-truth constituent blotter from canonical_anchors.
    Strictly audits single-seed constituent blotters (e.g., s42_gen18) where individual
    MDP trajectory identities (gross return, slippage, net return, non-linear penalized
    alpha) are preserved at floating-point precision. Multi-seed blends are excluded
    because non-linear reward operators cannot be verified on portfolio averages.
    """
    candidate_dirs = [
        OUTPUT_DIR / "canonical_anchors",
        OUTPUT_DIR,
    ]

    valid_files = []
    for d in candidate_dirs:
        if not d.exists():
            continue
        # Scan strictly for constituent single-seed continuous blotters
        for p in d.glob("blotter_continuous_s*.parquet"):
            valid_files.append((p.stat().st_mtime, p))

    if not valid_files:
        pytest.skip(
            "No constituent Parquet blotters found in canonical_anchors or output."
        )

    # Select the latest promoted constituent anchor (e.g., s42_gen18)
    valid_files.sort(key=lambda x: x[0], reverse=True)
    target_path = valid_files[0][1]

    df = pd.read_parquet(target_path)

    # Ensure temporal datetimes
    for col in ["date", "decision_date", "buy_date", "sell_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    cfg = TradingConfig()
    df.attrs["loss_aversion_penalty"] = float(
        getattr(cfg, "loss_aversion_penalty", 0.0)
    )
    df.attrs["upside_alpha_mult"] = float(getattr(cfg, "upside_alpha_mult", 1.0))
    df.attrs["artifact_path"] = str(target_path)
    return df


@pytest.fixture(scope="module")
def trading_config():
    """Loads the core dynamic configuration."""
    return TradingConfig()


# =====================================================================
# TESTS
# =====================================================================


def test_temporal_sanity(blotter_df):
    """Verifies strict Next-Day Close T+1 execution sequence: Decision < Buy < Sell."""
    first_row = blotter_df.iloc[0]
    assert (
        first_row["decision_date"] < first_row["buy_date"]
    ), "Row 0: Execution before Decision!"
    assert (
        first_row["buy_date"] < first_row["sell_date"]
    ), "Row 0: Liquidation before Purchase!"

    last_row = blotter_df.iloc[-1]
    assert (
        last_row["decision_date"] < last_row["buy_date"]
    ), "Terminal Row: Execution before Decision!"
    assert (
        last_row["buy_date"] < last_row["sell_date"]
    ), "Terminal Row: Liquidation before Purchase!"


def test_action_dimensionality(blotter_df, trading_config):
    """Verifies action tensor dimensions match K strategy features + 4 controls."""
    registry = get_strategy_registry(trading_config)
    expected_dim = len(registry) + 4

    for idx in [0, -1]:
        row = blotter_df.iloc[idx]
        raw_vector = np.array(row["raw_actions"])
        assert (
            len(raw_vector) == expected_dim
        ), f"Action dimensionality mismatch in row {idx}: expected {expected_dim}, got {len(raw_vector)}"


def test_intra_row_arithmetic(blotter_df, trading_config):
    """
    Recalculates the core MTMPortfolioEngine accounting identities from scratch
    to prove weights, gross returns, slippage drag, and penalized alpha reward math.
    """
    loss_aversion_penalty = blotter_df.attrs["loss_aversion_penalty"]
    upside_mult = blotter_df.attrs["upside_alpha_mult"]

    for idx in [0, -1]:
        row = blotter_df.iloc[idx]
        w_active = float(row["weight_active"])
        w_bm = float(row["weight_benchmark"])
        w_cash = float(row["weight_cash"])

        # 1. Weights Sum Invariant
        assert np.isclose(
            w_active + w_bm + w_cash, 1.0, atol=1e-5
        ), f"Row {idx} weights do not sum to 1.0: {w_active + w_bm + w_cash}"

        # 2. Gross Return Identity
        gross_stock = float(row["gross_stock_daily_simple_ret"])
        bm_ret = float(row["bm_daily_simple_ret"])
        cash_ret = float(row.get("cash_daily_simple_ret", 0.0))
        expected_gross = w_active * gross_stock + w_bm * bm_ret + w_cash * cash_ret
        assert np.isclose(
            expected_gross, float(row["gross_daily_simple_ret"]), atol=1e-6
        ), f"Row {idx} gross return drift: expected {expected_gross}, got {row['gross_daily_simple_ret']}"

        # 3. Net Return Slippage Identity
        expected_net = expected_gross - float(row["slippage_daily_simple_loss"])
        assert np.isclose(
            expected_net, float(row["net_daily_simple_ret"]), atol=1e-6
        ), f"Row {idx} net return drift: expected {expected_net}, got {row['net_daily_simple_ret']}"

        # 4. Raw Alpha Spread Identity
        expected_alpha = float(row["net_daily_simple_ret"]) - bm_ret
        assert np.isclose(
            expected_alpha, float(row["alpha_daily_simple_ret"]), atol=1e-6
        ), f"Row {idx} alpha drift: expected {expected_alpha}, got {row['alpha_daily_simple_ret']}"

        # 5. Penalized Alpha Reward Identity
        recorded_alpha = float(row["alpha_daily_simple_ret"])
        recorded_penalized = float(row["penalized_alpha_daily_simple_ret"])

        if recorded_alpha > 0.0:
            expected_penalized = recorded_alpha * upside_mult
        elif recorded_alpha < 0.0:
            expected_penalized = recorded_alpha * (
                1.0 + max(0.0, loss_aversion_penalty)
            )
        else:
            expected_penalized = 0.0

        assert np.isclose(
            expected_penalized, recorded_penalized, atol=1e-6
        ), f"Row {idx} penalized alpha drift: expected {expected_penalized}, got {recorded_penalized}"


def test_first_row_state_initialization(blotter_df):
    """Verifies that initial agent equity compounds strictly from $1.00 base capital."""
    first_row = blotter_df.iloc[0]
    expected_initial_equity = 1.0 * (1.0 + float(first_row["net_daily_simple_ret"]))
    assert np.isclose(
        first_row["agent_equity"], expected_initial_equity, atol=1e-6
    ), f"Initial equity drift: expected {expected_initial_equity}, got {first_row['agent_equity']}"


def test_cumulative_equity_provenance(blotter_df):
    """Verifies agent equity curve equals geometric cumprod without phantom returns."""
    manual_equity_curve = (1.0 + blotter_df["net_daily_simple_ret"]).cumprod()
    final_recorded_equity = float(blotter_df["agent_equity"].iloc[-1])
    final_manual_equity = float(manual_equity_curve.iloc[-1])
    assert np.isclose(
        final_manual_equity, final_recorded_equity, atol=1e-4
    ), f"Equity compounding discrepancy: manual {final_manual_equity} vs recorded {final_recorded_equity}"


def test_cumulative_alpha_equity_provenance(blotter_df):
    """
    INVARIANT: Alpha Equity must strictly equal the geometric wealth ratio:
    Alpha Multiplier(t) = Agent Equity(t) / Benchmark Equity(t).
    Additive compounding (1 + penalized_alpha) is strictly forbidden.
    """
    manual_agent = (1.0 + blotter_df["net_daily_simple_ret"]).cumprod()
    manual_bm = (1.0 + blotter_df["bm_daily_simple_ret"]).cumprod()
    expected_alpha_curve = manual_agent / np.maximum(manual_bm, 1e-8)

    final_recorded_alpha = float(blotter_df["alpha_equity"].iloc[-1])
    final_expected_alpha = float(expected_alpha_curve.iloc[-1])

    assert np.isclose(
        final_expected_alpha, final_recorded_alpha, atol=1e-4
    ), f"Geometric Alpha Ratio drift: expected {final_expected_alpha}, got {final_recorded_alpha}"
