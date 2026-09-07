import pytest
import pickle
import numpy as np
import pandas as pd
import re

from core.paths import OUTPUT_DIR
from core.settings import TradingConfig
from strategy.registry import get_strategy_registry


def _extract_run_hyperparameters(filename: str, metadata: dict | None = None) -> dict:
    metadata = metadata or {}
    params: dict = {}

    # 1. upside_alpha_mult: from metadata or tokens like _mult2.5_, _upside_2.5_, _up_2.5_
    mult_match = re.search(r"_(?:mult|upside|up)_?([0-9.]+)(?:_|$)", filename)
    if "upside_alpha_mult" in metadata:
        params["upside_alpha_mult"] = float(metadata["upside_alpha_mult"])
    elif mult_match:
        params["upside_alpha_mult"] = float(mult_match.group(1))

    # 2. loss_aversion_penalty: from metadata or tokens like _lossav_1.0_, _pen_2.0_, _loss_0.5_
    lossav_match = re.search(r"_(?:lossav|loss)_?([0-9.]+)(?:_|$)", filename)
    pen_match = re.search(r"_pen_?([0-9.]+)(?:_|$)", filename)
    if "loss_aversion_penalty" in metadata:
        params["loss_aversion_penalty"] = float(metadata["loss_aversion_penalty"])
    elif lossav_match:
        params["loss_aversion_penalty"] = float(lossav_match.group(1))
    elif pen_match:
        params["loss_aversion_penalty"] = max(0.0, float(pen_match.group(1)) - 1.0)

    # 3. min_basket_width: from metadata or tokens like _Width5_, _w5_
    w_match = re.search(r"_(?:Width|width|w)([0-9]+)(?:_|$)", filename)
    if "min_basket_width" in metadata:
        params["min_basket_width"] = int(metadata["min_basket_width"])
    elif w_match:
        params["min_basket_width"] = int(w_match.group(1))

    # 4. min_active_tilt: from metadata or tokens like _tilt0.85_, _Tilt85_
    tilt_match = re.search(r"_(?:tilt|Tilt)([0-9.]+)(?:_|$)", filename)
    if "min_active_tilt" in metadata:
        params["min_active_tilt"] = float(metadata["min_active_tilt"])
    elif tilt_match:
        val = float(tilt_match.group(1))
        params["min_active_tilt"] = val / 100.0 if val > 1.0 else val

    # 5. holding_period: from metadata or token like _T3_
    t_match = re.search(r"_T([0-9]+)(?:_|$)", filename)
    if "holding_period" in metadata:
        params["holding_period"] = int(metadata["holding_period"])
    elif t_match:
        params["holding_period"] = int(t_match.group(1))

    return params


# =====================================================================
# FIXTURES
# =====================================================================


@pytest.fixture(scope="module")
def blotter_df():
    """Finds the most recent OOS RL run and loads the raw blotter dataframe."""
    if not OUTPUT_DIR.exists():
        pytest.skip("Output directory does not exist.")

    pkl_files = list(OUTPUT_DIR.glob("results_*.pkl"))
    if not pkl_files:
        pytest.skip("No OOS results pickle files found.")

    latest_pkl = max(pkl_files, key=lambda p: p.stat().st_mtime)

    with open(latest_pkl, "rb") as f:
        results = pickle.load(f)

    meta = results.get("metadata", {})
    params = _extract_run_hyperparameters(latest_pkl.name, meta)

    loss_penalty = params.get(
        "loss_aversion_penalty",
        getattr(TradingConfig(), "loss_aversion_penalty", 0.0),
    )
    upside_mult = params.get(
        "upside_alpha_mult",
        getattr(TradingConfig(), "upside_alpha_mult", 1.0),
    )

    df = pd.DataFrame(results["blotter"])
    df["decision_date"] = pd.to_datetime(df["decision_date"])
    df["buy_date"] = pd.to_datetime(df["buy_date"])
    df["sell_date"] = pd.to_datetime(df["sell_date"])

    df.attrs["loss_aversion_penalty"] = loss_penalty
    df.attrs["upside_alpha_mult"] = upside_mult
    df.attrs["metadata"] = meta
    df.attrs["extracted_params"] = params
    return df


@pytest.fixture(scope="module")
def trading_config(blotter_df):
    """Loads the core configuration updated with hyperparameters extracted from the artifact."""
    config = TradingConfig()
    params = blotter_df.attrs.get("extracted_params", {})
    for k, v in params.items():
        if hasattr(config, k):
            setattr(config, k, v)
    return config


# =====================================================================
# TESTS
# =====================================================================


def test_temporal_sanity(blotter_df):
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
    registry = get_strategy_registry(trading_config)
    expected_dim = len(registry) + 4

    for idx in [0, -1]:
        row = blotter_df.iloc[idx]
        raw_vector = np.array(row["raw_actions"])

        assert len(raw_vector) == expected_dim, (
            f"Action vector dimensionality mismatch in row {idx}. "
            f"Expected strictly {expected_dim} (K={len(registry)} features + 4 controls), got {len(raw_vector)}."
        )


def test_intra_row_arithmetic(blotter_df, trading_config):
    """
    Recalculates the core MTMPortfolioEngine accounting logic from scratch
    to prove returns, slippage, and penalties were applied correctly.
    """
    loss_aversion_penalty = blotter_df.attrs.get(
        "loss_aversion_penalty", getattr(trading_config, "loss_aversion_penalty", 0.0)
    )
    meta = blotter_df.attrs.get("metadata", {})

    for idx in [0, -1]:
        row = blotter_df.iloc[idx]
        w_active = float(row["weight_active"])
        w_benchmark = float(row["weight_benchmark"])
        w_cash = float(row["weight_cash"])

        # 1. Verify Weights Sum to 1.0
        assert np.isclose(
            w_active + w_benchmark + w_cash, 1.0, atol=1e-5
        ), f"Weights do not sum to 1.0 in row {idx}: {w_active + w_benchmark + w_cash}"

        # 2. Verify Gross Portfolio Return Math
        gross_stock = float(row["gross_stock_daily_simple_ret"])
        bm_ret = float(row["bm_daily_simple_ret"])
        cash_ret = float(row["cash_daily_simple_ret"])
        expected_gross = (
            w_active * gross_stock + w_benchmark * bm_ret + w_cash * cash_ret
        )
        assert np.isclose(
            expected_gross, float(row["gross_daily_simple_ret"]), atol=1e-6
        ), f"Gross Return drift in row {idx}: expected {expected_gross}, got {row['gross_daily_simple_ret']}"

        # 3. Verify Net Return Math
        expected_net = expected_gross - float(row["slippage_daily_simple_loss"])
        assert np.isclose(
            expected_net, float(row["net_daily_simple_ret"]), atol=1e-6
        ), f"Net Return drift in row {idx}: expected {expected_net}, got {row['net_daily_simple_ret']}"

        # 4. Verify Alpha Math
        expected_alpha = float(row["net_daily_simple_ret"]) - bm_ret
        assert np.isclose(
            expected_alpha, float(row["alpha_daily_simple_ret"]), atol=1e-6
        ), f"Alpha Math drift in row {idx}: expected {expected_alpha}, got {row['alpha_daily_simple_ret']}"

        # 5. Verify Penalized Alpha Reward Math
        recorded_alpha = float(row["alpha_daily_simple_ret"])
        recorded_penalized = float(row["penalized_alpha_daily_simple_ret"])

        upside_mult = float(
            meta.get(
                "upside_alpha_mult",
                blotter_df.attrs.get(
                    "upside_alpha_mult",
                    getattr(trading_config, "upside_alpha_mult", 1.0),
                ),
            )
        )

        if recorded_alpha > 0.0:
            expected_penalized_alpha = recorded_alpha * upside_mult
        elif recorded_alpha < 0.0:
            loss_multiplier = 1.0 + max(0.0, loss_aversion_penalty)
            expected_penalized_alpha = recorded_alpha * loss_multiplier
        else:
            expected_penalized_alpha = 0.0

        assert np.isclose(
            expected_penalized_alpha,
            recorded_penalized,
            atol=1e-6,
        ), (
            f"\n--- [TRAP] ACCOUNTING DIAGNOSTICS ---\n"
            f"Metadata in PKL: {meta}\n"
            f"Row {idx}:\n"
            f"  recorded_alpha: {recorded_alpha}\n"
            f"  recorded_penalized: {recorded_penalized}\n"
            f"  expected_penalized: {expected_penalized_alpha}\n"
            f"  upside_mult_used: {upside_mult}\n"
            f"  loss_penalty_used: {loss_aversion_penalty}\n"
            f"  implied_mult: {recorded_penalized / recorded_alpha if recorded_alpha != 0 else 'N/A'}"
        )


def test_first_row_state_initialization(blotter_df):
    first_row = blotter_df.iloc[0]
    expected_initial_equity = 1.0 * (1.0 + float(first_row["net_daily_simple_ret"]))
    assert np.isclose(
        first_row["agent_equity"], expected_initial_equity, atol=1e-6
    ), f"First row equity initialization corrupted. Expected {expected_initial_equity}, got {first_row['agent_equity']}"


def test_cumulative_equity_provenance(blotter_df):
    manual_equity_curve = (1.0 + blotter_df["net_daily_simple_ret"]).cumprod()
    final_recorded_equity = blotter_df["agent_equity"].iloc[-1]
    final_manual_equity = manual_equity_curve.iloc[-1]
    assert np.isclose(
        final_manual_equity, final_recorded_equity, atol=1e-4
    ), f"Phantom Compounding Detected! Manual Final: {final_manual_equity}, Recorded Final: {final_recorded_equity}"
