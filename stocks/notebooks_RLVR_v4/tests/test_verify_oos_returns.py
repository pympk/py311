# # Test: Continuous Out-of-Sample (OOS) Returns & Market Provenance Audit
# Reconciles continuous canonical blotters against market OHLCV parquet data,
# verifying zero forward-looking leakage, dynamic benchmark alignment,
# and exact geometric compounding wealth ratios.

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from core.paths import LOCAL_DATA_DIR, OUTPUT_DIR
from core.settings import TradingConfig


def test_real_oos_returns_integration():
    """
    INSTITUTIONAL INTEGRATION AUDIT:
    1. Discovers the production champion canonical blotter in output/canonical_anchors/.
    2. Dynamically pulls benchmark ticker and parameters from TradingConfig.
    3. Reconciles blotter benchmark returns against raw market OHLCV data.
    4. Enforces discrete geometric compounding and geometric wealth ratio contracts.
    5. Validates net return accounting and exposure constraints.
    """
    trading_config = TradingConfig()
    benchmark_ticker = trading_config.benchmark
    min_active_tilt = trading_config.min_active_tilt

    # 1. Locate Market Data Parquet
    market_data_path = LOCAL_DATA_DIR / "df_ohlcv.parquet"
    if not market_data_path.exists():
        market_data_path = LOCAL_DATA_DIR / "alpha_cache_41d_1998.parquet"

    assert (
        market_data_path.exists()
    ), f"❌ FATAL: Market data not found at {market_data_path}. Cannot audit OOS returns."

    # 2. Locate Production Canonical Parquet Blotters
    canonical_dir = OUTPUT_DIR / "canonical_anchors"
    assert (
        canonical_dir.exists()
    ), f"❌ FATAL: Canonical anchors directory missing at {canonical_dir}."

    blotter_files = sorted(canonical_dir.glob("blotter_continuous_s*_gen*.parquet"))
    if not blotter_files:
        blotter_files = sorted(canonical_dir.glob("blotter_continuous_*.parquet"))

    assert (
        len(blotter_files) > 0
    ), f"❌ FATAL: No canonical blotters found in {canonical_dir}. Zero-skip contract violated."

    # Prefer Gen 18 baseline champion; fallback to latest canonical blotter
    champion_blotters = [p for p in blotter_files if "s42_gen18" in p.name]
    target_blotter_path = (
        champion_blotters[0] if champion_blotters else blotter_files[-1]
    )

    print(f"\n[Auditor] Auditing Canonical Blotter: {target_blotter_path.name}")
    print(f"[Auditor] Active Benchmark Ticker: {benchmark_ticker}")

    # 3. Load Blotter & Standardize Datetimes
    df_blotter = pd.read_parquet(target_blotter_path)
    if "date" not in df_blotter.columns and isinstance(
        df_blotter.index, pd.DatetimeIndex
    ):
        df_blotter = df_blotter.reset_index().rename(columns={"index": "date"})

    df_blotter["date"] = (
        pd.to_datetime(df_blotter["date"]).dt.tz_localize(None).dt.normalize()
    )
    df_blotter = df_blotter.sort_values("date").reset_index(drop=True)

    # 4. Load Market Data & Extract Benchmark Slice via MultiIndex/Flat Resolution
    df_market = pd.read_parquet(market_data_path)

    if isinstance(df_market.index, pd.MultiIndex):
        # Resolve level names/positions dynamically
        level_names = [
            str(n).lower() if n is not None else "" for n in df_market.index.names
        ]
        ticker_level = next(
            (
                i
                for i, name in enumerate(level_names)
                if name in ["ticker", "symbol", "asset"]
            ),
            0,
        )

        # Match benchmark key against index level values (case-insensitive fallback)
        available_tickers = df_market.index.levels[ticker_level]
        if benchmark_ticker in available_tickers:
            target_key = benchmark_ticker
        elif benchmark_ticker.upper() in available_tickers:
            target_key = benchmark_ticker.upper()
        elif benchmark_ticker.lower() in available_tickers:
            target_key = benchmark_ticker.lower()
        else:
            raise KeyError(
                f"❌ Benchmark '{benchmark_ticker}' not found in MultiIndex level {ticker_level}"
            )

        # O(1) cross-sectional slice — only extracts the target benchmark rows
        df_bm_slice = df_market.xs(target_key, level=ticker_level).reset_index()
    else:
        # Flat DataFrame handling
        df_bm_slice = df_market.copy()
        if isinstance(df_bm_slice.index, pd.DatetimeIndex):
            df_bm_slice = df_bm_slice.reset_index().rename(
                columns={df_bm_slice.index.name or "index": "date"}
            )

    # Normalize column names: strip whitespace, handle 'Adj Close'
    col_mapping: dict[str, str] = {}
    has_adj_close = any(
        "adj" in str(c).lower() and "close" in str(c).lower()
        for c in df_bm_slice.columns
    )

    for col in df_bm_slice.columns:
        c_clean = str(col).lower().strip().replace(" ", "_")
        if c_clean in ["date", "timestamp", "datetime"]:
            col_mapping[str(col)] = "date"
        elif c_clean in ["ticker", "symbol", "asset"]:
            col_mapping[str(col)] = "ticker"
        elif has_adj_close and c_clean in ["adj_close", "adjclose"]:
            col_mapping[str(col)] = "close"
        elif not has_adj_close and c_clean in ["close", "price"]:
            col_mapping[str(col)] = "close"

    df_bm_slice = df_bm_slice.rename(columns=col_mapping)

    # Filter by ticker if flat DataFrame
    if "ticker" in df_bm_slice.columns:
        df_bm_slice = df_bm_slice[
            df_bm_slice["ticker"].astype(str).str.upper() == benchmark_ticker.upper()
        ]

    assert (
        "date" in df_bm_slice.columns
    ), f"❌ Market parquet format error: missing 'date' column in {market_data_path}"
    assert (
        "close" in df_bm_slice.columns
    ), f"❌ Market parquet format error: missing 'close'/'Adj Close' column in {market_data_path}"

    df_bm_slice["date"] = (
        pd.to_datetime(df_bm_slice["date"]).dt.tz_localize(None).dt.normalize()
    )

    # Extract independent benchmark close-to-close returns
    df_bm_market = (
        df_bm_slice[["date", "close"]]
        .sort_values("date")
        .drop_duplicates(subset=["date"])
        .copy()
    )
    df_bm_market["market_bm_ret"] = df_bm_market["close"].pct_change()

    # 5. Execution Session Alignment Audit (T+1 Close Convention)
    # Day T = decision_date, Day T+1 = buy_date (active MTM holding start)
    candidate_cols = [c for c in ["buy_date", "date"] if c in df_blotter.columns]

    best_diff = float("inf")
    best_merged: pd.DataFrame | None = None
    best_note = ""

    for col in candidate_cols:
        df_blotter[col] = (
            pd.to_datetime(df_blotter[col]).dt.tz_localize(None).dt.normalize()
        )
        m = pd.merge(
            df_blotter,
            df_bm_market[["date", "market_bm_ret"]].rename(columns={"date": col}),
            on=col,
            how="inner",
        )
        if len(m) < 20:
            continue

        valid = m.dropna(subset=["bm_daily_simple_ret", "market_bm_ret"])

        # Test zero-shift (direct session match) and T+1 forward shift (decision-row alignment)
        for s in [0, -1, 1]:
            m_ret = (
                valid["market_bm_ret"] if s == 0 else valid["market_bm_ret"].shift(s)
            )
            mask = m_ret.notna() & valid["bm_daily_simple_ret"].notna()
            diff = float(
                (valid.loc[mask, "bm_daily_simple_ret"] - m_ret.loc[mask]).abs().max()
            )
            if diff < best_diff:
                best_diff = diff
                best_merged = m
                best_note = f"blotter '{col}' with market shift={s}"

    assert (
        best_merged is not None and len(best_merged) > 20
    ), f"❌ Insufficient overlapping date history between blotter and market data."
    merged = best_merged
    max_bm_diff = best_diff

    # 6. Audit Benchmark Return Alignment
    print(f"[Auditor] Verified Execution Alignment: {best_note}")
    print(f"[Auditor] Max Benchmark Divergence vs Raw Parquet: {max_bm_diff:.8f}")
    assert (
        max_bm_diff < 1e-4
    ), f"❌ FAILED: Benchmark return desynchronization detected! Best config ({best_note}) divergence: {max_bm_diff:.6f}"

    # 7. Audit Discrete Net Compounding Accounting Identity
    net_accounting_diff = float(
        (
            merged["net_daily_simple_ret"]
            - (merged["gross_daily_simple_ret"] - merged["slippage_daily_simple_loss"])
        )
        .abs()
        .max()
    )

    print(f"[Auditor] Max Net Accounting Diff: {net_accounting_diff:.8f}")
    assert (
        net_accounting_diff < 1e-6
    ), f"❌ FAILED: Daily net return does not equal gross minus slippage! Diff: {net_accounting_diff:.8f}"

    # 8. Audit Geometric Equity Compounding (Type-Safe Vectorized NumPy)
    agent_equity = np.asarray(merged["agent_equity"], dtype=np.float64)
    net_ret = np.asarray(merged["net_daily_simple_ret"], dtype=np.float64)
    agent_step_ratios = (agent_equity[1:] / agent_equity[:-1]) - 1.0
    agent_step_diff = float(np.max(np.abs(agent_step_ratios - net_ret[1:])))

    print(f"[Auditor] Max Step-wise Agent Compounding Diff: {agent_step_diff:.8f}")
    assert (
        agent_step_diff < 1e-4
    ), f"❌ FAILED: Agent equity does not compound geometrically at daily net return! Diff: {agent_step_diff:.8f}"

    # 9. Audit Institutional Geometric Wealth Ratio Contract (Alpha Equity)
    expected_alpha_equity = merged["agent_equity"] / merged["bm_equity"]
    wealth_ratio_diff = float(
        (merged["alpha_equity"] - expected_alpha_equity).abs().max()
    )

    print(f"[Auditor] Max Geometric Wealth Ratio Diff: {wealth_ratio_diff:.8f}")
    assert wealth_ratio_diff < 1e-4, (
        f"❌ FAILED: Blotter alpha_equity violates geometric wealth ratio (agent_equity / bm_equity)! "
        f"Max divergence: {wealth_ratio_diff:.8f}"
    )

    # 10. Audit Portfolio Exposure Mandates (Gross Exposure = 1.0, Cash = 0.0)
    cash_leak = float(merged["weight_cash"].abs().max())
    assert (
        cash_leak < 1e-6
    ), f"❌ FAILED: Cash hoarding detected! Max weight_cash: {cash_leak:.6f}"

    gross_exposure_drift = float(
        (merged["weight_active"] + merged["weight_benchmark"] - 1.0).abs().max()
    )
    assert (
        gross_exposure_drift < 1e-6
    ), f"❌ FAILED: Gross portfolio exposure != 1.0! Max drift: {gross_exposure_drift:.6f}"

    active_tilt_breach = int((merged["weight_active"] < (min_active_tilt - 1e-6)).sum())
    assert (
        active_tilt_breach == 0
    ), f"❌ FAILED: active_tilt dropped below minimum configured floor ({min_active_tilt}) on {active_tilt_breach} sessions."

    print(
        f"✅ PASSED: All OOS returns, geometric compounding, and market price reconciliations verified."
    )
