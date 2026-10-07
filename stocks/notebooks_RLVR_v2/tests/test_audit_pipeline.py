import pytest
import pandas as pd
import numpy as np
from types import SimpleNamespace

from core.settings import TradingConfig
from core.paths import GLOBAL_PROCESSED_DIR, LOCAL_DATA_DIR
from core.contracts import EngineInput
from walk_forward import (
    AlphaEngine,
    create_walk_forward_analyzer,
)
from core.utils import SystemUtils as SU
from strategy.registry import get_strategy_registry


@pytest.fixture(scope="module")
def audit_data():
    """Loads the actual processed features and raw OHLCV for auditing."""
    df_ohlcv = pd.read_parquet(GLOBAL_PROCESSED_DIR / "df_ohlcv.parquet")
    features_df = pd.read_parquet(LOCAL_DATA_DIR / "features_df.parquet")
    config = TradingConfig()
    return df_ohlcv, features_df, config


@pytest.fixture(scope="module")
def engine_data(audit_data):
    """Loads macro_df and unstacks the wide matrices on-the-fly for AlphaEngine."""
    df_ohlcv, features_df, config = audit_data
    macro_df = pd.read_parquet(LOCAL_DATA_DIR / "macro_df.parquet")

    print("\n🚀 Unstacking Wide Matrices on-the-fly for tests...")

    # 1. Unstack directly. Level 0 is 'Ticker' based on data_pipeline logic.
    df_close_wide = df_ohlcv["Adj Close"].unstack(level=0)
    df_atrp_wide = features_df["ATRP"].unstack(level=0)
    df_trp_wide = features_df["TRP"].unstack(level=0)

    # Ensure we only test on tickers that exist in BOTH features and prices
    common_tickers = df_close_wide.columns.intersection(
        features_df.index.get_level_values("Ticker").unique()
    )
    df_close_wide = df_close_wide[common_tickers]

    idx = pd.IndexSlice
    features_df = features_df.loc[idx[common_tickers, :], :]
    df_atrp_wide = df_atrp_wide[common_tickers]
    df_trp_wide = df_trp_wide[common_tickers]

    # 2. Terminal Fills ONLY.
    # SYSTEM RULE ENFORCEMENT: Never fill missing prices with 0.0 (nan_price_replacement).
    # Pre-IPO NaNs MUST remain NaN so the engine can correctly bfill() mid-period IPOs.
    df_atrp_wide = df_atrp_wide.fillna(0.0)
    df_trp_wide = df_trp_wide.fillna(0.0)

    return (
        df_ohlcv,
        features_df,
        macro_df,
        df_close_wide,
        df_atrp_wide,
        df_trp_wide,
        config,
    )


def test_audit_rsi_atrp_parity(audit_data):
    """Verifies Pipeline RSI/ATRP against manual Wilder's EWM math."""
    df_ohlcv, features_df, config = audit_data

    ticker = "NVDA"
    common_dates = df_ohlcv.xs(ticker, level="Ticker").index.intersection(
        features_df.xs(ticker, level="Ticker").index
    )
    target_date = common_dates[-10]

    feat_row = features_df.xs((ticker, target_date))
    pipe_rsi = feat_row["RSI"]
    pipe_atrp = feat_row["ATRP"]

    prices = df_ohlcv.xs(ticker, level="Ticker").loc[:target_date]
    adj_close = prices["Adj Close"]

    delta = adj_close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1 / config.rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / config.rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss

    manual_rsi_raw = (100 - (100 / (1 + rs))).iloc[-1]
    manual_rsi = (manual_rsi_raw - 50) / 50

    prev_close = adj_close.shift(1)
    tr = pd.concat(
        [
            (prices["Adj High"] - prices["Adj Low"]),
            (prices["Adj High"] - prev_close).abs(),
            (prices["Adj Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    manual_atr = tr.ewm(alpha=1 / config.atr_period, adjust=False).mean()
    manual_atrp = (manual_atr / adj_close).iloc[-1]

    assert np.isclose(
        pipe_rsi, manual_rsi, rtol=1e-8
    ), f"RSI Mismatch: {pipe_rsi} != {manual_rsi}"
    assert np.isclose(
        pipe_atrp, manual_atrp, rtol=1e-8
    ), f"ATRP Mismatch: {pipe_atrp} != {manual_atrp}"


def test_audit_momentum_drawdown_parity(audit_data):
    """Verifies Basic Group: Mom_21 and DD_21."""
    df_ohlcv, features_df, _ = audit_data
    ticker = "NVDA"
    target_date = features_df.xs(ticker, level="Ticker").index[-1]

    feat_row = features_df.xs((ticker, target_date))
    prices = df_ohlcv.xs(ticker, level="Ticker")["Adj Close"].loc[:target_date]

    manual_mom_21 = prices.pct_change(21).iloc[-1]
    rolling_max_21 = prices.rolling(window=21).max()
    manual_dd_21 = (prices / rolling_max_21 - 1).iloc[-1]

    assert np.isclose(feat_row["Mom_21"], manual_mom_21, rtol=1e-8)
    assert np.isclose(feat_row["DD_21"], manual_dd_21, rtol=1e-8)


def get_mapped_value(search_term, mapped_data):
    """Helper to extract nested values from SU.map_analyzer results."""
    for item in mapped_data:
        if item.get("name") == search_term or item.get("path") == search_term:
            return item.get("obj")
    raise ValueError(f"Key '{search_term}' not found in mapped audit data.")


def test_audit_portfolio_drift_weights(engine_data):
    """Verifies that the walk-forward engine correctly calculates drift weights during the holding period."""
    (
        df_ohlcv,
        features_df,
        macro_df,
        df_close_wide,
        df_atrp_wide,
        df_trp_wide,
        config,
    ) = engine_data

    # 1. Initialize Engine
    engine = AlphaEngine(
        df_ohlcv=df_ohlcv,
        features_df=features_df,
        macro_df=macro_df,
        df_close_wide=df_close_wide,
        df_atrp_wide=df_atrp_wide,
        df_trp_wide=df_trp_wide,
    )

    _inputs = EngineInput(
        mode="Ranking",
        decision_date=pd.Timestamp("2026-04-16"),
        lookback_period=189,
        holding_period=5,
        metric="Sharpe (TRP)",
        benchmark_ticker=config.benchmark_ticker,
        rank_start=1,
        rank_end=200,
        debug=True,
    )

    # 2. Run Single Simulation Step
    analyzer, _ = create_walk_forward_analyzer(engine, _inputs, universe_subset=None)
    analyzer.last_run = engine.run(_inputs)

    # DIAGNOSTIC TRAP: Unmask the silent engine error
    if analyzer.last_run.error_msg:
        pytest.fail(f"[ENGINE ABORTED] Reason: {analyzer.last_run.error_msg}")

    result_map = SU.map_analyzer(analyzer=analyzer)

    if not result_map:
        pytest.fail("result_map is empty! Engine run failed to populate audit data.")

    # 3. Extract Dates & Raw Components
    start_date = SU.fetch("start_date", result_map)
    decision_date = SU.fetch("decision_date", result_map)
    buy_date = SU.fetch("buy_date", result_map)
    holding_end_date = SU.fetch("holding_end_date", result_map)

    raw_prices = get_mapped_value(
        "audit_pack -> debug_data -> portfolio_raw_components -> prices", result_map
    )
    raw_atrp = get_mapped_value(
        "audit_pack -> debug_data -> portfolio_raw_components -> atrp", result_map
    )
    raw_trp = get_mapped_value(
        "audit_pack -> debug_data -> portfolio_raw_components -> trp", result_map
    )

    # Assert all components were extracted properly
    assert raw_prices is not None, "raw_prices failed to map from result_map"
    assert raw_atrp is not None, "raw_atrp failed to map from result_map"
    assert raw_trp is not None, "raw_trp failed to map from result_map"

    audit_data_vals = {
        f"{p}_p_{s}": get_mapped_value(f"{p}_p_{s}", result_map)
        for s in ["gain", "sharpe", "sharpe_atrp", "sharpe_trp"]
        for p in ["full", "lookback", "holding"]
    }

    period_slices = {
        "Full": (start_date, holding_end_date),
        "Lookback": (start_date, decision_date),
        "Holding": (buy_date, holding_end_date),
    }

    # 4. Manual Drift Weight Verification Math
    for name, (s_date, e_date) in period_slices.items():
        p_slice = raw_prices.loc[s_date:e_date].dropna(how="all", axis=1)
        a_slice = raw_atrp.loc[s_date:e_date]
        t_slice = raw_trp.loc[s_date:e_date]

        # Mimic QuantUtils.compute_portfolio_stats NaN handling perfectly
        base_prices = p_slice.bfill().iloc[0]
        valid_cols = base_prices.dropna().index

        p_slice = p_slice[valid_cols]
        base_prices = base_prices[valid_cols]

        norm_prices = p_slice.div(base_prices)

        # Initial equal weights for valid columns
        initial_weights = pd.Series(1.0 / len(valid_cols), index=valid_cols)

        weighted_components = norm_prices.mul(initial_weights, axis=1)
        equity_curve = weighted_components.sum(axis=1)

        returns = equity_curve.pct_change().dropna()
        weights = weighted_components.div(equity_curve, axis=0)

        a_slice = a_slice.reindex(columns=valid_cols)
        t_slice = t_slice.reindex(columns=valid_cols)

        port_atrp = (weights * a_slice).sum(axis=1).loc[returns.index]
        port_trp = (weights * t_slice).sum(axis=1).loc[returns.index]

        # Manual Results
        manual_log_gain = float(np.log(equity_curve.iloc[-1] / equity_curve.iloc[0]))
        manual_sharpe = float((returns.mean() / returns.std()) * np.sqrt(252))
        manual_sharpe_atrp = float(returns.mean() / port_atrp.mean())
        manual_sharpe_trp = float(returns.mean() / port_trp.mean())

        p_prefix = name.lower()
        sys_log_gain = float(audit_data_vals[f"{p_prefix}_p_gain"] or 0)

        # 5. Assertions (Relaxed for float32 precision)
        assert np.isclose(
            manual_log_gain, sys_log_gain, rtol=1e-4, atol=1e-5
        ), f"{name} Log Gain mismatch. Manual: {manual_log_gain} vs Sys: {sys_log_gain}"

        assert np.isclose(
            manual_sharpe,
            float(audit_data_vals[f"{p_prefix}_p_sharpe"] or 0),
            rtol=1e-4,
            atol=1e-5,
        ), f"{name} Sharpe mismatch"

        assert np.isclose(
            manual_sharpe_atrp,
            float(audit_data_vals[f"{p_prefix}_p_sharpe_atrp"] or 0),
            rtol=1e-4,
            atol=1e-5,
        ), f"{name} Sharpe ATRP mismatch"

        assert np.isclose(
            manual_sharpe_trp,
            float(audit_data_vals[f"{p_prefix}_p_sharpe_trp"] or 0),
            rtol=1e-4,
            atol=1e-5,
        ), f"{name} Sharpe TRP mismatch"


def test_audit_cross_sectional_blueprints(audit_data):
    """Verifies all 12 Strategy Registry Blueprints execute cleanly and strictly preserve K=12 invariant."""
    _, features_df, config = audit_data
    registry = get_strategy_registry(config)

    assert (
        len(registry) == 12
    ), f"Registry factor count violated! Expected 12, got {len(registry)}"

    all_dates = features_df.index.get_level_values("Date")
    target_date = all_dates.max()
    daily_snapshot = features_df.xs(target_date, level="Date")
    idx = daily_snapshot.index

    # Construct complete observation mock with all 12 Gen 16 factor fields
    obs = SimpleNamespace(
        mom_252_21=daily_snapshot.get("Mom_252_21", pd.Series(0.0, index=idx)),
        res_mom_126=daily_snapshot.get("ResMom_126", pd.Series(0.0, index=idx)),
        range_pos_52w=daily_snapshot.get("Range_Pos_52w", pd.Series(1.0, index=idx)),
        ir_63=daily_snapshot.get("IR_63", pd.Series(0.0, index=idx)),
        rsi=daily_snapshot.get("RSI", pd.Series(50.0, index=idx)),
        dd_21=daily_snapshot.get("DD_21", pd.Series(0.0, index=idx)),
        trend_r2_63=daily_snapshot.get("Trend_R2_63", pd.Series(0.0, index=idx)),
        beta_down_63=daily_snapshot.get("Beta_Down_63", pd.Series(1.0, index=idx)),
        atrp=daily_snapshot.get("ATRP", pd.Series(0.02, index=idx)),
        er_63=daily_snapshot.get("ER_63", pd.Series(0.5, index=idx)),
        mom_126=daily_snapshot.get("Mom_126", pd.Series(0.0, index=idx)),
        ivol_63=daily_snapshot.get("IVol_63", pd.Series(0.0, index=idx)),
    )

    # Invariant assertion: All 12 blueprints must execute and return full cross-sectional series
    for name, blueprint in registry.items():
        scores = blueprint(obs)
        assert len(scores) == len(
            daily_snapshot
        ), f"Blueprint '{name}' failed dimension check"
        assert not scores.isna().all(), f"Blueprint '{name}' returned all NaNs"
