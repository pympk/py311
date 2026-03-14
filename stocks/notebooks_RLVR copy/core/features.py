import pandas as pd
import numpy as np

from typing import Tuple
from core.quant import QuantUtils, TickerEngine
from core.settings import GLOBAL_SETTINGS


def generate_features(
    df_ohlcv: pd.DataFrame,
    df_indices: pd.DataFrame = None,
    benchmark_ticker: str = GLOBAL_SETTINGS["benchmark_ticker"],
    atr_period: int = GLOBAL_SETTINGS["atr_period"],
    rsi_period: int = GLOBAL_SETTINGS["rsi_period"],
    win_5d: int = GLOBAL_SETTINGS["5d_window"],
    win_21d: int = GLOBAL_SETTINGS["21d_window"],
    win_63d: int = GLOBAL_SETTINGS["63d_window"],
    feature_zscore_clip: float = GLOBAL_SETTINGS["feature_zscore_clip"],
    quality_window: int = GLOBAL_SETTINGS["quality_window"],
    quality_min_periods: int = GLOBAL_SETTINGS["quality_min_periods"],
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    print(f"⚡ Generating Decoupled Features (Benchmark: {benchmark_ticker})...")

    # --- 0. PREP ---
    df_ohlcv = df_ohlcv.sort_index(level=["Ticker", "Date"])
    all_dates = df_ohlcv.index.get_level_values("Date").unique().sort_values()

    # --- 1. MACRO ENGINE ---
    macro_df = pd.DataFrame(index=all_dates)
    if benchmark_ticker in df_ohlcv.index.get_level_values("Ticker"):
        mkt_close = (
            df_ohlcv.xs(benchmark_ticker, level="Ticker")["Adj Close"]
            .reindex(all_dates)
            .ffill()
        )
        macro_df["Mkt_Ret"] = mkt_close.pct_change().fillna(0.0)
        macro_df["Macro_Trend"] = (mkt_close / mkt_close.rolling(200).mean()) - 1.0
    else:
        macro_df["Mkt_Ret"] = 0.0
        macro_df["Macro_Trend"] = 0.0

    # --- TREND VELOCITY & MOMENTUM ---
    macro_df["Macro_Trend_Vel"] = macro_df["Macro_Trend"].diff(win_21d)
    macro_df["Macro_Trend_Vel_Z"] = (
        macro_df["Macro_Trend_Vel"] / macro_df["Macro_Trend"].rolling(win_63d).std()
    ).clip(-feature_zscore_clip, feature_zscore_clip)
    macro_df["Macro_Trend_Mom"] = (
        np.sign(macro_df["Macro_Trend"])
        * np.sign(macro_df["Macro_Trend_Vel"])
        * np.abs(macro_df["Macro_Trend_Vel"])
    ).fillna(0)

    # VIX Extraction (Same as before)
    macro_df["Macro_Vix_Z"] = 0.0
    macro_df["Macro_Vix_Ratio"] = 1.0
    if df_indices is not None:
        idx_names = df_indices.index.get_level_values(0).unique()
        if "^VIX" in idx_names:
            v = df_indices.xs("^VIX", level=0)["Adj Close"].reindex(all_dates).ffill()
            macro_df["Macro_Vix_Z"] = (
                (v - v.rolling(63).mean()) / v.rolling(63).std()
            ).clip(-feature_zscore_clip, feature_zscore_clip)
        if "^VIX" in idx_names and "^VIX3M" in idx_names:
            v3 = (
                df_indices.xs("^VIX3M", level=0)["Adj Close"].reindex(all_dates).ffill()
            )
            macro_df["Macro_Vix_Ratio"] = (v / v3).fillna(1.0)
    macro_df.fillna(0.0, inplace=True)

    # --- 2. TICKER ENGINE ---
    grouped = df_ohlcv.groupby(level="Ticker")

    # STEP 1: Refactor Returns via TickerEngine Orchestrator
    # No more manual groupby or reset_index needed.
    rets = TickerEngine.map_kernels(df_ohlcv["Adj Close"], QuantUtils.compute_returns)

    mkt_ret_series = macro_df["Mkt_Ret"]  # The "Master" market vector

    # A. Hybrid Metrics (Beta & IR)
    # 1. IR_63 (Remains same for now as it uses internal rolling logic)
    active_ret = rets.sub(mkt_ret_series, axis=0, level="Date")
    roll_active = active_ret.groupby(level="Ticker").rolling(win_63d)
    ir_63 = (
        (roll_active.mean() / roll_active.std())
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    # 2. Beta_63 - Refactored using TickerEngine Orchestrator
    # We pass the market series as a keyword argument (benchmark_rets)
    beta_63 = TickerEngine.map_kernels(
        rets,
        QuantUtils.calculate_rolling_beta,
        benchmark_rets=mkt_ret_series,
        window=win_63d,
    )

    # B. Volatility (ATR / TRP) - Using TickerEngine to bridge multiple kernels
    def get_ticker_vol(df_slice):
        """Internal bridge to call multiple TR/ATR kernels for a single ticker."""
        h, l, c = df_slice["Adj High"], df_slice["Adj Low"], df_slice["Adj Close"]
        return pd.DataFrame(
            {
                "TR_Raw": QuantUtils.calculate_tr(h, l, c),
                "ATR_Smooth": QuantUtils.calculate_atr(h, l, c, atr_period),
            },
            index=df_slice.index,
        )

    # The Orchestrator handles the ticker-by-ticker application
    vol_bundle = TickerEngine.map_kernels(df_ohlcv, get_ticker_vol)

    # Alignment is guaranteed by the Orchestrator
    atr = vol_bundle["ATR_Smooth"]
    natr = (atr / df_ohlcv["Adj Close"]).fillna(0)  # ATRP (normalized)
    trp = (vol_bundle["TR_Raw"] / df_ohlcv["Adj Close"]).fillna(0)  # TRP (raw)

    # C. Momentum & Consistency (Keep existing for this step)
    mom_21 = grouped["Adj Close"].pct_change(win_21d)
    consistency = (
        (rets > 0)
        .astype(float)
        .groupby(level="Ticker")
        .rolling(win_5d)
        .mean()
        .reset_index(level=0, drop=True)
    )
    dd_21 = (
        df_ohlcv["Adj Close"]
        / grouped["Adj Close"].rolling(win_21d).max().reset_index(level=0, drop=True)
    ) - 1.0

    # STEP 2: Refactor RSI via TickerEngine Orchestrator
    # Clean, declarative call to the Wilder's RSI math kernel
    rsi = TickerEngine.map_kernels(
        df_ohlcv["Adj Close"], QuantUtils.calculate_rsi, period=rsi_period
    )

    # E. Assemble Features (Remains the same)
    features_df = pd.DataFrame(
        {
            "ATR": atr,
            "ATRP": natr,
            "TRP": trp,
            "RSI": rsi,
            "Mom_21": mom_21,
            "Consistency": consistency,
            "IR_63": ir_63,
            "Beta_63": beta_63,
            "DD_21": dd_21.fillna(0),
            "Ret_1d": rets,
        }
    )

    # F. Quality (Universe Filtering) - Optimized
    quality_temp = pd.DataFrame(
        {
            "IsStale": np.where(
                (df_ohlcv["Volume"] == 0)
                | (df_ohlcv["Adj High"] == df_ohlcv["Adj Low"]),
                1,
                0,
            ),
            "DollarVolume": df_ohlcv["Adj Close"] * df_ohlcv["Volume"],
            "HasSameVolume": (grouped["Volume"].diff() == 0).astype(int),
        },
        index=df_ohlcv.index,
    )

    # Calculate rolling stats separately (avoid slow dict agg) and use .values to bypass index alignment overhead
    grp = quality_temp.groupby(level="Ticker")
    rolling_quality = pd.DataFrame(
        {
            "RollingStalePct": grp["IsStale"]
            .rolling(window=quality_window, min_periods=quality_min_periods)
            .mean()
            .values,
            "RollMedDollarVol": grp["DollarVolume"]
            .rolling(window=quality_window, min_periods=quality_min_periods)
            .median()
            .values,
            "RollingSameVolCount": grp["HasSameVolume"]
            .rolling(window=quality_window, min_periods=quality_min_periods)
            .sum()
            .values,
        },
        index=quality_temp.index,
    )

    return pd.concat([features_df, rolling_quality], axis=1).sort_index(), macro_df


#
