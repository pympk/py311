import pandas as pd
import numpy as np

from typing import List
from core.quant import QuantUtils
from core.settings import GLOBAL_SETTINGS
from typing import List, Union, Tuple


def generate_features(
    df_ohlcv: pd.DataFrame, df_indices: pd.DataFrame = None, **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Orchestrates feature generation by delegating math to QuantUtils.
    """

    # --- PREP & CONFIG ---
    cfg = {**GLOBAL_SETTINGS, **kwargs}

    # 1. Map Windows (Handles the previous KeyError)
    win_5 = cfg.get("5d_window", 5)
    win_21 = cfg.get("21d_window", 21)
    win_63 = cfg.get("63d_window", 63)

    # 2. Extract specific settings safely
    benchmark = cfg.get("benchmark_ticker", "SPY")
    clip_val = cfg.get("feature_zscore_clip", 3.0)

    # 3. Prepare Data (Essential for rolling math)
    df_ohlcv = df_ohlcv.sort_index(level=["Ticker", "Date"])
    all_dates = df_ohlcv.index.get_level_values("Date").unique().sort_values()

    benchmark = cfg["benchmark_ticker"]
    clip_val = cfg["feature_zscore_clip"]

    print(f"Generating Features (Benchmark: {benchmark})...")

    # --- 1. MACRO ENGINE ---
    macro_df = pd.DataFrame(index=all_dates)
    if benchmark in df_ohlcv.index.get_level_values("Ticker"):
        mkt_close = (
            df_ohlcv.xs(benchmark, level="Ticker")["Adj Close"]
            .reindex(all_dates)
            .ffill()
        )
        macro_df["Mkt_Ret"] = mkt_close.pct_change().fillna(0.0)
        macro_df["Macro_Trend"] = (mkt_close / mkt_close.rolling(200).mean()) - 1.0
    else:
        macro_df["Mkt_Ret"], macro_df["Macro_Trend"] = 0.0, 0.0

    macro_df["Macro_Trend_Vel"] = macro_df["Macro_Trend"].diff(win_21)
    macro_df["Macro_Trend_Vel_Z"] = (
        macro_df["Macro_Trend_Vel"] / macro_df["Macro_Trend"].rolling(win_63).std()
    ).clip(-clip_val, clip_val)

    macro_df["Macro_Trend_Mom"] = (
        np.sign(macro_df["Macro_Trend"])
        * np.sign(macro_df["Macro_Trend_Vel"])
        * np.abs(macro_df["Macro_Trend_Vel"])
    ).fillna(0)

    # VIX Extraction
    macro_df["Macro_Vix_Z"], macro_df["Macro_Vix_Ratio"] = 0.0, 1.0
    if df_indices is not None:
        idx_names = df_indices.index.get_level_values(0).unique()
        if "^VIX" in idx_names:
            v = df_indices.xs("^VIX", level=0)["Adj Close"].reindex(all_dates).ffill()
            macro_df["Macro_Vix_Z"] = (
                (v - v.rolling(63).mean()) / v.rolling(63).std()
            ).clip(-clip_val, clip_val)
            if "^VIX3M" in idx_names:
                v3 = (
                    df_indices.xs("^VIX3M", level=0)["Adj Close"]
                    .reindex(all_dates)
                    .ffill()
                )
                macro_df["Macro_Vix_Ratio"] = (v / v3).fillna(1.0)
    macro_df.fillna(0.0, inplace=True)

    # --- 2. TICKER ENGINE (MICRO) ---
    grouped = df_ohlcv.groupby(level="Ticker", group_keys=False)
    adj_close = df_ohlcv["Adj Close"]

    ###########
    # Calculate returns for the specific price column
    rets = grouped["Adj Close"].pct_change()

    # Beta: Perform per-ticker to avoid data leaking between different symbols
    mkt_rets_aligned = macro_df["Mkt_Ret"].reindex(df_ohlcv.index, level="Date")

    # We use apply on the group to keep the rolling window bounded by Ticker
    beta_63 = grouped.apply(
        lambda x: QuantUtils.calculate_rolling_beta(
            x["Adj Close"].pct_change(), mkt_rets_aligned.loc[x.index], win_63
        )
    )

    active_ret = rets - mkt_rets_aligned
    roll_active = active_ret.groupby(level="Ticker").rolling(win_63)

    ir_63 = (
        (roll_active.mean() / roll_active.std())
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    atr = grouped.apply(
        lambda x: QuantUtils.calculate_atr(
            x["Adj High"], x["Adj Low"], x["Adj Close"], cfg["atr_period"]
        )
    )

    natr = (atr / adj_close).fillna(0)

    # Vectorized True Range for TRP
    tr = np.maximum(
        df_ohlcv["Adj High"] - df_ohlcv["Adj Low"],
        np.maximum(
            (df_ohlcv["Adj High"] - adj_close.shift(1)).abs(),
            (df_ohlcv["Adj Low"] - adj_close.shift(1)).abs(),
        ),
    )
    trp = (tr / adj_close).fillna(0)

    # C. RSI & Momentum
    rsi = grouped["Adj Close"].apply(QuantUtils.calculate_rsi, period=cfg["rsi_period"])
    mom_21 = grouped["Adj Close"].pct_change(win_21)
    consistency = (
        (rets > 0)
        .astype(float)
        .groupby(level="Ticker")
        .rolling(win_5)
        .mean()
        .reset_index(level=0, drop=True)
    )

    dd_21 = (
        adj_close
        / grouped["Adj Close"].rolling(win_21).max().reset_index(level=0, drop=True)
    ) - 1.0

    # D. Assemble Features
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
        },
        index=df_ohlcv.index,
    )

    # E. Quality Filtering
    quality_temp = pd.DataFrame(
        {
            "IsStale": np.where(
                (df_ohlcv["Volume"] == 0)
                | (df_ohlcv["Adj High"] == df_ohlcv["Adj Low"]),
                1,
                0,
            ),
            "DollarVolume": adj_close * df_ohlcv["Volume"],
            "HasSameVolume": (grouped["Volume"].diff() == 0).astype(int),
        },
        index=df_ohlcv.index,
    )

    grp_q = quality_temp.groupby(level="Ticker")
    rolling_quality = pd.DataFrame(
        {
            "RollingStalePct": grp_q["IsStale"]
            .rolling(
                window=cfg["quality_window"], min_periods=cfg["quality_min_periods"]
            )
            .mean()
            .values,
            "RollMedDollarVol": grp_q["DollarVolume"]
            .rolling(
                window=cfg["quality_window"], min_periods=cfg["quality_min_periods"]
            )
            .median()
            .values,
            "RollingSameVolCount": grp_q["HasSameVolume"]
            .rolling(
                window=cfg["quality_window"], min_periods=cfg["quality_min_periods"]
            )
            .sum()
            .values,
        },
        index=quality_temp.index,
    )

    return pd.concat([features_df, rolling_quality], axis=1).sort_index(), macro_df


#
