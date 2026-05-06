import pandas as pd
import numpy as np

from typing import Tuple
from core.quant import QuantUtils, TickerEngine
from core.settings import GLOBAL_SETTINGS


def generate_features(
    df_ohlcv: pd.DataFrame,
    df_indices: pd.DataFrame = None,
    df_fed: pd.DataFrame = None,
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

    # --- 1.2 FED Data Integration ---
    if df_fed is not None:
        fed_data = df_fed.reindex(all_dates).ffill().bfill()
        macro_df["High_Yield_Spread"] = fed_data["High_Yield_Spread"]
        macro_df["Yield_Curve_10Y2Y"] = fed_data["Yield_Curve_10Y2Y"]

        for col in ["High_Yield_Spread", "Yield_Curve_10Y2Y"]:
            roll_mean = macro_df[col].rolling(252, min_periods=60).mean()
            roll_std = macro_df[col].rolling(252, min_periods=60).std()
            macro_df[f"{col}_Z"] = (
                ((macro_df[col] - roll_mean) / roll_std)
                .clip(-feature_zscore_clip, feature_zscore_clip)
                .fillna(0.0)
            )
    else:
        macro_df["High_Yield_Spread"] = 0.0
        macro_df["Yield_Curve_10Y2Y"] = 0.0
        macro_df["High_Yield_Spread_Z"] = 0.0
        macro_df["Yield_Curve_10Y2Y_Z"] = 0.0

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

    # VIX Extraction
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
    # STEP 1: Returns via TickerEngine Orchestrator
    rets = TickerEngine.map_kernels(df_ohlcv["Adj Close"], QuantUtils.compute_returns)
    autocorr_15 = TickerEngine.map_kernels(
        rets, QuantUtils.calculate_autocorr, lag=1, window=15
    )

    mkt_ret_series = macro_df["Mkt_Ret"]

    # A. Hybrid Metrics (Beta & IR)
    ir_63 = TickerEngine.map_kernels(
        rets,
        QuantUtils.calculate_rolling_ir,
        benchmark_rets=mkt_ret_series,
        window=win_63d,
    )

    beta_63 = TickerEngine.map_kernels(
        rets,
        QuantUtils.calculate_rolling_beta,
        benchmark_rets=mkt_ret_series,
        window=win_63d,
    )

    # B. Volatility (ATR / TRP)
    def get_ticker_vol(df_slice):
        h, l, c = df_slice["Adj High"], df_slice["Adj Low"], df_slice["Adj Close"]
        return pd.DataFrame(
            {
                "TR_Raw": QuantUtils.calculate_tr(h, l, c),
                "ATR_Smooth": QuantUtils.calculate_atr(h, l, c, atr_period),
            },
            index=df_slice.index,
        )

    vol_bundle = TickerEngine.map_kernels(df_ohlcv, get_ticker_vol)
    atr = vol_bundle["ATR_Smooth"]
    natr = (atr / df_ohlcv["Adj Close"]).fillna(0)
    trp = (vol_bundle["TR_Raw"] / df_ohlcv["Adj Close"]).fillna(0)

    # C. Momentum & Consistency (Fixed to use TickerEngine to guarantee MultiIndex)
    mom_21 = TickerEngine.map_kernels(
        df_ohlcv["Adj Close"], lambda x: x.pct_change(win_21d)
    )

    consistency = TickerEngine.map_kernels(
        rets, lambda x: (x > 0).astype(float).rolling(win_5d).mean()
    )

    dd_21 = TickerEngine.map_kernels(
        df_ohlcv["Adj Close"], lambda x: (x / x.rolling(win_21d).max()) - 1.0
    )

    # STEP 2: RSI
    rsi = TickerEngine.map_kernels(
        df_ohlcv["Adj Close"], QuantUtils.calculate_rsi, period=rsi_period
    )

    def get_range_pos_kernel(df_slice):
        rp = QuantUtils.calculate_range_pos(
            df_slice["Adj High"],
            df_slice["Adj Low"],
            df_slice["Adj Close"],
            window=GLOBAL_SETTINGS.get("range_pos_period", 20),
        )
        # Wrap in DataFrame to prevent Pandas from pivoting the output
        return pd.DataFrame({"RP": rp})

    range_pos_20 = TickerEngine.map_kernels(df_ohlcv, get_range_pos_kernel)["RP"]

    def get_obv_kernel(df_slice):
        v = df_slice["Volume"]
        v_baseline = v.rolling(window=win_63d, min_periods=1).mean().replace(0, 1e-8)
        v_rel = v / v_baseline
        obv_val = QuantUtils.calculate_obv_fast(df_slice["Adj Close"], v_rel)
        # Wrap in DataFrame to prevent Pandas from pivoting the output
        return pd.DataFrame({"OBV": obv_val})

    obv = TickerEngine.map_kernels(df_ohlcv, get_obv_kernel)["OBV"]

    log_price = np.log(df_ohlcv["Adj Close"].replace(0, 1e-8))
    slope_p = TickerEngine.map_kernels(
        log_price, QuantUtils.calculate_rolling_slope_5d_fast
    )
    slope_v = TickerEngine.map_kernels(obv, QuantUtils.calculate_rolling_slope_5d_fast)

    convexity = TickerEngine.map_kernels(
        slope_p, QuantUtils.calculate_convexity_5d_fast
    )

    # E. Assemble Features
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
            "AutoCorr_15": autocorr_15,
            "Ret_1d": rets,
            "Range_Pos_20": range_pos_20,
            "Slope_P_5": slope_p,
            "Slope_V_5": slope_v,
            "Convexity": convexity,
        }
    )

    # F. Quality (Universe Filtering) - Fixed to use safe alignment
    quality_temp = pd.DataFrame(
        {
            "IsStale": np.where(
                (df_ohlcv["Volume"] == 0)
                | (df_ohlcv["Adj High"] == df_ohlcv["Adj Low"]),
                1,
                0,
            ),
            "DollarVolume": df_ohlcv["Adj Close"] * df_ohlcv["Volume"],
        },
        index=df_ohlcv.index,
    )

    # Calculate HasSameVolume safely using TickerEngine
    quality_temp["HasSameVolume"] = TickerEngine.map_kernels(
        df_ohlcv["Volume"], lambda x: (x.diff() == 0).astype(int)
    )

    def get_quality(slice_df):
        return pd.DataFrame(
            {
                "RollingStalePct": slice_df["IsStale"]
                .rolling(window=quality_window, min_periods=quality_min_periods)
                .mean(),
                "RollMedDollarVol": slice_df["DollarVolume"]
                .rolling(window=quality_window, min_periods=quality_min_periods)
                .median(),
                "RollingSameVolCount": slice_df["HasSameVolume"]
                .rolling(window=quality_window, min_periods=quality_min_periods)
                .sum(),
            },
            index=slice_df.index,
        )

    rolling_quality = TickerEngine.map_kernels(quality_temp, get_quality)

    return pd.concat([features_df, rolling_quality], axis=1).sort_index(), macro_df
