import pandas as pd
import numpy as np

from typing import Tuple, Optional

from core.quant import QuantUtils, TickerEngine
from core.settings import TradingConfig

pd.set_option("future.no_silent_downcasting", True)


class MacroFeaturePipeline:
    @staticmethod
    def process(
        df_ohlcv: pd.DataFrame,
        df_indices: Optional[pd.DataFrame],
        df_fed: Optional[pd.DataFrame],
        config: TradingConfig,
    ) -> pd.DataFrame:
        all_dates = df_ohlcv.index.get_level_values("Date").unique().sort_values()
        macro_df = pd.DataFrame(index=all_dates)

        # 1. Benchmark & Trend
        if config.benchmark_ticker in df_ohlcv.index.get_level_values("Ticker"):
            mkt_close = (
                df_ohlcv.xs(config.benchmark_ticker, level="Ticker")["Adj Close"]
                .reindex(all_dates)
                .ffill()
            )
            macro_df["Mkt_Ret"] = mkt_close.pct_change().fillna(0.0)
            macro_df["Macro_Trend"] = (mkt_close / mkt_close.rolling(200).mean()) - 1.0

            # NEW: Z-Scored Market Metrics
            mkt_roll_mean = macro_df["Mkt_Ret"].rolling(63, min_periods=21).mean()
            mkt_roll_std = (
                macro_df["Mkt_Ret"].rolling(63, min_periods=21).std().replace(0, 1e-8)
            )
            macro_df["Mkt_Ret_Z"] = (
                ((macro_df["Mkt_Ret"] - mkt_roll_mean) / mkt_roll_std)
                .clip(-config.feature_zscore_clip, config.feature_zscore_clip)
                .fillna(0.0)
            )

            trend_roll_mean = (
                macro_df["Macro_Trend"].rolling(252, min_periods=60).mean()
            )
            trend_roll_std = (
                macro_df["Macro_Trend"]
                .rolling(252, min_periods=60)
                .std()
                .replace(0, 1e-8)
            )
            macro_df["Macro_Trend_Z"] = (
                ((macro_df["Macro_Trend"] - trend_roll_mean) / trend_roll_std)
                .clip(-config.feature_zscore_clip, config.feature_zscore_clip)
                .fillna(0.0)
            )

            mkt_vol = macro_df["Mkt_Ret"].rolling(63, min_periods=21).std()
            vol_roll_mean = mkt_vol.rolling(252, min_periods=60).mean()
            vol_roll_std = mkt_vol.rolling(252, min_periods=60).std().replace(0, 1e-8)
            macro_df["Mkt_Vol_63d_Z"] = (
                ((mkt_vol - vol_roll_mean) / vol_roll_std)
                .clip(-config.feature_zscore_clip, config.feature_zscore_clip)
                .fillna(0.0)
            )
        else:
            macro_df["Mkt_Ret"] = 0.0
            macro_df["Macro_Trend"] = 0.0
            macro_df["Mkt_Ret_Z"] = 0.0
            macro_df["Macro_Trend_Z"] = 0.0
            macro_df["Mkt_Vol_63d_Z"] = 0.0

        # 2. FED Data Integration
        if df_fed is not None:
            fed_data = (
                df_fed.reindex(all_dates).ffill().bfill().infer_objects(copy=False)
            )
            for col in ["High_Yield_Spread", "Yield_Curve_10Y2Y"]:
                roll_mean = fed_data[col].rolling(252, min_periods=60).mean()
                roll_std = (
                    fed_data[col].rolling(252, min_periods=60).std().replace(0, 1e-8)
                )
                macro_df[f"{col}_Z"] = (
                    ((fed_data[col] - roll_mean) / roll_std)
                    .clip(-config.feature_zscore_clip, config.feature_zscore_clip)
                    .fillna(0.0)
                )
        else:
            macro_df["High_Yield_Spread_Z"] = 0.0
            macro_df["Yield_Curve_10Y2Y_Z"] = 0.0

        # 3. Trend Velocity & Momentum
        win_21 = getattr(config, "win_21d", 21)
        win_63 = getattr(config, "win_63d", 63)

        vel = macro_df["Macro_Trend"].diff(win_21)
        macro_df["Macro_Trend_Vel_Z"] = (
            (vel / macro_df["Macro_Trend"].rolling(win_63).std().replace(0, 1e-8))
            .clip(-config.feature_zscore_clip, config.feature_zscore_clip)
            .fillna(0.0)
        )

        # Scaled Momentum
        momentum_raw = (
            np.sign(macro_df["Macro_Trend"])
            * np.sign(vel)
            * np.abs(macro_df["Macro_Trend_Vel_Z"])
        )
        macro_df["Macro_Trend_Mom"] = pd.Series(
            momentum_raw, index=macro_df.index
        ).fillna(0.0)

        # 4. VIX Extraction
        macro_df["Macro_Vix_Z"] = 0.0
        macro_df["Macro_Vix_Ratio"] = 1.0

        if df_indices is not None:
            idx_names = df_indices.index.get_level_values(0).unique()
            if "^VIX" in idx_names:
                v = (
                    df_indices.xs("^VIX", level=0)["Adj Close"]
                    .reindex(all_dates)
                    .ffill()
                )
                macro_df["Macro_Vix_Z"] = (
                    ((v - v.rolling(63).mean()) / v.rolling(63).std().replace(0, 1e-8))
                    .clip(-config.feature_zscore_clip, config.feature_zscore_clip)
                    .fillna(0.0)
                )
            if "^VIX" in idx_names and "^VIX3M" in idx_names:
                v3 = (
                    df_indices.xs("^VIX3M", level=0)["Adj Close"]
                    .reindex(all_dates)
                    .ffill()
                )
                macro_df["Macro_Vix_Ratio"] = (
                    (v / v3)
                    .fillna(1.0)
                    .clip(-config.feature_ratio_clip, config.feature_ratio_clip)
                )

        macro_df.fillna(0.0, inplace=True)

        # FINAL GUARD: Strictly return EXACTLY 11 columns to preserve the 33-Dim Space!
        # Drops unscaled noise automatically.
        final_11_cols = [
            "Mkt_Ret",
            "Mkt_Ret_Z",
            "Macro_Trend",
            "Macro_Trend_Z",
            "High_Yield_Spread_Z",
            "Yield_Curve_10Y2Y_Z",
            "Macro_Trend_Vel_Z",
            "Macro_Trend_Mom",
            "Macro_Vix_Z",
            "Macro_Vix_Ratio",
            "Mkt_Vol_63d_Z",
        ]
        return macro_df[final_11_cols]


class MicroFeaturePipeline:
    @staticmethod
    def process(
        df_ohlcv: pd.DataFrame, macro_df: pd.DataFrame, config: TradingConfig
    ) -> pd.DataFrame:
        win_5 = getattr(config, "win_5d", 5)
        win_21 = getattr(config, "win_21d", 21)
        win_63 = getattr(config, "win_63d", 63)
        atr_period = getattr(config, "atr_period", 14)
        rsi_period = getattr(config, "rsi_period", 14)
        range_pos_period = getattr(config, "range_pos_period", 20)

        rets = TickerEngine.map_kernels(
            df_ohlcv["Adj Close"], QuantUtils.compute_returns
        )
        autocorr_15 = TickerEngine.map_kernels(
            rets, QuantUtils.calculate_autocorr, lag=1, window=15
        )
        mkt_ret_series = macro_df["Mkt_Ret"]

        ir_63 = TickerEngine.map_kernels(
            rets,
            QuantUtils.calculate_rolling_ir,
            benchmark_rets=mkt_ret_series,
            window=win_63,
        )
        beta_63 = TickerEngine.map_kernels(
            rets,
            QuantUtils.calculate_rolling_beta,
            benchmark_rets=mkt_ret_series,
            window=win_63,
        )

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

        mom_21 = TickerEngine.map_kernels(
            df_ohlcv["Adj Close"], lambda x: x.pct_change(win_21, fill_method=None)
        )
        consistency = TickerEngine.map_kernels(
            rets, lambda x: (x > 0).astype(float).rolling(win_5).mean()
        )
        dd_21 = TickerEngine.map_kernels(
            df_ohlcv["Adj Close"], lambda x: (x / x.rolling(win_21).max()) - 1.0
        )

        # FIX 1: THE RSI BOMB. Normalize 0-100 down to strictly [-1.0, 1.0].
        rsi_raw = TickerEngine.map_kernels(
            df_ohlcv["Adj Close"], QuantUtils.calculate_rsi, period=rsi_period
        )
        rsi_scaled = (rsi_raw - 50.0) / 50.0

        def get_range_pos_kernel(df_slice):
            rp = QuantUtils.calculate_range_pos(
                df_slice["Adj High"],
                df_slice["Adj Low"],
                df_slice["Adj Close"],
                window=range_pos_period,
            )
            return pd.DataFrame({"RP": rp})

        range_pos_20 = TickerEngine.map_kernels(df_ohlcv, get_range_pos_kernel)["RP"]

        def get_obv_kernel(df_slice):
            v = df_slice["Volume"]
            v_baseline = v.rolling(window=win_63, min_periods=1).mean().replace(0, 1e-8)
            v_rel = v / v_baseline
            obv_val = QuantUtils.calculate_obv_fast(df_slice["Adj Close"], v_rel)
            return pd.DataFrame({"OBV": obv_val})

        obv = TickerEngine.map_kernels(df_ohlcv, get_obv_kernel)["OBV"]
        log_price = np.log(df_ohlcv["Adj Close"].replace(0, 1e-8))

        slope_p = TickerEngine.map_kernels(
            log_price, QuantUtils.calculate_rolling_slope_5d_fast
        )
        slope_v = TickerEngine.map_kernels(
            obv, QuantUtils.calculate_rolling_slope_5d_fast
        )
        convexity = TickerEngine.map_kernels(
            slope_p, QuantUtils.calculate_convexity_5d_fast
        )

        # FIX 2: THE OBV DEAD NODE. Provide temporal rolling Z-Scores to prevent lookahead bias
        def temporal_zscore(series, window=win_63):
            roll_mean = series.rolling(window, min_periods=10).mean()
            roll_std = series.rolling(window, min_periods=10).std().replace(0, 1e-8)
            return (series - roll_mean) / roll_std

        slope_p_z = TickerEngine.map_kernels(slope_p, temporal_zscore, window=win_63)
        slope_v_z = TickerEngine.map_kernels(slope_v, temporal_zscore, window=win_63)

        return pd.DataFrame(
            {
                "ATR": atr,
                "ATRP": natr,
                "TRP": trp,
                "RSI": rsi_scaled,  # Uses bounded RSI
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
                "Slope_P_5_Z": slope_p_z,  # Exported for Registry Blueprint
                "Slope_V_5_Z": slope_v_z,  # Exported for Registry Blueprint
                "Convexity": convexity,
            }
        )


class QualityFilterPipeline:
    @staticmethod
    def process(df_ohlcv: pd.DataFrame, config: TradingConfig) -> pd.DataFrame:
        quality_window = getattr(config, "quality_window", 21)
        quality_min_periods = getattr(config, "quality_min_periods", 10)

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

        return TickerEngine.map_kernels(quality_temp, get_quality)


def generate_features(
    df_ohlcv: pd.DataFrame,
    df_indices: Optional[pd.DataFrame] = None,
    df_fed: Optional[pd.DataFrame] = None,
    config: Optional[TradingConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    actual_config = config if config is not None else TradingConfig()
    print(
        f"[EXEC] Generating Decoupled Features (Benchmark: {actual_config.benchmark_ticker})..."
    )

    df_ohlcv = df_ohlcv.sort_index(level=["Ticker", "Date"])

    macro_df = MacroFeaturePipeline.process(df_ohlcv, df_indices, df_fed, actual_config)
    micro_df = MicroFeaturePipeline.process(df_ohlcv, macro_df, actual_config)
    quality_df = QualityFilterPipeline.process(df_ohlcv, actual_config)

    features_df = pd.concat([micro_df, quality_df], axis=1).sort_index()

    print("[EXEC] Sanitizing features to prevent RL Environment crashes...")
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    features_df = features_df.groupby(level="Ticker").ffill()
    features_df = features_df.fillna(0.0)

    return features_df, macro_df
