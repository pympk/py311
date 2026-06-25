import pandas as pd
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
        df_indices: Optional[pd.DataFrame],  # Allow None
        df_fed: Optional[pd.DataFrame],  # Allow None
        config: TradingConfig,  # Keep as mandatory TradingConfig
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
        else:
            macro_df["Mkt_Ret"] = 0.0
            macro_df["Macro_Trend"] = 0.0

        # 2. FED Data Integration
        if df_fed is not None:
            fed_data = (
                df_fed.reindex(all_dates).ffill().bfill().infer_objects(copy=False)
            )
            macro_df["High_Yield_Spread"] = fed_data["High_Yield_Spread"]
            macro_df["Yield_Curve_10Y2Y"] = fed_data["Yield_Curve_10Y2Y"]

            for col in ["High_Yield_Spread", "Yield_Curve_10Y2Y"]:
                roll_mean = macro_df[col].rolling(252, min_periods=60).mean()
                roll_std = macro_df[col].rolling(252, min_periods=60).std()
                macro_df[f"{col}_Z"] = (
                    ((macro_df[col] - roll_mean) / roll_std)
                    .clip(-config.feature_zscore_clip, config.feature_zscore_clip)
                    .fillna(0.0)
                )
        else:
            macro_df["High_Yield_Spread"] = 0.0
            macro_df["Yield_Curve_10Y2Y"] = 0.0
            macro_df["High_Yield_Spread_Z"] = 0.0
            macro_df["Yield_Curve_10Y2Y_Z"] = 0.0

        # 3. Trend Velocity & Momentum
        win_21 = getattr(config, "win_21d", 21)
        win_63 = getattr(config, "win_63d", 63)

        macro_df["Macro_Trend_Vel"] = macro_df["Macro_Trend"].diff(win_21)
        macro_df["Macro_Trend_Vel_Z"] = (
            macro_df["Macro_Trend_Vel"] / macro_df["Macro_Trend"].rolling(win_63).std()
        ).clip(-config.feature_zscore_clip, config.feature_zscore_clip)

        macro_df["Macro_Trend_Mom"] = (
            np.sign(macro_df["Macro_Trend"])
            * np.sign(macro_df["Macro_Trend_Vel"])
            * np.abs(macro_df["Macro_Trend_Vel"])
        )

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
                    (v - v.rolling(63).mean()) / v.rolling(63).std()
                ).clip(-config.feature_zscore_clip, config.feature_zscore_clip)
            if "^VIX" in idx_names and "^VIX3M" in idx_names:
                v3 = (
                    df_indices.xs("^VIX3M", level=0)["Adj Close"]
                    .reindex(all_dates)
                    .ffill()
                )
                macro_df["Macro_Vix_Ratio"] = (v / v3).fillna(1.0)

        macro_df.fillna(0.0, inplace=True)
        return macro_df


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

        # 1. Returns via TickerEngine Orchestrator
        rets = TickerEngine.map_kernels(
            df_ohlcv["Adj Close"], QuantUtils.compute_returns
        )
        autocorr_15 = TickerEngine.map_kernels(
            rets, QuantUtils.calculate_autocorr, lag=1, window=15
        )
        mkt_ret_series = macro_df["Mkt_Ret"]

        # 2. Hybrid Metrics (Beta & IR)
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

        # 3. Volatility (ATR / TRP)
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

        # 4. Momentum & Consistency
        mom_21 = TickerEngine.map_kernels(
            df_ohlcv["Adj Close"], lambda x: x.pct_change(win_21, fill_method=None)
        )
        consistency = TickerEngine.map_kernels(
            rets, lambda x: (x > 0).astype(float).rolling(win_5).mean()
        )
        dd_21 = TickerEngine.map_kernels(
            df_ohlcv["Adj Close"], lambda x: (x / x.rolling(win_21).max()) - 1.0
        )

        # 5. RSI
        rsi = TickerEngine.map_kernels(
            df_ohlcv["Adj Close"], QuantUtils.calculate_rsi, period=rsi_period
        )

        # 6. Range Position
        def get_range_pos_kernel(df_slice):
            rp = QuantUtils.calculate_range_pos(
                df_slice["Adj High"],
                df_slice["Adj Low"],
                df_slice["Adj Close"],
                window=range_pos_period,
            )
            return pd.DataFrame({"RP": rp})

        range_pos_20 = TickerEngine.map_kernels(df_ohlcv, get_range_pos_kernel)["RP"]

        # 7. OBV and Physics
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

        return pd.DataFrame(
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
    # Set these to Optional with = None so they can be omitted in tests
    df_indices: Optional[pd.DataFrame] = None,
    df_fed: Optional[pd.DataFrame] = None,
    config: Optional[TradingConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Orchestrates the independent feature pipelines."""
    actual_config = config if config is not None else TradingConfig()

    print(
        f"[EXEC] Generating Decoupled Features (Benchmark: {actual_config.benchmark_ticker})..."
    )

    # PREP
    df_ohlcv = df_ohlcv.sort_index(level=["Ticker", "Date"])

    # RUN PIPELINES
    macro_df = MacroFeaturePipeline.process(df_ohlcv, df_indices, df_fed, actual_config)
    micro_df = MicroFeaturePipeline.process(df_ohlcv, macro_df, actual_config)
    quality_df = QualityFilterPipeline.process(df_ohlcv, actual_config)

    # ASSEMBLE
    features_df = pd.concat([micro_df, quality_df], axis=1).sort_index()
    return features_df, macro_df
