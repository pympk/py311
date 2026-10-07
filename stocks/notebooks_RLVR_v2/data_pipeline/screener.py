import logging
from typing import List, Optional, cast

import numpy as np
import pandas as pd

from core.contracts import EngineInput, MarketObservation
from core.quant import QuantUtils
from core.settings import TradingConfig


class UniverseScreener:
    """
    Handles temporal validation, universe gating, and State Observation construction.
    Isolates data-prep complexity away from execution engines.
    """

    def __init__(
        self,
        df_close: pd.DataFrame,
        features_df: pd.DataFrame,
        macro_df: pd.DataFrame,
        trading_calendar: pd.DatetimeIndex,
        config: TradingConfig,
    ):
        self.df_close = df_close
        self.features_df = features_df
        self.macro_df = macro_df
        self.trading_calendar = trading_calendar
        self.config = config

    def validate_timeline(
        self, inputs: EngineInput
    ) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
        cal = self.trading_calendar
        last_idx = len(cal) - 1

        if len(cal) <= inputs.lookback_period:
            raise ValueError(
                f"[ERROR] Dataset too small. Need > {inputs.lookback_period} days."
            )

        min_decision_date = cal[inputs.lookback_period]
        if inputs.decision_date < min_decision_date:
            raise ValueError(
                f"[ERROR] Not enough history. Earliest valid: {min_decision_date.date()}"
            )

        required_future_days = 1 + inputs.holding_period
        latest_valid_idx = last_idx - required_future_days

        if latest_valid_idx < 0:
            raise ValueError("[ERROR] Holding period too long for available data.")

        if inputs.decision_date > cal[latest_valid_idx]:
            latest_date = cal[latest_valid_idx].date()
            logging.warning(
                f"\n{'='*65}\n"
                f"[WARNING] DATA/UI MISMATCH WARNING\n"
                f"Requested Decision Date: {inputs.decision_date.date()} is not available.\n"
                f"The UI Decision Date input box is showing a date beyond available history.\n"
                f"REPLACING WITH LATEST AVAILABLE DATE: {latest_date}\n"
                f"{'='*65}"
            )
            decision_idx = latest_valid_idx
        else:
            decision_idx = cal.searchsorted(inputs.decision_date)

        start_idx = decision_idx - inputs.lookback_period
        entry_idx = decision_idx + 1
        end_idx = entry_idx + inputs.holding_period

        return (
            cal[int(start_idx)],
            cal[int(decision_idx)],
            cal[int(entry_idx)],
            cal[int(end_idx)],
        )

    def filter_universe(
        self, date_ts: pd.Timestamp, thresholds, audit_container: Optional[dict] = None
    ) -> List[str]:
        avail_dates = self.features_df.index.get_level_values("Date").unique()

        if date_ts not in avail_dates:
            logging.debug(
                f"{date_ts.date()} missing from features. Returning empty universe."
            )
            return []

        day_features = self.features_df.xs(date_ts, level="Date")
        vol_cutoff = thresholds.min_median_dollar_volume

        if thresholds.min_liquidity_percentile is not None:
            vol_cutoff = max(
                vol_cutoff,
                day_features["RollMedDollarVol"].quantile(
                    thresholds.min_liquidity_percentile
                ),
            )

        valid_price_mask = pd.Series(True, index=day_features.index)
        if not self.df_close.empty and date_ts in self.df_close.index:
            prices_today = self.df_close.loc[date_ts]
            valid_price_mask = prices_today.notna() & (prices_today > 0)
            valid_price_mask = valid_price_mask.reindex(day_features.index).fillna(
                False
            )

        mask = (
            (day_features["RollMedDollarVol"] >= vol_cutoff)
            & (day_features["RollingStalePct"] <= thresholds.max_stale_pct)
            & (day_features["RollingSameVolCount"] <= thresholds.max_same_vol_count)
            & (day_features["RecentStaleDays"] < 3)
            & (day_features["IsZeroPrice"] == 0)
            & valid_price_mask
        )

        if audit_container is not None:
            audit_container.update(
                {
                    "date": date_ts,
                    "total_tickers_available": len(day_features),
                    "percentile_setting": thresholds.min_liquidity_percentile,
                    "final_cutoff_usd": vol_cutoff,
                    "tickers_passed": mask.sum(),
                    "universe_snapshot": day_features.assign(Passed_Final=mask),
                }
            )

        return day_features[mask].index.tolist()

    def build_observation(
        self,
        decision_date: pd.Timestamp,
        candidates: List[str],
        start_date: pd.Timestamp,
    ) -> MarketObservation:
        try:
            full_window_dates = self.trading_calendar[
                (self.trading_calendar >= start_date)
                & (self.trading_calendar <= decision_date)
            ]
            active_dates = full_window_dates[1:]

            idx = pd.IndexSlice
            feat_window = self.features_df.loc[idx[candidates, active_dates], :]

            obs_atrp = (
                feat_window["ATRP"].groupby(level="Ticker").mean().reindex(candidates)
            )
            obs_trp = (
                feat_window["TRP"].groupby(level="Ticker").mean().reindex(candidates)
            )

            if decision_date not in self.features_df.index.get_level_values("Date"):
                raise ValueError(
                    f"[ERROR] Decision date {decision_date.date()} missing from features database."
                )

            feat_now = self.features_df.xs(decision_date, level="Date").reindex(
                candidates
            )
            macro_snapshot = cast(pd.Series, self.macro_df.loc[decision_date])

            lookback_close = self.df_close.loc[full_window_dates, candidates]
            lookback_returns = lookback_close.ffill().pct_change(fill_method=None)

            # Benchmark alignment for relative factor calculation
            bm_ticker = getattr(
                self.config,
                "benchmark",
                getattr(self.config, "benchmark_ticker", "SPY"),
            )
            if not self.df_close.empty and bm_ticker in self.df_close.columns:
                bm_close = self.df_close.loc[full_window_dates, bm_ticker]
                bm_rets = bm_close.pct_change(fill_method=None)
            else:
                bm_rets = pd.Series(0.0, index=full_window_dates)

            # -----------------------------------------------------------------
            # GENERATION 16 ORTHOGONAL FACTOR WIRING
            # -----------------------------------------------------------------
            # 1. Residual Momentum (126d)
            if (
                "ResMom_126" in feat_now.columns
                and not feat_now["ResMom_126"].isna().all()
            ):
                obs_res_mom_126 = feat_now["ResMom_126"].fillna(0.0)
            else:
                res_mom_df = QuantUtils.calculate_residual_momentum(
                    lookback_returns, bm_rets, window=126
                )
                obs_res_mom_126 = res_mom_df.iloc[-1].reindex(candidates).fillna(0.0)

            # 2. Range Position (52w High)
            if (
                "Range_Pos_52w" in feat_now.columns
                and not feat_now["Range_Pos_52w"].isna().all()
            ):
                obs_range_pos_52w = feat_now["Range_Pos_52w"].fillna(1.0)
            else:
                range_df = QuantUtils.calculate_range_pos_52w(
                    lookback_close, window=252
                )
                obs_range_pos_52w = range_df.iloc[-1].reindex(candidates).fillna(1.0)

            # 3. Downside Beta (-Beta_Down_63)
            if (
                "Beta_Down_63" in feat_now.columns
                and not feat_now["Beta_Down_63"].isna().all()
            ):
                obs_beta_down_63 = feat_now["Beta_Down_63"].fillna(1.0)
            else:
                beta_down_df = QuantUtils.calculate_downside_beta(
                    lookback_returns, bm_rets, window=63
                )
                obs_beta_down_63 = beta_down_df.iloc[-1].reindex(candidates).fillna(1.0)

            # 4. Efficiency Ratio (ER_63)
            if "ER_63" in feat_now.columns and not feat_now["ER_63"].isna().all():
                obs_er_63 = feat_now["ER_63"].fillna(0.0)
            else:
                er_df = QuantUtils.calculate_efficiency_ratio(lookback_close, window=63)
                obs_er_63 = er_df.iloc[-1].reindex(candidates).fillna(0.0)

            return MarketObservation(
                lookback_close=lookback_close,
                lookback_returns=lookback_returns,
                atrp=obs_atrp,
                trp=obs_trp,
                atr=feat_now["ATR"],
                rsi=feat_now["RSI"],
                ir_63=feat_now["IR_63"],
                dd_21=feat_now["DD_21"],
                mom_126=feat_now["Mom_126"],
                ivol_63=feat_now["IVol_63"],
                mom_252_21=feat_now["Mom_252_21"],
                trend_r2_63=feat_now["Trend_R2_63"],
                macro_trend=float(macro_snapshot["Macro_Trend"]),
                macro_trend_vel=float(macro_snapshot["Macro_Trend_Vel_Z"]),
                macro_vix_z=float(macro_snapshot["Macro_Vix_Z"]),
                macro_vix_ratio=float(macro_snapshot["Macro_Vix_Ratio"]),
                res_mom_126=obs_res_mom_126,
                range_pos_52w=obs_range_pos_52w,
                beta_down_63=obs_beta_down_63,
                er_63=obs_er_63,
                consistency=(
                    feat_now["Consistency"]
                    if "Consistency" in feat_now.columns
                    else pd.Series(0.0, index=candidates)
                ),
                mom_21=(
                    feat_now["Mom_21"]
                    if "Mom_21" in feat_now.columns
                    else pd.Series(0.0, index=candidates)
                ),
                mom_63=(
                    feat_now["Mom_63"]
                    if "Mom_63" in feat_now.columns
                    else pd.Series(0.0, index=candidates)
                ),
                beta_63=(
                    feat_now["Beta_63"]
                    if "Beta_63" in feat_now.columns
                    else pd.Series(1.0, index=candidates)
                ),
                semidev_63=(
                    feat_now["SemiDev_63"]
                    if "SemiDev_63" in feat_now.columns
                    else pd.Series(0.0, index=candidates)
                ),
            )

        except Exception as e:
            raise ValueError(f"[ERROR] Data Assembly Error: {str(e)}")
