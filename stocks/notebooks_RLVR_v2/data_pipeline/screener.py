import pandas as pd
import logging
from typing import List, Optional, cast

from core.contracts import MarketObservation, EngineInput
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

        # Wrap index variables in **`int()`**. This converts the NumPy integer into a standard Python integer
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

        mask = (
            (day_features["RollMedDollarVol"] >= vol_cutoff)
            & (day_features["RollingStalePct"] <= thresholds.max_stale_pct)
            & (day_features["RollingSameVolCount"] <= thresholds.max_same_vol_count)
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

            # Add .reindex(candidates) to force alignment with the input order
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

            # Cast the result of .loc to a Series so Pylance knows it's a single row
            macro_snapshot = cast(pd.Series, self.macro_df.loc[decision_date])

            lookback_close = self.df_close.loc[full_window_dates, candidates]

            return MarketObservation(
                lookback_close=lookback_close,
                lookback_returns=lookback_close.ffill().pct_change(),
                atrp=obs_atrp,
                trp=obs_trp,
                # Use square brackets for guaranteed columns.
                # This returns a pd.Series (one value per candidate).
                atr=feat_now["ATR"],
                rsi=feat_now["RSI"],
                # If a column might be missing, use .get() with a default and cast
                # consistency=cast(pd.Series, feat_now.get("Consistency", 0.0)),
                consistency=feat_now["Consistency"],
                mom_21=feat_now["Mom_21"],
                ir_63=feat_now["IR_63"],
                beta_63=feat_now["Beta_63"],
                dd_21=feat_now["DD_21"],
                autocorr_15=feat_now["AutoCorr_15"],
                range_pos_20=feat_now["Range_Pos_20"],
                slope_p_5=feat_now["Slope_P_5"],
                slope_v_5=feat_now["Slope_V_5"],
                convexity=feat_now["Convexity"],
                # For macro_snapshot (a Series), indexing returns a scalar.
                # Use float() to ensure Pylance knows it's a scalar float.
                macro_trend=float(macro_snapshot["Macro_Trend"]),
                macro_trend_vel=float(macro_snapshot["Macro_Trend_Vel"]),
                macro_vix_z=float(macro_snapshot["Macro_Vix_Z"]),
                macro_vix_ratio=float(macro_snapshot["Macro_Vix_Ratio"]),
            )

        except Exception as e:
            raise ValueError(f"[ERROR] Data Assembly Error: {str(e)}")
