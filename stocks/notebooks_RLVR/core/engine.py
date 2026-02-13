import pandas as pd
import numpy as np

from typing import List

from core.result import Result
from core.contracts import EngineInput, EngineOutput, MarketObservation
from core.settings import GLOBAL_SETTINGS
from core.quant import (
    QuantUtils,
    calculate_buy_and_hold_performance,
)
from strategy.registry import METRIC_REGISTRY


class AlphaEngine:

    def __init__(
        self,
        df_ohlcv: pd.DataFrame,
        features_df: pd.DataFrame,
        macro_df: pd.DataFrame,
        df_close_wide: pd.DataFrame = None,
        df_atrp_wide: pd.DataFrame = None,
        df_trp_wide: pd.DataFrame = None,
        master_ticker: str = GLOBAL_SETTINGS["calendar_ticker"],
    ):
        # Initialize with raw data; data processing will be moved to data loaders/processors
        self.df_ohlcv_raw = df_ohlcv
        self.features_df = features_df
        self.macro_df = macro_df
        self.df_close = (
            df_close_wide
            if df_close_wide is not None
            else df_ohlcv["Adj Close"].unstack(level=0)
        )
        self.df_atrp = (
            df_atrp_wide
            if df_atrp_wide is not None
            else self.features_df["ATRP"].unstack(level=0)
        )
        self.df_trp = (
            df_trp_wide
            if df_trp_wide is not None
            else self.features_df["TRP"].unstack(level=0)
        )

        # Alignment & Sanitization (will be refactored into data/processor.py)
        common_idx = self.df_close.index
        common_cols = self.df_close.columns
        self.df_atrp = self.df_atrp.reindex(index=common_idx, columns=common_cols)
        self.df_trp = self.df_trp.reindex(index=common_idx, columns=common_cols)

        if GLOBAL_SETTINGS["handle_zeros_as_nan"]:
            self.df_close = self.df_close.replace(0, np.nan)
        self.df_close = self.df_close.ffill(limit=GLOBAL_SETTINGS["max_data_gap_ffill"])
        self.df_close = self.df_close.fillna(GLOBAL_SETTINGS["nan_price_replacement"])

        if master_ticker not in self.df_close.columns:
            master_ticker = self.df_close.columns[0]
        self.trading_calendar = (
            self.df_close[master_ticker].dropna().index.unique().sort_values()
        )
        # Placeholder for the Result class (will move to core/result.py)
        self.Result = Result

    # --- Methods to be moved/refactored ---
    def _validate_timeline(self, inputs: EngineInput) -> Result:
        # This logic will be moved to core/engine.py or a dedicated timeline module
        cal = self.trading_calendar
        last_idx = len(cal) - 1

        if len(cal) <= inputs.lookback_period:
            return self.Result(
                err=f"❌ Dataset too small.\nNeed > {inputs.lookback_period} days of history."
            )

        min_decision_date = cal[inputs.lookback_period]
        if inputs.start_date < min_decision_date:
            return self.Result(
                err=f"❌ Not enough history for a {inputs.lookback_period}-day lookback.\n"
                f"Earliest valid Decision Date: {min_decision_date.date()}"
            )

        required_future_days = 1 + inputs.holding_period
        latest_valid_idx = last_idx - required_future_days

        if latest_valid_idx < 0:
            return self.Result(
                err=f"❌ Holding period too long.\n{inputs.holding_period} days exceeds available data."
            )

        if inputs.start_date > cal[latest_valid_idx]:
            latest_date = cal[latest_valid_idx].date()
            return self.Result(
                err=f"❌ Decision Date too late for a {inputs.holding_period}-day hold.\n"
                f"Latest valid date: {latest_date}. Please move picker back."
            )

        decision_idx = cal.searchsorted(inputs.start_date)
        if decision_idx > latest_valid_idx:
            decision_idx = latest_valid_idx

        start_idx = decision_idx - inputs.lookback_period
        entry_idx = decision_idx + 1
        end_idx = entry_idx + inputs.holding_period

        dates = (cal[start_idx], cal[decision_idx], cal[entry_idx], cal[end_idx])
        return self.Result(val=dates)  # Success!

    def _build_observation(
        self,
        decision_date: pd.Timestamp,
        candidates: List[str],
        start_date: pd.Timestamp,
    ) -> Result:
        # This will be refactored into data/processor.py or a dedicated ObservationBuilder
        try:
            idx = pd.IndexSlice

            # 1. Feature Window Check
            feat_window = self.features_df.loc[
                idx[candidates, start_date:decision_date], :
            ]
            if feat_window.empty:
                return self.Result(
                    err=f"❌ No feature data found between {start_date.date()} and {decision_date.date()}"
                )

            # 2. Current Snapshot Check (Decision Date)
            if decision_date not in self.features_df.index.get_level_values("Date"):
                return self.Result(
                    err=f"❌ Decision date {decision_date.date()} missing from features database."
                )

            feat_now = self.features_df.xs(decision_date, level="Date").reindex(
                candidates
            )

            # 3. Macro Check
            if decision_date not in self.macro_df.index:
                return self.Result(
                    err=f"❌ Macro data missing for {decision_date.date()}"
                )
            macro_snapshot = self.macro_df.loc[decision_date]

            # 4. Price Window
            lookback_close = self.df_close.loc[start_date:decision_date, candidates]
            if lookback_close.empty:
                return self.Result(err="❌ Price lookback window is empty.")

            # Build the observation dictionary
            obs = {
                "lookback_close": lookback_close,
                "lookback_returns": lookback_close.ffill().pct_change(),
                "atrp": feat_window["ATRP"].groupby(level="Ticker").mean(),
                "trp": feat_window["TRP"].groupby(level="Ticker").mean(),
                "atr": feat_now.get("ATR"),
                "rsi": feat_now.get("RSI"),
                "consistency": feat_now.get("Consistency", 0.0),
                "mom_21": feat_now.get("Mom_21"),
                "ir_63": feat_now.get("IR_63"),
                "beta_63": feat_now.get("Beta_63"),
                "dd_21": feat_now.get("DD_21"),
                "macro_trend": macro_snapshot.get("Macro_Trend"),
                "macro_trend_vel": macro_snapshot.get("Macro_Trend_Vel"),
                "macro_vix_z": macro_snapshot.get("Macro_Vix_Z"),
                "macro_vix_ratio": macro_snapshot.get("Macro_Vix_Ratio"),
            }

            return self.Result(val=obs)

        except Exception as e:
            return self.Result(err=f"❌ Data Assembly Error: {str(e)}")

    def _execute_strategy(
        self, observation: MarketObservation, metric_name: str
    ) -> Result:
        # This will be refactored into strategy/registry.py and potentially strategy/executor.py
        if metric_name not in METRIC_REGISTRY:
            return self.Result(
                err=f"❌ Strategy '{metric_name}' not found in METRIC_REGISTRY."
            )

        try:
            scores = METRIC_REGISTRY[metric_name](observation)
            if scores.empty:
                return self.Result(
                    err=f"❌ Strategy '{metric_name}' returned no scores."
                )
            return self.Result(val=scores)
        except Exception as e:
            return self.Result(err=f"❌ Math Error in '{metric_name}': {str(e)}")

    def _rank_and_slice(self, raw_scores, inputs, observation) -> Result:
        # 1. Strip out the NaNs before ranking
        clean_scores = raw_scores.dropna()
        dropped_tickers = raw_scores[
            raw_scores.isna()
        ].index.tolist()  # Track what we removed

        if clean_scores.empty:
            return Result(
                err="❌ All strategy scores are NaN or missing.",
                debug={
                    "dropped_tickers": dropped_tickers
                },  # Still useful even on failure
            )

        # 2. Rank only the valid data
        sorted_tickers = clean_scores.sort_values(ascending=False)
        start_idx = max(0, inputs.rank_start - 1)
        end_idx = inputs.rank_end
        selected_tickers = sorted_tickers.iloc[start_idx:end_idx].index.tolist()

        if not selected_tickers:
            return self.Result(
                err=f"❌ Ranking returned zero tickers for range {inputs.rank_start}-{inputs.rank_end}.",
                debug={
                    "dropped_tickers": dropped_tickers,
                    "available_count": len(clean_scores),
                },
            )

        try:
            debug_artifact = pd.DataFrame(
                {
                    "Strategy_Score": raw_scores,
                    "Lookback_Return_Ann": observation["lookback_returns"].mean() * 252,
                    "Lookback_ATRP": observation["atrp"],
                }
            )

            # Add visibility into data quality issues
            debug_artifact["Was_Dropped"] = debug_artifact.index.isin(dropped_tickers)

            results_table = pd.DataFrame(
                {
                    "Rank": range(
                        inputs.rank_start, inputs.rank_start + len(selected_tickers)
                    ),
                    "Ticker": selected_tickers,
                    "Strategy Value": sorted_tickers.loc[selected_tickers].values,
                }
            ).set_index("Ticker")

            return self.Result(
                val={
                    "tickers": selected_tickers,
                    "table": results_table,
                    "debug": {
                        "full_universe_ranking": debug_artifact,
                        "meta": {
                            "dropped_count": len(dropped_tickers),
                            "dropped_tickers": dropped_tickers,  # Explicit list
                            "clean_count": len(clean_scores),
                            "selection_range": f"{inputs.rank_start}-{inputs.rank_end}",
                        },
                    },
                }
            )
        except Exception as e:
            return self.Result(
                err=f"❌ Ranking/Slicing Error: {str(e)}",
                debug={"dropped_tickers": dropped_tickers},  # Preserve context on crash
            )

    def _select_tickers(self, inputs, start_date, decision_date):
        # This logic will be moved to core/engine.py or a dedicated selection module
        debug_dict = {}

        if inputs.mode == "Manual List":
            valid = [t for t in inputs.manual_tickers if t in self.df_close.columns]
            return self.Result(
                val={"tickers": valid, "table": pd.DataFrame(index=valid), "debug": {}}
            )

        audit_info = {}
        if inputs.universe_subset is not None:
            candidates = [
                t for t in inputs.universe_subset if t in self.df_close.columns
            ]
            if inputs.debug:
                debug_dict["audit_liquidity"] = {
                    "mode": "Cascade",
                    "tickers_passed": len(candidates),
                    "forced_list": True,
                }
        else:
            candidates = self._filter_universe(
                decision_date, inputs.quality_thresholds, audit_info
            )
            if inputs.debug:
                debug_dict["audit_liquidity"] = audit_info

        if not candidates:
            return self.Result(err="No survivors.")

        res_obs = self._build_observation(decision_date, candidates, start_date)
        if not res_obs.ok:
            return res_obs

        actual_obs = res_obs.val
        res_strat = self._execute_strategy(actual_obs, inputs.metric)
        if not res_strat.ok:
            return res_strat

        actual_scores = res_strat.val
        res_rank = self._rank_and_slice(actual_scores, inputs, actual_obs)
        if not res_rank.ok:
            return res_rank

        sel = res_rank.val["tickers"]
        table = res_rank.val["table"]
        debug_dict.update(res_rank.val["debug"])

        return self.Result(val={"tickers": sel, "table": table, "debug": debug_dict})

    def _get_debug_components(self, tickers, start, end):
        # This will be moved to core/engine.py or a dedicated analyzer module
        idx = pd.IndexSlice
        return {
            "prices": self.df_close[tickers].loc[start:end],
            "atrp": self.df_atrp[tickers].loc[start:end],
            "trp": self.df_trp[tickers].loc[start:end],
            "ohlcv_raw": self.df_ohlcv_raw.loc[idx[tickers, start:end], :],
        }

    def _calculate_period_metrics(
        self,
        f_val,
        f_ret,
        f_atrp,
        f_trp,
        decision_date,
        h_val,
        h_ret,
        h_atrp,
        h_trp,
        prefix,
    ):
        # This will be moved to trading/performance.py
        m, s = {}, {}
        lb_val, lb_ret, lb_atrp, lb_trp = (
            f_val.loc[:decision_date],
            f_ret.loc[:decision_date],
            f_atrp.loc[:decision_date],
            f_trp.loc[:decision_date],
        )

        # LEGACY KEYS
        m[f"full_{prefix}_gain"] = QuantUtils.calculate_gain(f_val)
        m[f"full_{prefix}_sharpe"] = QuantUtils.calculate_sharpe(f_ret)
        m[f"full_{prefix}_sharpe_atrp"] = QuantUtils.calculate_sharpe_vol(f_ret, f_atrp)
        m[f"full_{prefix}_sharpe_trp"] = QuantUtils.calculate_sharpe_vol(f_ret, f_trp)
        m[f"lookback_{prefix}_gain"] = QuantUtils.calculate_gain(lb_val)
        m[f"lookback_{prefix}_sharpe"] = QuantUtils.calculate_sharpe(lb_ret)
        m[f"lookback_{prefix}_sharpe_atrp"] = QuantUtils.calculate_sharpe_vol(
            lb_ret, lb_atrp
        )
        m[f"lookback_{prefix}_sharpe_trp"] = QuantUtils.calculate_sharpe_vol(
            lb_ret, lb_trp
        )
        m[f"holding_{prefix}_gain"] = QuantUtils.calculate_gain(h_val)
        m[f"holding_{prefix}_sharpe"] = QuantUtils.calculate_sharpe(h_ret)
        m[f"holding_{prefix}_sharpe_atrp"] = QuantUtils.calculate_sharpe_vol(
            h_ret, h_atrp
        )
        m[f"holding_{prefix}_sharpe_trp"] = QuantUtils.calculate_sharpe_vol(
            h_ret, h_trp
        )

        # SLICES FOR AUDIT MAP
        s["full_val"], s["full_ret"], s["full_atrp"], s["full_trp"] = (
            f_val,
            f_ret,
            f_atrp,
            f_trp,
        )
        s["lookback_val"], s["lookback_ret"], s["lookback_atrp"], s["lookback_trp"] = (
            lb_val,
            lb_ret,
            lb_atrp,
            lb_trp,
        )
        s["holding_val"], s["holding_ret"], s["holding_atrp"], s["holding_trp"] = (
            h_val,
            h_ret,
            h_atrp,
            h_trp,
        )
        return m, s

    def _filter_universe(self, date_ts, thresholds, audit_container=None):
        # This will be moved to data/processor.py
        avail_dates = (
            self.features_df.index.get_level_values("Date").unique().sort_values()
        )
        target_date = avail_dates[avail_dates <= date_ts][-1]
        day_features = self.features_df.xs(target_date, level="Date")
        vol_cutoff = thresholds.get("min_median_dollar_volume", 0)
        if "min_liquidity_percentile" in thresholds:
            vol_cutoff = max(
                vol_cutoff,
                day_features["RollMedDollarVol"].quantile(
                    thresholds["min_liquidity_percentile"]
                ),
            )

        mask = (
            (day_features["RollMedDollarVol"] >= vol_cutoff)
            & (day_features["RollingStalePct"] <= thresholds["max_stale_pct"])
            & (day_features["RollingSameVolCount"] <= thresholds["max_same_vol_count"])
        )

        if audit_container is not None:
            audit_container.update(
                {
                    "date": target_date,
                    "total_tickers_available": len(day_features),
                    "percentile_setting": thresholds.get(
                        "min_liquidity_percentile", "N/A"
                    ),
                    "final_cutoff_usd": vol_cutoff,
                    "tickers_passed": mask.sum(),
                    "universe_snapshot": day_features.assign(Passed_Final=mask),
                }
            )
        return day_features[mask].index.tolist()

    def _get_normalized_plot_data(self, tickers, start, end):
        # This will be moved to visualization or analysis module
        if not tickers:
            return pd.DataFrame()
        data = self.df_close[list(set(tickers))].loc[start:end]
        return data / data.bfill().iloc[0]

    def _error_result(self, msg) -> EngineOutput:
        # This will be moved to core/engine.py for a consistent error return
        return EngineOutput(
            portfolio_series=None,
            benchmark_series=None,
            normalized_plot_data=None,
            tickers=[],
            initial_weights=None,
            perf_metrics={},
            results_df=pd.DataFrame(),
            start_date=pd.Timestamp.min,
            decision_date=pd.Timestamp.min,
            buy_date=pd.Timestamp.min,
            holding_end_date=pd.Timestamp.min,
            portfolio_atrp_series=None,
            benchmark_atrp_series=None,
            portfolio_trp_series=None,
            benchmark_trp_series=None,
            error_msg=msg,
            debug_data=None,
            macro_df=None,
        )

    # --- End Methods to be moved/refactored ---

    def run(self, inputs: EngineInput) -> EngineOutput:
        # --- 1. Timeline & Selection ---
        # Call to _validate_timeline (which will be in this file for now)
        res_tl = self._validate_timeline(inputs)
        if not res_tl.ok:
            return self._error_result(res_tl.err)  # Use the error handler

        safe_start, safe_decision, safe_buy, safe_end = res_tl.val

        # Call to _select_tickers (which will be in this file for now)
        res_sel = self._select_tickers(inputs, safe_start, safe_decision)
        if not res_sel.ok:
            return self._error_result(res_sel.err)  # Use the error handler

        tickers_to_trade = res_sel.val["tickers"]
        results_table = res_sel.val["table"]
        debug_dict = res_sel.val["debug"]

        # --- 2. Unified Performance Loop ---
        targets = [
            ("p", tickers_to_trade),  # Portfolio
            ("b", [inputs.benchmark_ticker]),  # Benchmark
        ]

        perf_store = {}
        all_metrics = {}
        verification_slices = {}

        for prefix, tks in targets:
            try:
                # Call to calculate_buy_and_hold_performance (will be in trading/performance.py)
                full = calculate_buy_and_hold_performance(
                    self.df_close, self.df_atrp, self.df_trp, tks, safe_start, safe_end
                )
                hold = calculate_buy_and_hold_performance(
                    self.df_close, self.df_atrp, self.df_trp, tks, safe_buy, safe_end
                )

                # SANITY CHECK
                if full[0].empty or full[0].isna().all():
                    return self._error_result(
                        f"❌ No price data found for {prefix} in the selected period."
                    )

            except Exception as e:
                return self._error_result(
                    f"❌ Backtest Math Error ({prefix}): {str(e)}"
                )

            # Call to _calculate_period_metrics (will be in trading/performance.py)
            m, slices = self._calculate_period_metrics(
                *full, safe_decision, *hold, prefix=prefix
            )

            all_metrics.update(m)
            verification_slices[prefix] = slices
            perf_store[prefix] = full

        # --- 3. Consolidated Debug Logic ---
        if inputs.debug:
            # Call to _get_debug_components (will be in this file for now)
            portfolio_debug_comps = self._get_debug_components(
                tickers_to_trade, safe_start, safe_end
            )
            benchmark_debug_comps = self._get_debug_components(
                [inputs.benchmark_ticker], safe_start, safe_end
            )
            # Call to _get_normalized_plot_data (will be in this file for now)
            normalized_plot_data = self._get_normalized_plot_data(
                tickers_to_trade, safe_start, safe_end
            )

            debug_dict.update(
                {
                    "inputs_snapshot": inputs,
                    "verification": verification_slices,
                    "portfolio_raw_components": portfolio_debug_comps,
                    "benchmark_raw_components": benchmark_debug_comps,
                    "selection_audit": debug_dict.get("full_universe_ranking"),
                }
            )
        else:
            normalized_plot_data = (
                pd.DataFrame()
            )  # Ensure it's not None if debug is False

        # Prepare initial weights (will be in a helper/utils file)
        initial_weights = self._prepare_initial_weights(tickers_to_trade)

        return EngineOutput(
            portfolio_series=perf_store["p"][0] if "p" in perf_store else None,
            benchmark_series=perf_store["b"][0] if "b" in perf_store else None,
            portfolio_atrp_series=(
                perf_store["p"][2]
                if "p" in perf_store and len(perf_store["p"]) > 2
                else None
            ),
            benchmark_atrp_series=(
                perf_store["b"][2]
                if "b" in perf_store and len(perf_store["b"]) > 2
                else None
            ),
            portfolio_trp_series=(
                perf_store["p"][3]
                if "p" in perf_store and len(perf_store["p"]) > 3
                else None
            ),
            benchmark_trp_series=(
                perf_store["b"][3]
                if "b" in perf_store and len(perf_store["b"]) > 3
                else None
            ),
            normalized_plot_data=normalized_plot_data,
            tickers=tickers_to_trade,
            initial_weights=initial_weights,
            perf_metrics=all_metrics,
            results_df=results_table,
            start_date=safe_start,
            decision_date=safe_decision,
            buy_date=safe_buy,
            holding_end_date=safe_end,
            debug_data=(
                debug_dict if inputs.debug else None
            ),  # Only include if debug is True
            macro_df=self.macro_df,
        )

    # Helper for initial weights, will move to a utility file
    def _prepare_initial_weights(self, tickers: List[str]) -> pd.Series:
        if not tickers:
            return pd.Series()
        # Simple equal weighting for now
        return pd.Series(1.0 / len(tickers), index=tickers)
