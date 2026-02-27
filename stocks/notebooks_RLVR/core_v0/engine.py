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

        self.df_ohlcv_raw = df_ohlcv
        self.features_df = features_df
        self.macro_df = macro_df

        # We call a helper to do the "dirty work"
        self._prepare_data(df_close_wide, df_atrp_wide, df_trp_wide, master_ticker)

        self.Result = Result

    def _prepare_data(self, df_close_wide, df_atrp_wide, df_trp_wide, master_ticker):
        self.df_close = (
            df_close_wide
            if df_close_wide is not None
            else self.df_ohlcv_raw["Adj Close"].unstack(level=0)
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

        common_idx = self.df_close.index
        common_cols = self.df_close.columns
        self.df_atrp = self.df_atrp.reindex(index=common_idx, columns=common_cols)
        self.df_trp = self.df_trp.reindex(index=common_idx, columns=common_cols)

        if GLOBAL_SETTINGS["handle_zeros_as_nan"]:
            self.df_close = self.df_close.replace(0, np.nan)

        self.df_close = self.df_close.ffill(limit=GLOBAL_SETTINGS["max_data_gap_ffill"])

        # Don't forget this one:
        self.df_close = self.df_close.fillna(GLOBAL_SETTINGS["nan_price_replacement"])

        if master_ticker not in self.df_close.columns:
            master_ticker = self.df_close.columns[0]

        self.trading_calendar = (
            self.df_close[master_ticker].dropna().index.unique().sort_values()
        )

    # --- Methods to be moved/refactored ---

    def _validate_timeline(self, inputs: EngineInput):
        cal = self.trading_calendar
        last_idx = len(cal) - 1

        if len(cal) <= inputs.lookback_period:
            raise ValueError(
                f"❌ Dataset too small. Need > {inputs.lookback_period} days."
            )

        min_decision_date = cal[inputs.lookback_period]
        if inputs.start_date < min_decision_date:
            # NO 'return self.Result' needed here
            raise ValueError(
                f"❌ Not enough history for a {inputs.lookback_period}-day lookback.\n"
                f"Earliest valid Decision Date: {min_decision_date.date()}"
            )

        required_future_days = 1 + inputs.holding_period
        latest_valid_idx = last_idx - required_future_days

        if latest_valid_idx < 0:
            raise ValueError(
                f"❌ Holding period too long. {inputs.holding_period} days exceeds data."
            )

        if inputs.start_date > cal[latest_valid_idx]:
            latest_date = cal[latest_valid_idx].date()
            raise ValueError(f"❌ Decision Date too late. Latest valid: {latest_date}")

        decision_idx = cal.searchsorted(inputs.start_date)
        if decision_idx > latest_valid_idx:
            decision_idx = latest_valid_idx

        start_idx = decision_idx - inputs.lookback_period
        entry_idx = decision_idx + 1
        end_idx = entry_idx + inputs.holding_period

        # SUCCESS PATH: Just return the tuple
        return cal[start_idx], cal[decision_idx], cal[entry_idx], cal[end_idx]

    def _build_observation(self, decision_date, candidates, start_date):

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

            # NEW: Return the dictionary directly
            return obs

        except Exception as e:
            # NEW: Raise error directly
            raise ValueError(f"❌ Data Assembly Error: {str(e)}")

    def _execute_strategy(self, observation: MarketObservation, metric_name: str):
        if metric_name not in METRIC_REGISTRY:
            # NEW: Raise error directly
            raise ValueError(f"❌ Strategy '{metric_name}' not found.")

        try:
            scores = METRIC_REGISTRY[metric_name](observation)
            if scores.empty:
                raise ValueError(f"❌ Strategy '{metric_name}' returned no scores.")

            # NEW: Return data directly
            return scores
        except Exception as e:
            raise ValueError(f"❌ Math Error in '{metric_name}': {str(e)}")

    def _rank_and_slice(self, raw_scores, inputs, observation):
        # 1. Strip out the NaNs
        clean_scores = raw_scores.dropna()
        dropped_tickers = raw_scores[raw_scores.isna()].index.tolist()

        if clean_scores.empty:
            # FIX: Raise instead of returning Result
            raise ValueError(
                f"❌ All strategy scores are NaN. Dropped: {dropped_tickers}"
            )

        # 2. Rank and Slice
        sorted_tickers = clean_scores.sort_values(ascending=False)
        start_idx = max(0, inputs.rank_start - 1)
        end_idx = inputs.rank_end
        selected_tickers = sorted_tickers.iloc[start_idx:end_idx].index.tolist()

        if not selected_tickers:
            # FIX: Raise instead of returning Result
            raise ValueError(
                f"❌ Ranking returned zero tickers for range {inputs.rank_start}-{inputs.rank_end}."
            )

        try:
            # Build Debug Table
            debug_artifact = pd.DataFrame(
                {
                    "Strategy_Score": raw_scores,
                    "Lookback_Return_Ann": observation["lookback_returns"].mean() * 252,
                    "Lookback_ATRP": observation["atrp"],
                }
            )
            debug_artifact["Was_Dropped"] = debug_artifact.index.isin(dropped_tickers)

            # Build Results Table
            results_table = pd.DataFrame(
                {
                    "Rank": range(
                        inputs.rank_start, inputs.rank_start + len(selected_tickers)
                    ),
                    "Ticker": selected_tickers,
                    "Strategy Value": sorted_tickers.loc[selected_tickers].values,
                }
            ).set_index("Ticker")

        except Exception as e:
            # FIX: Catch math errors during table building
            raise ValueError(f"❌ Error assembling ranking tables: {str(e)}")

        # 3. Success: Return the dictionary directly
        return {
            "tickers": selected_tickers,
            "table": results_table,
            "debug": {
                "full_universe_ranking": debug_artifact,
                "meta": {
                    "dropped_count": len(dropped_tickers),
                    "dropped_tickers": dropped_tickers,
                    "clean_count": len(clean_scores),
                    "selection_range": f"{inputs.rank_start}-{inputs.rank_end}",
                },
            },
        }

    def _select_tickers(self, inputs, start_date, decision_date):
        debug_dict = {}
        audit_info = {}  # We collect info here regardless of the flag

        # 1. Manual List Mode
        if inputs.mode == "Manual List":
            valid = [t for t in inputs.manual_tickers if t in self.df_close.columns]
            # We move the debug update here
            if inputs.debug:
                debug_dict["audit_liquidity"] = {
                    "mode": "Manual",
                    "tickers_passed": len(valid),
                }
            return {
                "tickers": valid,
                "table": pd.DataFrame(index=valid),
                "debug": debug_dict,
            }

        # 2. Filter Universe
        if inputs.universe_subset is not None:
            candidates = [
                t for t in inputs.universe_subset if t in self.df_close.columns
            ]

            # Put your specific debug block back here:
            if inputs.debug:
                debug_dict["audit_liquidity"] = {
                    "mode": "Cascade",
                    "tickers_passed": len(candidates),
                    "forced_list": True,
                }
        else:
            # This method fills 'audit_info' internally
            candidates = self._filter_universe(
                decision_date, inputs.quality_thresholds, audit_info
            )
            print(
                f"DEBUG: {len(candidates)} stocks passed filters on {decision_date.date()}"
            )

            if inputs.debug:
                debug_dict["audit_liquidity"] = audit_info

        if not candidates:
            raise ValueError("❌ No survivors.")

        # CLEAN: No more .ok checks!
        # (Make sure _build_observation returns the dict directly, not a Result)
        obs = self._build_observation(decision_date, candidates, start_date)

        # CLEAN: No more .ok checks!
        # (Make sure _execute_strategy returns the scores directly)
        scores = self._execute_strategy(obs, inputs.metric)

        # CLEAN: No more .ok checks!
        # (Make sure _rank_and_slice returns the dict directly)
        rank_results = self._rank_and_slice(scores, inputs, obs)

        # 3. Final Assembly
        rank_results = self._rank_and_slice(scores, inputs, obs)

        # Merge all debug info at the end
        return {
            "tickers": rank_results["tickers"],
            "table": rank_results["table"],
            "debug": {**debug_dict, **rank_results.get("debug", {})},
        }

    #

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

        avail_dates = self.features_df.index.get_level_values("Date").unique()

        # 1. STRICT CHECK: If today isn't in the data, stop.
        if date_ts not in avail_dates:
            print(
                f"DEBUG: {date_ts.date()} missing from features. Returning empty universe."
            )
            return []

        # 2. Since we know date_ts exists, use it directly.
        # No more 'target_date' lookup needed!
        day_features = self.features_df.xs(date_ts, level="Date")

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
                    "date": date_ts,
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

    # def run(self, inputs: EngineInput) -> EngineOutput:
    #     # --- 1. Timeline ---
    #     try:
    #         # CLEAN: No more .ok checks. Tuple unpacking gives us variables directly.
    #         safe_start, safe_decision, safe_buy, safe_end = self._validate_timeline(
    #             inputs
    #         )
    #     except ValueError as e:
    #         # If any 'raise ValueError' triggers inside the timeline, we catch it here.
    #         return self._error_result(str(e))

    #     # --- 2. Selection (We'll clean this up next) ---
    #     res_sel = self._select_tickers(inputs, safe_start, safe_decision)
    #     if not res_sel.ok:
    #         return self._error_result(res_sel.err)

    #     tickers_to_trade = res_sel.val["tickers"]
    #     results_table = res_sel.val["table"]
    #     debug_dict = res_sel.val["debug"]

    #     # --- 2. Unified Performance Loop ---
    #     targets = [
    #         ("p", tickers_to_trade),  # Portfolio
    #         ("b", [inputs.benchmark_ticker]),  # Benchmark
    #     ]

    #     perf_store = {}
    #     all_metrics = {}
    #     verification_slices = {}

    #     for prefix, tks in targets:
    #         try:
    #             # Call to calculate_buy_and_hold_performance (will be in trading/performance.py)
    #             full = calculate_buy_and_hold_performance(
    #                 self.df_close, self.df_atrp, self.df_trp, tks, safe_start, safe_end
    #             )
    #             hold = calculate_buy_and_hold_performance(
    #                 self.df_close, self.df_atrp, self.df_trp, tks, safe_buy, safe_end
    #             )

    #             # SANITY CHECK
    #             if full[0].empty or full[0].isna().all():
    #                 return self._error_result(
    #                     f"❌ No price data found for {prefix} in the selected period."
    #                 )

    #         except Exception as e:
    #             return self._error_result(
    #                 f"❌ Backtest Math Error ({prefix}): {str(e)}"
    #             )

    #         # Call to _calculate_period_metrics (will be in trading/performance.py)
    #         m, slices = self._calculate_period_metrics(
    #             *full, safe_decision, *hold, prefix=prefix
    #         )

    #         all_metrics.update(m)
    #         verification_slices[prefix] = slices
    #         perf_store[prefix] = full

    #     # --- 3. Consolidated Debug Logic ---
    #     if inputs.debug:
    #         # Call to _get_debug_components (will be in this file for now)
    #         portfolio_debug_comps = self._get_debug_components(
    #             tickers_to_trade, safe_start, safe_end
    #         )
    #         benchmark_debug_comps = self._get_debug_components(
    #             [inputs.benchmark_ticker], safe_start, safe_end
    #         )
    #         # Call to _get_normalized_plot_data (will be in this file for now)
    #         normalized_plot_data = self._get_normalized_plot_data(
    #             tickers_to_trade, safe_start, safe_end
    #         )

    #         debug_dict.update(
    #             {
    #                 "inputs_snapshot": inputs,
    #                 "verification": verification_slices,
    #                 "portfolio_raw_components": portfolio_debug_comps,
    #                 "benchmark_raw_components": benchmark_debug_comps,
    #                 "selection_audit": debug_dict.get("full_universe_ranking"),
    #             }
    #         )
    #     else:
    #         normalized_plot_data = (
    #             pd.DataFrame()
    #         )  # Ensure it's not None if debug is False

    #     # Prepare initial weights (will be in a helper/utils file)
    #     initial_weights = self._prepare_initial_weights(tickers_to_trade)

    #     return EngineOutput(
    #         portfolio_series=perf_store["p"][0] if "p" in perf_store else None,
    #         benchmark_series=perf_store["b"][0] if "b" in perf_store else None,
    #         portfolio_atrp_series=(
    #             perf_store["p"][2]
    #             if "p" in perf_store and len(perf_store["p"]) > 2
    #             else None
    #         ),
    #         benchmark_atrp_series=(
    #             perf_store["b"][2]
    #             if "b" in perf_store and len(perf_store["b"]) > 2
    #             else None
    #         ),
    #         portfolio_trp_series=(
    #             perf_store["p"][3]
    #             if "p" in perf_store and len(perf_store["p"]) > 3
    #             else None
    #         ),
    #         benchmark_trp_series=(
    #             perf_store["b"][3]
    #             if "b" in perf_store and len(perf_store["b"]) > 3
    #             else None
    #         ),
    #         normalized_plot_data=normalized_plot_data,
    #         tickers=tickers_to_trade,
    #         initial_weights=initial_weights,
    #         perf_metrics=all_metrics,
    #         results_df=results_table,
    #         start_date=safe_start,
    #         decision_date=safe_decision,
    #         buy_date=safe_buy,
    #         holding_end_date=safe_end,
    #         debug_data=(
    #             debug_dict if inputs.debug else None
    #         ),  # Only include if debug is True
    #         macro_df=self.macro_df,
    #     )

    def run(self, inputs: EngineInput) -> EngineOutput:

        try:
            # 1. Timeline (Clean)
            safe_start, safe_decision, safe_buy, safe_end = self._validate_timeline(
                inputs
            )

            # 2. Selection (Clean - NEW CODE HERE)
            # No more 'if not res_sel.ok'!
            selection = self._select_tickers(inputs, safe_start, safe_decision)

            # Access keys directly from the dictionary
            tickers_to_trade = selection["tickers"]
            results_table = selection["table"]
            debug_dict = selection["debug"]

        except ValueError as e:
            # All errors from timeline OR selection land here
            return self._error_result(str(e))

        # --- 3. Unified Performance Loop ---
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
