import pandas as pd
import numpy as np

from typing import List
from dataclasses import dataclass
from core.result import TaskResult
from core.contracts import (
    MarketObservation,
    SelectionResult,
    EngineInput,
    EngineOutput,
    DiscoveryResult,
)
from core.settings import GLOBAL_SETTINGS
from core.performance import calculate_buy_and_hold_performance, PerformanceCalculator
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

        self.Result = TaskResult

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

    def _build_observation(
        self, decision_date, candidates, start_date
    ) -> MarketObservation:
        try:
            # 1. Identify the full range of dates in the window
            full_window_dates = self.trading_calendar[
                (self.trading_calendar >= start_date)
                & (self.trading_calendar <= decision_date)
            ]

            # 2. Define the 'Active Window' (Everything EXCEPT the first day anchor)
            active_dates = full_window_dates[1:]

            # 3. Slice Features using the Active Window only (Ensures N=21, not 22)
            idx = pd.IndexSlice

            feat_window = self.features_df.loc[idx[candidates, active_dates], :]

            # 4. Calculate Means on the Active Window
            # This ensures SHV Sharpe (ATRP) returns 0.8410
            obs_atrp = feat_window["ATRP"].groupby(level="Ticker").mean()
            obs_trp = feat_window["TRP"].groupby(level="Ticker").mean()

            # 2. Current Snapshot Check (Decision Date)
            if decision_date not in self.features_df.index.get_level_values("Date"):
                return self.Result(
                    err=f"❌ Decision date {decision_date.date()} missing from features database."
                )

            feat_now = self.features_df.xs(decision_date, level="Date").reindex(
                candidates
            )
            macro_snapshot = self.macro_df.loc[decision_date]

            # Ensure lookback_close still uses start_date (P0) so returns have an anchor
            lookback_close = self.df_close.loc[full_window_dates, candidates]

            return MarketObservation(
                lookback_close=lookback_close,
                lookback_returns=lookback_close.pct_change(),  # P0 becomes NaN
                atrp=obs_atrp,
                trp=obs_trp,
                # Snapshot features remain at decision_date
                atr=self.features_df.xs(decision_date, level="Date")
                .reindex(candidates)
                .get("ATR"),
                rsi=feat_now.get("RSI"),
                consistency=feat_now.get("Consistency", 0.0),
                mom_21=feat_now.get("Mom_21"),
                ir_63=feat_now.get("IR_63"),
                beta_63=feat_now.get("Beta_63"),
                dd_21=feat_now.get("DD_21"),
                macro_trend=float(macro_snapshot.get("Macro_Trend", 0)),
                macro_trend_vel=float(macro_snapshot.get("Macro_Trend_Vel", 0)),
                macro_vix_z=float(macro_snapshot.get("Macro_Vix_Z", 0)),
                macro_vix_ratio=float(macro_snapshot.get("Macro_Vix_Ratio", 0)),
            )
        except Exception as e:
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
            debug_artifact = pd.DataFrame(
                {
                    "Strategy_Score": raw_scores,
                    # "Lookback_Return_Ann": observation.lookback_returns.mean() * 252,
                    "Lookback_Return": observation.lookback_returns.mean(),
                    "Lookback_ATRP": observation.atrp,
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

    def _select_tickers(self, inputs, start_date, decision_date) -> SelectionResult:
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
            # CHANGE THIS: Return the Dataclass, not a dict
            return SelectionResult(
                tickers=valid, table=pd.DataFrame(index=valid), debug=debug_dict
            )

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

        # NEW: Return the Dataclass instead of a dict
        return SelectionResult(
            tickers=rank_results["tickers"],
            table=rank_results["table"],
            debug={**debug_dict, **rank_results.get("debug", {})},
        )

    def _get_debug_components(self, tickers, start, end):
        # This will be moved to core/engine.py or a dedicated analyzer module
        idx = pd.IndexSlice
        return {
            "prices": self.df_close[tickers].loc[start:end],
            "atrp": self.df_atrp[tickers].loc[start:end],
            "trp": self.df_trp[tickers].loc[start:end],
            "ohlcv_raw": self.df_ohlcv_raw.loc[idx[tickers, start:end], :],
        }

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

    def _build_engine_output(
        self,
        perf_store,
        tickers,
        results_table,
        debug_dict,
        start,
        decision,
        buy,
        end,
        metrics,
        plot_data,
        inputs,
    ) -> EngineOutput:

        # Helper to safely grab items from the perf_store tuples (Val, Ret, ATRP, TRP)
        p = perf_store.get("p", (None, None, None, None))
        b = perf_store.get("b", (None, None, None, None))

        return EngineOutput(
            portfolio_series=p[0],
            benchmark_series=b[0],
            portfolio_atrp_series=p[2] if len(p) > 2 else None,
            benchmark_atrp_series=b[2] if len(b) > 2 else None,
            portfolio_trp_series=p[3] if len(p) > 3 else None,
            benchmark_trp_series=b[3] if len(b) > 3 else None,
            normalized_plot_data=plot_data,
            tickers=tickers,
            initial_weights=self._prepare_initial_weights(tickers),
            perf_metrics=metrics,
            results_df=results_table,
            start_date=start,
            decision_date=decision,
            buy_date=buy,
            holding_end_date=end,
            debug_data=debug_dict if inputs.debug else None,
            macro_df=self.macro_df,
        )

    def run(self, inputs: EngineInput) -> EngineOutput:
        try:
            # 1. Timeline & Selection
            safe_start, safe_decision, safe_buy, safe_end = self._validate_timeline(
                inputs
            )
            selection = self._select_tickers(inputs, safe_start, safe_decision)

            tickers_to_trade = selection.tickers
            results_table = selection.table
            debug_dict = selection.debug

            # 2. Performance Loop
            targets = [("p", tickers_to_trade), ("b", [inputs.benchmark_ticker])]
            perf_store, all_metrics, verification_slices = {}, {}, {}

            for prefix, tks in targets:

                full = calculate_buy_and_hold_performance(
                    self.df_close, self.df_atrp, self.df_trp, tks, safe_start, safe_end
                )
                hold = calculate_buy_and_hold_performance(
                    self.df_close, self.df_atrp, self.df_trp, tks, safe_buy, safe_end
                )

                if full[0].empty:
                    raise ValueError(
                        f"❌ No price data for {prefix} in selected period."
                    )

                m, slices = PerformanceCalculator.calculate_period_metrics(
                    full, hold, safe_decision, prefix
                )

                all_metrics.update(m)
                verification_slices[prefix] = slices
                perf_store[prefix] = full

            # 3. Debug Logic (Now correctly inside the 'try')

            normalized_plot_data = pd.DataFrame()

            if inputs.debug:
                portfolio_debug_comps = self._get_debug_components(
                    tickers_to_trade, safe_start, safe_end
                )
                benchmark_debug_comps = self._get_debug_components(
                    [inputs.benchmark_ticker], safe_start, safe_end
                )
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

            # SUCCESS: Always return the output (moved outside the 'if' block)
            return self._build_engine_output(
                perf_store,
                tickers_to_trade,
                results_table,
                debug_dict,
                safe_start,
                safe_decision,
                safe_buy,
                safe_end,
                all_metrics,
                normalized_plot_data,
                inputs,
            )

        except Exception as e:
            return self._error_result(str(e))

    # Helper for initial weights, will move to a utility file
    def _prepare_initial_weights(self, tickers: List[str]) -> pd.Series:
        if not tickers:
            return pd.Series()
        # Simple equal weighting for now
        return pd.Series(1.0 / len(tickers), index=tickers)

    # RL helpers
    def compute_alpha_matrix(
        self, decision_date: pd.Timestamp, lookback_period: int
    ) -> pd.DataFrame:
        """
        HEADLESS SCORER: Computes all metrics in METRIC_REGISTRY for the
        entire universe in a single vectorized pass.
        """
        # 1. Timeline alignment (Using existing logic)
        # We simulate an EngineInput to reuse the validation logic
        mock_input = EngineInput(
            mode="Discovery",
            start_date=decision_date,
            lookback_period=lookback_period,
            holding_period=1,  # Irrelevant for scoring
            metric="All",
            benchmark_ticker=GLOBAL_SETTINGS["benchmark_ticker"],
        )

        try:
            safe_start, safe_decision, _, _ = self._validate_timeline(mock_input)
        except ValueError as e:
            print(f"Timeline Error for {decision_date.date()}: {e}")
            return pd.DataFrame()

        # 2. Extract Full Universe Candidates (Survivors only for this date)
        # We use an empty audit container as this is headless
        candidates = self._filter_universe(
            safe_decision, GLOBAL_SETTINGS["thresholds"], audit_container={}
        )

        if not candidates:
            return pd.DataFrame()

        # 3. Build a "Bulk Observation" (Entire Universe for this date)
        # This uses your existing MarketObservation dataclass but with FULL data
        obs = self._build_observation(safe_decision, candidates, safe_start)

        # 4. Vectorized Execution of the Registry
        # We avoid a for loop over tickers. We only loop over the Strategy names (usually < 20).
        alpha_results = {}

        for name, metric_func in METRIC_REGISTRY.items():
            try:
                # Most of your registry functions are already vectorized (QuantUtils)
                # and will return a pd.Series where index = Tickers.
                scores = metric_func(obs)

                # Ensure the output is a Series for consistency
                if isinstance(scores, (pd.Series, pd.DataFrame)):
                    alpha_results[name] = scores
                else:
                    # Fallback for scalar returns (unlikely given current registry)
                    alpha_results[name] = pd.Series(scores, index=candidates)

            except Exception as e:
                print(f"Warning: Strategy '{name}' failed during headless run: {e}")
                alpha_results[name] = pd.Series(np.nan, index=candidates)

        # 5. Assemble the Alpha Matrix
        # Final shape: [Tickers x Strategies]
        alpha_matrix = pd.DataFrame(alpha_results)

        # Metadata attachment for the RL Agent
        alpha_matrix.index.name = "Ticker"
        return alpha_matrix

    def normalize_alpha_matrix(self, alpha_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        STEP 2: Normalizes the Alpha Matrix so the RL Agent sees
        comparable scales across different metrics.
        """
        if alpha_matrix.empty:
            return alpha_matrix

        # 1. Handle Outliers (The "Clipping" Logic)
        # We don't want a 1,000% gain stock to break the neural network's gradients.
        # We use the clip value from your core.settings.py
        clip_val = GLOBAL_SETTINGS.get("feature_zscore_clip", 4.0)

        # 2. Cross-Sectional Z-Score (Vectorized)
        # Calculation: (Value - Mean) / Std
        # We use axis=0 to normalize each column (strategy) across all tickers.
        normalized = (alpha_matrix - alpha_matrix.mean()) / alpha_matrix.std().replace(
            0, 1
        )

        # 3. Apply the Clip and Fill NaNs
        # NaNs happen if a ticker has no data; we fill with 0.0 (the neutral average)
        normalized = normalized.clip(-clip_val, clip_val).fillna(0.0)

        return normalized

    def compute_context_vector(self, decision_date: pd.Timestamp) -> pd.Series:
        """
        Gathers the 'Market Weather' from the macro_df for the decision date.
        This provides the Regime Awareness shown in the charts.
        """
        if self.macro_df is None or decision_date not in self.macro_df.index:
            # Fallback to neutral values if data is missing
            return pd.Series(
                {
                    "Context_Trend": 0.0,
                    "Context_Vel_Z": 0.0,
                    "Context_Vix_Z": 0.0,
                    "Context_Vix_Ratio": 1.0,
                }
            )

        macro_row = self.macro_df.loc[decision_date]

        # We normalize the raw Macro_Trend (usually -0.3 to 0.3)
        # so it's on a similar scale to Z-scores.
        context = pd.Series(
            {
                "Context_Trend": float(macro_row.get("Macro_Trend", 0.0))
                * 10,  # Scaled for visibility
                "Context_Vel_Z": float(macro_row.get("Macro_Trend_Vel_Z", 0.0)),
                "Context_Vix_Z": float(macro_row.get("Macro_Vix_Z", 0.0)),
                "Context_Vix_Ratio": float(macro_row.get("Macro_Vix_Ratio", 1.0))
                - 1.0,  # Centered at 0.0
            }
        )

        return context

    def precompute_reward_matrix(self, holding_period: int):
        """
        PRECOMPUTE: We store Arithmetic Returns for proper portfolio averaging later.
        """
        # Arithmetic: (P_future / P_now) - 1
        self.reward_matrix = (
            self.df_close.shift(-(holding_period + 1)) / self.df_close.shift(-1)
        ) - 1.0
        self.reward_matrix = self.reward_matrix.fillna(0.0)

    def get_batch_reward(
        self, decision_date: pd.Timestamp, tickers: List[str]
    ) -> float:
        """
        Step 1: Calculate the Arithmetic Mean of the group (The Real World).
        Step 2: Log-transform the result (The Agent's Math).
        """
        if decision_date not in self.reward_matrix.index:
            return 0.0

        # Calculate arithmetic mean of the group
        arithmetic_group_return = self.reward_matrix.loc[decision_date, tickers].mean()

        # Transform to Log Gain: ln(1 + R)
        # This makes the rewards additive for the RL agent's episode total.
        veritable_log_reward = np.log1p(arithmetic_group_return)

        return float(veritable_log_reward)

    def run_discovery_action(
        self,
        decision_date: pd.Timestamp,
        lookback_period: int,
        holding_period: int,
        weights: np.ndarray,
    ) -> DiscoveryResult:

        raw_matrix = self.compute_alpha_matrix(decision_date, lookback_period)
        norm_matrix = self.normalize_alpha_matrix(raw_matrix)

        if norm_matrix.empty:
            return None

        discovery_scores = norm_matrix.values @ weights
        discovery_series = pd.Series(discovery_scores, index=norm_matrix.index)
        top_tickers = (
            discovery_series.sort_values(ascending=False).head(10).index.tolist()
        )
        veritable_reward = self.get_batch_reward(decision_date, top_tickers)

        return DiscoveryResult(
            action_weights=dict(zip(list(METRIC_REGISTRY.keys()), weights)),
            selected_tickers=top_tickers,
            veritable_reward=veritable_reward,
            metric_values=discovery_series.loc[top_tickers],
            raw_alpha_matrix=raw_matrix,  # Included here for your SHV check
        )

    def compute_alpha_ensemble(
        self, decision_date: pd.Timestamp, lookback_periods: List[int]
    ) -> pd.DataFrame:
        """
        ENSEMBLE SCORER: Generates a multi-resolution feature set.
        """
        # 1. THE GATEKEEPER: Filter the universe for this specific date
        # We use the verified thresholds from our GLOBAL_SETTINGS
        candidates = self._filter_universe(
            date_ts=decision_date,
            thresholds=GLOBAL_SETTINGS["thresholds"],
            audit_container={},  # Headless run, no audit needed
        )

        if not candidates:
            return pd.DataFrame()

        ensemble_parts = []

        # 2. RESOLUTION LOOP (21d, 63d, 189d)
        for lb in lookback_periods:
            try:
                # Calculate the P0 Anchor for this resolution
                decision_idx = self.trading_calendar.get_loc(decision_date)
                start_idx = decision_idx - lb
                start_date = self.trading_calendar[start_idx]

                # 3. BUILD THE OBS: Now 'candidates' is defined!
                obs = self._build_observation(
                    decision_date=decision_date,
                    candidates=candidates,
                    start_date=start_date,
                )

                # 4. EXECUTE REGISTRY: Vectorized scoring for the whole universe
                for name, metric_func in METRIC_REGISTRY.items():
                    # .copy() ensures we don't mutate the shared observation object
                    score_series = metric_func(obs).copy()

                    # Tag with resolution for the AI (e.g., '21d_Sharpe_ATRP')
                    score_series.name = f"{lb}d_{name}"
                    ensemble_parts.append(score_series)

            except Exception as e:
                # Senior Dev Tip: Log errors but keep baking the rest of the dates
                print(
                    f"⚠️ Warning: Lookback {lb} failed for {decision_date.date()}: {e}"
                )
                continue

        if not ensemble_parts:
            return pd.DataFrame()

        # Join all metrics into one matrix [Tickers x 33]
        return pd.concat(ensemble_parts, axis=1)


class AlphaCache:
    """
    THE FEATURE CUBE: Now with 'Time-Slicing' to prevent 60-year loops.
    """

    def __init__(self, engine, lookbacks: List[int]):
        self.engine = engine
        self.lookbacks = lookbacks
        self.feature_cube = pd.DataFrame()

    def build(self, start_date: str = "2024-01-01"):
        """Only 'bakes' the features from the start_date onwards."""
        all_dates = self.engine.trading_calendar

        # SLICE: Filter the calendar to only include dates from start_date
        # This reduces 16,000 days to ~500 days.
        target_dates = [d for d in all_dates if d >= pd.Timestamp(start_date)]

        cache_parts = []
        print(
            f"🏗️ Building AlphaCache for {len(target_dates)} days (Starting {start_date})..."
        )

        for i, date in enumerate(target_dates):
            # The engine already handles the lookback logic internally
            # It will skip dates if there isn't enough history behind 'date'
            ensemble = self.engine.compute_alpha_ensemble(date, self.lookbacks)

            if ensemble.empty:
                continue

            # Prepare for MultiIndex: (Date, Ticker)
            ensemble["Date"] = date
            ensemble = ensemble.set_index(["Date", ensemble.index])
            cache_parts.append(ensemble)

            if i % 20 == 0:
                print(f"  Processed {i}/{len(target_dates)} days...")

        if not cache_parts:
            print(
                "❌ Error: No features were generated. Check if start_date is too early for lookbacks."
            )
            return

        self.feature_cube = pd.concat(cache_parts).sort_index()
        print(f"✅ AlphaCache built. Shape: {self.feature_cube.shape}")

    def get_vision(self, date: pd.Timestamp) -> pd.DataFrame:
        """Instant O(1) lookup."""
        try:
            # Note: Because of how we built it, date is level 0
            return self.feature_cube.xs(date, level="Date")
        except KeyError:
            return pd.DataFrame()


#
