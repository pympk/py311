import pandas as pd
import numpy as np
import gc
import logging

from typing import List, cast

from core.contracts import MarketObservation, SelectionResult, EngineInput, EngineOutput
from core.settings import TradingConfig
from strategy.registry import get_strategy_registry
from data_pipeline.screener import UniverseScreener
from walk_forward.performance import (
    calculate_buy_and_hold_performance,
    PerformanceCalculator,
)


class AlphaEngine:
    # Add these type hints here
    trading_calendar: pd.DatetimeIndex
    df_close: pd.DataFrame
    df_atrp: pd.DataFrame
    df_trp: pd.DataFrame

    def __init__(
        self,
        df_ohlcv: pd.DataFrame,
        features_df: pd.DataFrame,
        macro_df: pd.DataFrame,
        config: TradingConfig | None = None,  # Use | None to allow the None default
        df_close_wide: pd.DataFrame | None = None,
        df_atrp_wide: pd.DataFrame | None = None,
        df_trp_wide: pd.DataFrame | None = None,
    ):
        self.df_ohlcv_raw = df_ohlcv
        self.features_df = features_df
        self.macro_df = macro_df
        self.config = config or TradingConfig()

        self._prepare_data(
            df_close_wide, df_atrp_wide, df_trp_wide, self.config.calendar_ticker
        )

        # Inject UniverseScreener from the data_pipeline slice
        self.screener = UniverseScreener(
            df_close=self.df_close,
            features_df=self.features_df,
            macro_df=self.macro_df,
            trading_calendar=self.trading_calendar,
            config=self.config,
        )

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

        if self.config.handle_zeros_as_nan:
            self.df_close = self.df_close.replace(0, np.nan)

        self.df_close = self.df_close.ffill(limit=self.config.max_data_gap_ffill)
        self.df_close = self.df_close.fillna(self.config.nan_price_replacement)
        self.df_close.index = pd.to_datetime(self.df_close.index)

        if master_ticker not in self.df_close.columns:
            master_ticker = self.df_close.columns[0]

        # Use cast here to satisfy the UniverseScreener requirement
        self.trading_calendar = cast(
            pd.DatetimeIndex,
            self.df_close[master_ticker].dropna().index.unique().sort_values(),
        )

    def _execute_strategy(self, observation: MarketObservation, metric_name: str):
        registry = get_strategy_registry(self.config)
        if metric_name not in registry:
            valid_keys = list(registry.keys())
            raise ValueError(
                f"[ERROR] Strategy '{metric_name}' not found. Available strategies: {valid_keys}"
            )
        try:
            return registry[metric_name](observation)
        except Exception as e:
            raise ValueError(f"[ERROR] Math Error in '{metric_name}': {str(e)}")

    def _rank_and_slice(self, raw_scores, inputs, observation):
        clean_scores = raw_scores.dropna()
        dropped_tickers = raw_scores[raw_scores.isna()].index.tolist()

        if clean_scores.empty:
            raise ValueError(
                f"[ERROR] All strategy scores are NaN. Dropped: {dropped_tickers}"
            )

        sorted_tickers = clean_scores.sort_values(ascending=False)
        start_idx = max(0, inputs.rank_start - 1)
        end_idx = inputs.rank_end
        selected_tickers = sorted_tickers.iloc[start_idx:end_idx].index.tolist()

        if not selected_tickers:
            raise ValueError(
                f"[ERROR] Ranking returned zero tickers for range {inputs.rank_start}-{inputs.rank_end}."
            )

        try:
            debug_artifact = pd.DataFrame(
                {
                    "Strategy_Score": raw_scores,
                    "Raw_Price_Start": observation.lookback_close.iloc[0],
                    "Raw_Price_End": observation.lookback_close.iloc[-1],
                    "Raw_TRP_Mean": observation.trp,
                    "Raw_ATRP_Mean": observation.atrp,
                    "Raw_Mom_21": observation.mom_21,
                    "Raw_IR_63": observation.ir_63,
                    "Raw_Consistency": observation.consistency,
                    "Raw_RSI": observation.rsi,
                    "Raw_DD_21": observation.dd_21,
                    "Raw_Range_Pos_20": observation.range_pos_20,
                    "Raw_Slope_P_5": observation.slope_p_5,
                    "Raw_Slope_V_5": observation.slope_v_5,
                    "Raw_Convexity": observation.convexity,
                    "Raw_AutoCorr_15": observation.autocorr_15,
                }
            )

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

        except Exception as e:
            raise ValueError(f"[ERROR] Error assembling ranking tables: {str(e)}")

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
        audit_info = {}

        if inputs.mode == "Manual List":
            valid = [t for t in inputs.manual_tickers if t in self.df_close.columns]
            if inputs.debug:
                debug_dict["audit_liquidity"] = {
                    "mode": "Manual",
                    "tickers_passed": len(valid),
                }
            return SelectionResult(
                tickers=valid, table=pd.DataFrame(index=valid), debug=debug_dict
            )

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
            candidates = self.screener.filter_universe(
                decision_date, inputs.quality_thresholds, audit_info
            )
            logging.debug(
                f"{len(candidates)} stocks passed filters on {decision_date.date()}"
            )
            if inputs.debug:
                debug_dict["audit_liquidity"] = audit_info

        if not candidates:
            raise ValueError("[ERROR] No survivors.")

        obs = self.screener.build_observation(decision_date, candidates, start_date)
        scores = self._execute_strategy(obs, inputs.metric)
        rank_results = self._rank_and_slice(scores, inputs, obs)

        # DEBUG TRAP
        alignment_error = not (obs.lookback_returns.columns == obs.trp.index).all()
        if alignment_error:
            print(
                f"[CRITICAL] Alignment mismatch! Returns: {obs.lookback_returns.columns[0]}, TRP: {obs.trp.index[0]}"
            )

        return SelectionResult(
            tickers=rank_results["tickers"],
            table=rank_results["table"],
            debug={**debug_dict, **rank_results.get("debug", {})},
        )

    def _get_debug_components(self, tickers, start, end):
        idx = pd.IndexSlice
        return {
            "prices": self.df_close[tickers].loc[start:end],
            "atrp": self.df_atrp[tickers].loc[start:end],
            "trp": self.df_trp[tickers].loc[start:end],
            "ohlcv_raw": self.df_ohlcv_raw.loc[idx[tickers, start:end], :],
        }

    def _get_normalized_plot_data(self, tickers, start, end):
        if not tickers:
            return pd.DataFrame()
        data = self.df_close[list(set(tickers))].loc[start:end]
        return data / data.bfill().iloc[0]

    def _error_result(self, msg) -> EngineOutput:
        return EngineOutput(
            portfolio_series=pd.Series(dtype=float),
            benchmark_series=pd.Series(dtype=float),
            normalized_plot_data=pd.DataFrame(),
            tickers=[],
            initial_weights=pd.Series(dtype=float),
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
            # --- START DEBUG TRAP ---
            timeline_result = self.screener.validate_timeline(inputs)
            if timeline_result is None:
                print(
                    f"[DEBUG TRAP] validate_timeline returned None! Calendar len: {len(self.screener.trading_calendar)}"
                )
                return self._error_result("validate_timeline returned None")

            safe_start, safe_decision, safe_buy, safe_end = timeline_result
            # --- END DEBUG TRAP ---

            selection = self._select_tickers(inputs, safe_start, safe_decision)
            # --- END DEBUG TRAP ---

            safe_start, safe_decision, safe_buy, safe_end = (
                self.screener.validate_timeline(inputs)
            )
            selection = self._select_tickers(inputs, safe_start, safe_decision)

            tickers_to_trade = selection.tickers
            results_table = selection.table
            debug_dict = selection.debug

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
                        f"[ERROR] No price data for {prefix} in selected period."
                    )

                m, slices = PerformanceCalculator.calculate_period_metrics(
                    full, hold, safe_decision, prefix
                )
                all_metrics.update(m)
                verification_slices[prefix] = slices
                perf_store[prefix] = full

            normalized_plot_data = pd.DataFrame()

            if inputs.debug:
                # To maintain debug UI functionality for Alpha perception, we use the registry directly.
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
            if inputs.debug:
                import traceback

                print(f"\n[ENGINE CRASH TRACEBACK] [CRASH]")
                traceback.print_exc()
            return self._error_result(str(e))

    def shutdown(self):
        print(f"Shutting down Engine {id(self)}...")
        self.features_df = pd.DataFrame()
        self.df_ohlcv_raw = pd.DataFrame()
        self.macro_df = pd.DataFrame()
        gc.collect()

    def _prepare_initial_weights(self, tickers: List[str]) -> pd.Series:
        if not tickers:
            return pd.Series()
        return pd.Series(1.0 / len(tickers), index=tickers)
