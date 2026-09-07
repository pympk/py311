from collections import deque
import random
from typing import Any, Dict, List, Optional, cast
import numpy as np
import pandas as pd

from core.accounting import MTMPortfolioEngine, MTMStepResult
from core.logic import SelectionLogic
from core.settings import TradingConfig


class DiscoveryEnv:
    """
    Reinforcement Learning Discovery Environment with 1-Day Mark-to-Market (MTM)
    transitions and FIFO sleeve accounting.
    """

    def __init__(
        self,
        feature_cube: pd.DataFrame,
        simple_ret_matrix: pd.DataFrame,
        calendar: Optional[pd.DatetimeIndex] = None,
        macro_df: Optional[pd.DataFrame] = None,
        config: Optional[TradingConfig] = None,
        randomize_start: bool = False,
        episode_steps: Optional[int] = None,
    ):
        self.cube = feature_cube
        self.simple_ret_matrix = simple_ret_matrix
        self.calendar = calendar if calendar is not None else pd.DatetimeIndex([])
        self.macro_df = macro_df if macro_df is not None else pd.DataFrame()
        self.config = config or TradingConfig()
        self.holding_period = self.config.holding_period

        self.randomize_start = randomize_start or getattr(
            self.config, "randomize_start", False
        )
        # episode_steps <= 0 denotes complete calendar execution without step truncation
        if episode_steps is not None:
            self.episode_steps = max(0, int(episode_steps))
        else:
            self.episode_steps = max(0, int(getattr(self.config, "episode_steps", 0)))

        # Initialize dedicated MTM Portfolio Engine
        self.portfolio_engine = MTMPortfolioEngine(config=self.config)

        # Pre-cache daily slices
        self._ensembles = []
        self._macro_rows = []
        self._bm_rows = []
        self._stock_simple_rets: List[pd.Series] = []
        self._mkt_rets: List[float] = []
        self._cash_rets: List[float] = []

        macro_cols = self.macro_df.columns if not self.macro_df.empty else []
        default_macro_row = pd.Series(0.0, index=macro_cols)

        feature_cols = self.cube.columns if not self.cube.empty else []
        default_bm_row = pd.Series(0.0, index=feature_cols)

        benchmark = self.config.benchmark_ticker
        try:
            bm_cube_slice = self.cube.xs(benchmark, level="Ticker")
        except KeyError:
            bm_cube_slice = pd.DataFrame()

        for date in self.calendar:
            # 1. Strategy Ensembles
            try:
                ensemble = self.cube.xs(date, level="Date")
            except KeyError:
                ensemble = pd.DataFrame()
            self._ensembles.append(ensemble)

            # 2. Benchmark Feature Row
            if not bm_cube_slice.empty and date in bm_cube_slice.index:
                self._bm_rows.append(bm_cube_slice.loc[date])
            else:
                self._bm_rows.append(default_bm_row)

            # 3. Macro Rows
            if date in self.macro_df.index:
                self._macro_rows.append(self.macro_df.loc[date])
            else:
                self._macro_rows.append(default_macro_row)

            # 4. Daily Universe Stock Simple Returns
            if date in self.simple_ret_matrix.index:
                row_val = self.simple_ret_matrix.loc[date]
                if isinstance(row_val, pd.DataFrame):
                    self._stock_simple_rets.append(cast(pd.Series, row_val.iloc[0]))
                else:
                    self._stock_simple_rets.append(cast(pd.Series, row_val))
            else:
                self._stock_simple_rets.append(pd.Series(dtype=float))

            # 5. 1-Day Benchmark Simple Return
            if (
                date in self.simple_ret_matrix.index
                and benchmark in self.simple_ret_matrix.columns
            ):
                try:
                    fwd_mkt_raw = self.simple_ret_matrix.loc[date, benchmark]
                    fwd_mkt_arr = np.asarray(fwd_mkt_raw).ravel()
                    val = float(fwd_mkt_arr[0]) if fwd_mkt_arr.size > 0 else 0.0
                    self._mkt_rets.append(val if np.isfinite(val) else 0.0)
                except Exception:
                    self._mkt_rets.append(0.0)
            else:
                self._mkt_rets.append(0.0)

            # 6. 1-Day Cash Simple Return
            if (
                date in self.simple_ret_matrix.index
                and "CASH" in self.simple_ret_matrix.columns
            ):
                try:
                    fwd_cash_raw = self.simple_ret_matrix.loc[date, "CASH"]
                    fwd_cash_arr = np.asarray(fwd_cash_raw).ravel()
                    val = float(fwd_cash_arr[0]) if fwd_cash_arr.size > 0 else 0.0
                    self._cash_rets.append(val if np.isfinite(val) else 0.0)
                except Exception:
                    self._cash_rets.append(0.0)
            else:
                self._cash_rets.append(0.0)

        self._audit_temporal_integrity()
        self.reset()

    def _audit_temporal_integrity(self) -> None:
        """Temporal Sentinel: Verifies simple_ret_matrix is forward-looking."""
        if self.simple_ret_matrix.empty or len(self.calendar) < 10 or self.cube.empty:
            return

        cube_dates = set(self.cube.index.get_level_values("Date").unique())
        ret_dates = set(self.simple_ret_matrix.index)
        sample_dates = [
            d
            for d in self.calendar[5 : min(25, len(self.calendar))]
            if d in cube_dates and d in ret_dates
        ]
        correlations = []

        for d in sample_dates:
            try:
                features = self.cube.xs(d, level="Date")
                gain_cols = [
                    c for c in features.columns if "Price Gain" in c or "Mom" in c
                ]
                if not gain_cols:
                    continue

                feat_vals = pd.to_numeric(
                    features[gain_cols[0]], errors="coerce"
                ).dropna()
                row_raw = self.simple_ret_matrix.loc[d]
                row_s = (
                    row_raw.iloc[0] if isinstance(row_raw, pd.DataFrame) else row_raw
                )
                ret_vals = pd.to_numeric(
                    row_s.reindex(feat_vals.index), errors="coerce"
                ).dropna()

                common = feat_vals.index.intersection(ret_vals.index)
                if len(common) >= 5:
                    v1 = feat_vals.loc[common].to_numpy(dtype=float)
                    v2 = ret_vals.loc[common].to_numpy(dtype=float)
                    if np.std(v1) > 1e-8 and np.std(v2) > 1e-8:
                        c = np.corrcoef(v1, v2)[0, 1]
                        if np.isfinite(c):
                            correlations.append(c)
            except Exception:
                continue

        if correlations and np.mean(correlations) > 0.65:
            raise ValueError(
                f"🚨 CRITICAL DATA LEAKAGE DETECTED! "
                f"Reward matrix has average cross-sectional correlation of {np.mean(correlations):.2f} "
                f"with contemporaneous features. Pass forward returns (shift(-1)), never Ret_1d!"
            )

    @property
    def active_sleeves(self) -> deque:
        return self.portfolio_engine.active_sleeves

    @property
    def equity_curve(self) -> List[float]:
        return self.portfolio_engine.portfolio_equity_curve

    @property
    def alpha_equity_curve(self) -> List[float]:
        return self.portfolio_engine.alpha_equity_curve

    def reset(self, start_date=None, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
        if not hasattr(self, "rng"):
            self.rng = random.Random()

        if start_date:
            idx = self.calendar.get_loc(start_date)
            self.current_date_idx = cast(int, idx)
        elif self.randomize_start and self.episode_steps > 0:
            max_start_idx = (
                len(self.calendar) - self.episode_steps - self.holding_period - 1
            )
            self.current_date_idx = (
                self.rng.randint(0, max_start_idx) if max_start_idx > 0 else 0
            )
        else:
            self.current_date_idx = 0

        self.steps_taken = 0
        self.portfolio_engine.reset()
        return self._get_observation()

    def _get_observation(self) -> Dict[str, Any]:
        if len(self.calendar) == 0 or len(self._ensembles) == 0:
            return {
                "ensemble": pd.DataFrame(),
                "date": pd.Timestamp.min,
                "macro_row": pd.Series(dtype=float),
                "bm_row": pd.Series(dtype=float),
            }
        idx = min(self.current_date_idx, len(self.calendar) - 1)
        return {
            "ensemble": self._ensembles[idx],
            "date": self.calendar[idx],
            "macro_row": self._macro_rows[idx],
            "bm_row": self._bm_rows[idx],
        }

    def step(self, action: np.ndarray):
        date = self.calendar[self.current_date_idx]
        obs_dict = self._get_observation()
        ensemble = obs_dict["ensemble"]

        # 1. Decode Action into Tickers & Dynamic Allocation Controls
        (
            selected_tickers,
            top_3,
            offset,
            width,
            equity_exposure,
            active_tilt,
            max_s,
            min_s,
        ) = SelectionLogic.apply_action(
            ensemble,
            action,
            rank_max_offset_percentile=self.config.rank_max_offset_percentile,
            rank_max_width=self.config.rank_max_width,
            min_basket_width=self.config.min_basket_width,
            max_cash_pct=self.config.max_cash_pct,
            min_active_tilt=self.config.min_active_tilt,
        )

        # 2. Extract Cross-Sectional Stock Simple Returns for Today
        if self.current_date_idx < len(self._stock_simple_rets):
            stock_simple_rets = self._stock_simple_rets[self.current_date_idx]
        elif date in self.simple_ret_matrix.index:
            row_val = self.simple_ret_matrix.loc[date]
            stock_simple_rets = cast(
                pd.Series,
                row_val.iloc[0] if isinstance(row_val, pd.DataFrame) else row_val,
            )
        else:
            stock_simple_rets = pd.Series(dtype=float)

        daily_bm_simple_ret = self._mkt_rets[self.current_date_idx]
        daily_cash_simple_ret = self._cash_rets[self.current_date_idx]

        # 3. Delegate Quantitative Accounting to MTMPortfolioEngine
        step_result: MTMStepResult = self.portfolio_engine.step(
            selected_tickers=selected_tickers,
            equity_exposure=equity_exposure,
            active_tilt=active_tilt,
            stock_simple_rets=stock_simple_rets,
            bm_daily_simple_ret=daily_bm_simple_ret,
            cash_daily_simple_ret=daily_cash_simple_ret,
        )

        decision_idx = self.current_date_idx
        self.current_date_idx += 1
        self.steps_taken += 1

        calendar_done = self.current_date_idx >= (len(self.calendar) - 2)
        steps_done = self.episode_steps > 0 and self.steps_taken >= self.episode_steps
        terminated = False  # Trading portfolio has no absorbing terminal failure state
        truncated = calendar_done or steps_done

        # 4. Standardized Telemetry Payload
        info = {
            # Timeline
            "date": date,
            "decision_date": date,
            "buy_date": self.calendar[min(decision_idx + 1, len(self.calendar) - 1)],
            "sell_date": self.calendar[
                min(
                    decision_idx + 1 + self.holding_period,
                    len(self.calendar) - 1,
                )
            ],
            # Stock Selections & Strategy Controls
            "tickers": selected_tickers,
            "top_3": top_3,
            "universe_size": len(ensemble),
            "offset": offset,
            "width": width,
            # Dynamic Asset Allocation Weights
            "weight_active": step_result.weight_active,
            "weight_benchmark": step_result.weight_benchmark,
            "weight_cash": step_result.weight_cash,
            "equity_exposure": step_result.equity_exposure,
            "active_tilt": step_result.active_tilt,
            "max_score": max_s,
            "min_score": min_s,
            # Daily Simple Returns
            "gross_stock_daily_simple_ret": step_result.gross_stock_daily_simple_ret,
            "bm_daily_simple_ret": step_result.bm_daily_simple_ret,
            "cash_daily_simple_ret": step_result.cash_daily_simple_ret,
            "slippage_daily_simple_loss": step_result.slippage_daily_simple_loss,
            "gross_daily_simple_ret": step_result.gross_daily_simple_ret,
            "net_daily_simple_ret": step_result.net_daily_simple_ret,
            "alpha_daily_simple_ret": step_result.alpha_daily_simple_ret,
            "penalized_alpha_daily_simple_ret": step_result.penalized_alpha_daily_simple_ret,
            # Policy Log Rewards
            "raw_stock_daily_log_ret": step_result.raw_stock_daily_log_ret,
            "net_daily_log_ret": step_result.net_daily_log_ret,
            "penalized_alpha_daily_log_reward": step_result.penalized_alpha_daily_log_reward,
            # Compounding Equity Curves
            "agent_equity": step_result.agent_equity,
            "alpha_equity": step_result.alpha_equity,
        }

        return (
            self._get_observation(),
            step_result.penalized_alpha_daily_simple_ret,
            terminated,
            truncated,
            info,
        )
