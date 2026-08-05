import pandas as pd
import numpy as np
import random  # <--- [NEW] Add random
from typing import cast
from core.logic import AlphaLogic, SelectionLogic
from core.settings import TradingConfig


class DiscoveryEnv:
    def __init__(
        self,
        feature_cube: pd.DataFrame,
        reward_matrix: pd.DataFrame,
        calendar: pd.DatetimeIndex,
        macro_df: pd.DataFrame,
        config: TradingConfig | None = None,
        randomize_start: bool = False,  # <--- [NEW] Switch to enable random starts
        episode_steps: int = 0,  # <--- [NEW] Accepts NUM_STEPS dynamically
    ):
        self.cube = feature_cube
        self.reward_matrix = reward_matrix
        self.calendar = calendar
        self.macro_df = macro_df
        self.config = config or TradingConfig()
        self.holding_period = self.config.holding_period

        self.randomize_start = randomize_start  # <--- [NEW]
        self.episode_steps = episode_steps  # <--- [NEW]

        # ---> PHASE 1: CPU ACCELERATION (PRE-CACHING) <---
        # We pre-slice Pandas structures into aligned O(1) Python lists.
        self._ensembles = []
        self._macro_rows = []
        self._mkt_rets = []

        # Default empty row in case macro_df drops a date
        macro_cols = self.macro_df.columns if not self.macro_df.empty else []
        default_macro_row = pd.Series(0.0, index=macro_cols)

        # for date in self.calendar:
        #     # 1. Pre-cache Strategy Ensembles (avoids .xs in loops)
        #     try:
        #         ensemble = self.cube.xs(date, level="Date")
        #     except KeyError:
        #         ensemble = pd.DataFrame()
        #     self._ensembles.append(ensemble)

        #     # 2. Pre-cache Macro Rows and Market Returns (avoids .loc in loops)
        #     if date in self.macro_df.index:
        #         self._macro_rows.append(self.macro_df.loc[date])
        #         self._mkt_rets.append(self.macro_df.loc[date, "Mkt_Ret"])
        #     else:
        #         self._macro_rows.append(default_macro_row)
        #         self._mkt_rets.append(0.0)

        for date in self.calendar:
            # 1. Pre-cache Strategy Ensembles (avoids .xs in loops)
            try:
                ensemble = self.cube.xs(date, level="Date")
            except KeyError:
                ensemble = pd.DataFrame()
            self._ensembles.append(ensemble)

            # 2. Pre-cache Macro Rows
            if date in self.macro_df.index:
                self._macro_rows.append(self.macro_df.loc[date])
            else:
                self._macro_rows.append(default_macro_row)

            # 3. Pre-cache Forward Market Return (avoids .loc in loops)
            # Extracted from the pre-shifted reward_matrix to perfectly match T+1 to T+1+HP
            benchmark = self.config.benchmark_ticker
            if (
                date in self.reward_matrix.index
                and benchmark in self.reward_matrix.columns
            ):
                try:
                    fwd_mkt_raw = self.reward_matrix.loc[date, benchmark]

                    # Convert to flat numpy array to safely extract scalar and bypass Pylance strictness
                    fwd_mkt_array = np.asarray(fwd_mkt_raw).ravel()

                    if fwd_mkt_array.size > 0:
                        fwd_mkt = float(fwd_mkt_array[0])
                        self._mkt_rets.append(fwd_mkt if not np.isnan(fwd_mkt) else 0.0)
                    else:
                        self._mkt_rets.append(0.0)
                except Exception:
                    # Trap any unexpected extraction errors
                    self._mkt_rets.append(0.0)
            else:
                self._mkt_rets.append(0.0)

        self.reset()

    def reset(self, start_date=None):
        if start_date:
            idx = self.calendar.get_loc(start_date)
            self.current_date_idx = cast(int, idx)
        elif self.randomize_start and self.episode_steps > 0:
            # <--- [NEW FIX] Training Mode: Pick a random starting date! --->
            # We dynamically subtract episode_steps (NUM_STEPS) so we don't
            # run off the edge of the calendar.
            max_start_idx = (
                len(self.calendar) - self.episode_steps - self.holding_period - 1
            )

            if max_start_idx > 0:
                self.current_date_idx = random.randint(0, max_start_idx)
            else:
                self.current_date_idx = 0
        else:
            # <--- [NEW FIX] Validation/Test Mode: Always start at Day 0 --->
            self.current_date_idx = 0

        # Track both Absolute Return & Alpha Outperformance
        self.equity_curve = [1.0]
        self.alpha_equity_curve = [1.0]
        return self._get_observation()

    def _get_observation(self):
        # FAST O(1) lookups utilizing our pre-cached lists
        date = self.calendar[self.current_date_idx]
        ensemble = self._ensembles[self.current_date_idx]
        macro_row = self._macro_rows[self.current_date_idx]

        return {
            "ensemble": ensemble,
            "date": date,
            "macro_row": macro_row,  # Pushed into dict for adapter
        }

    def step(self, action: np.ndarray):
        date = self.calendar[self.current_date_idx]
        obs_dict = self._get_observation()
        ensemble = obs_dict["ensemble"]

        # 1. Delegate Ticker Selection
        selected_tickers, top_3, offset, width, max_s, min_s = (
            SelectionLogic.apply_action(
                ensemble,
                action,
                self.config.rank_max_offset_percentile,
                self.config.rank_max_width,
            )
        )

        # 2. Extract Raw Truth Reward
        log_reward = AlphaLogic.calculate_veritable_reward(
            self.reward_matrix, date, selected_tickers
        )

        # 3. Apply Slippage, Constraints, & Alpha Math
        raw_sleeve_return = np.exp(log_reward) - 1.0
        slippage_applied = 0.0

        if len(selected_tickers) > 0:
            slippage_applied = self.config.slippage_rate
            raw_sleeve_return -= slippage_applied

        # FAST O(1) list lookup
        mkt_return = self._mkt_rets[self.current_date_idx]
        alpha = raw_sleeve_return - mkt_return

        # Penalize underperformance aggressively for the RL Agent
        penalized_alpha = alpha * self.config.downside_penalty if alpha < 0 else alpha

        # 4. Update Internal State curves
        # Math scales it down by holding period logic per capital deployment
        portfolio_impact = raw_sleeve_return / self.holding_period
        alpha_impact = alpha / self.holding_period

        self.equity_curve.append(self.equity_curve[-1] * (1.0 + portfolio_impact))
        self.alpha_equity_curve.append(
            self.alpha_equity_curve[-1] * (1.0 + alpha_impact)
        )

        # Store the decision index BEFORE we increment it
        decision_idx = self.current_date_idx

        self.current_date_idx += 1

        # We need enough room for T+1 (Buy) and T+1+HP (Sell)
        done = self.current_date_idx >= (len(self.calendar) - self.holding_period - 1)

        # 5. Temporal Alignment & BLOTTER Update
        # Match the Oracle: Buy is T+1, Sell is T+1+HP
        buy_date = self.calendar[decision_idx + 1]
        sell_date = self.calendar[decision_idx + 1 + self.holding_period]

        info = {
            "date": date,
            "buy_date": buy_date,
            "sell_date": sell_date,
            "tickers": selected_tickers,
            "top_3": top_3,
            "universe_size": len(ensemble),
            "offset": offset,
            "width": width,
            "max_score": max_s,
            "min_score": min_s,
            # BLOTTER METRICS
            "raw_log_reward": log_reward,
            "actual_return": raw_sleeve_return,
            "mkt_return": mkt_return,
            "alpha": alpha,
            "penalized_alpha": penalized_alpha,
            "slippage_applied": slippage_applied,
            # Passing pre-calculated impacts and curves
            "portfolio_impact": portfolio_impact,
            "alpha_impact": alpha_impact,
            "agent_equity": self.equity_curve[-1],
            "alpha_equity": self.alpha_equity_curve[-1],
        }

        # The RL Engine receives penalized_alpha to optimize
        return self._get_observation(), penalized_alpha, done, info
