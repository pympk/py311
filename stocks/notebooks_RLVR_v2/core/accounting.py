from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

from core.quant import QuantUtils
from core.settings import TradingConfig


@dataclass(frozen=True)
class MTMStepResult:
    """Standardized Mark-To-Market (MTM) Portfolio Accounting Result."""

    # --- Tri-Asset Portfolio Allocation Weights ---
    weight_active: float
    weight_benchmark: float
    weight_cash: float
    equity_exposure: float
    active_tilt: float

    # --- Simple Returns (Daily Slices & Portfolio Impact) ---
    gross_stock_daily_simple_ret: float
    bm_daily_simple_ret: float
    cash_daily_simple_ret: float
    slippage_daily_simple_loss: float
    gross_daily_simple_ret: float
    net_daily_simple_ret: float
    alpha_daily_simple_ret: float
    penalized_alpha_daily_simple_ret: float

    # --- Log Returns & Policy Optimization Rewards ---
    raw_stock_daily_log_ret: float
    net_daily_log_ret: float
    penalized_alpha_daily_log_reward: float

    # --- Compounding Wealth & Alpha Tracking (Base 1.0) ---
    agent_equity: float
    benchmark_equity: float
    alpha_equity: float  # Strictly V_p(t) / V_bm(t)

    # --- Sleeve Management State ---
    active_sleeves_count: int
    active_tickers: List[str]


class MTMPortfolioEngine:
    """STATEFUL: Mark-to-Market Overlapping FIFO Sleeve Portfolio Engine with

    strict T+1 Next-Day Close Execution and Intra-Sleeve Risk-Parity Weighting.
    """

    def __init__(self, config: Optional[TradingConfig] = None):
        self.config = config or TradingConfig()
        self.holding_period = self.config.holding_period
        self.active_sleeves: deque = deque(maxlen=self.holding_period)
        self.pending_sleeve: Optional[Dict[str, float]] = None

        # Compounding tracking curves (Base 1.0)
        self.portfolio_equity_curve: List[float] = [1.0]
        self.benchmark_equity_curve: List[float] = [1.0]
        self.alpha_equity_curve: List[float] = [1.0]

    def reset(self) -> None:
        """Resets the FIFO queue, pending order, and compounding wealth curves to 1.0."""
        self.active_sleeves = deque(maxlen=self.holding_period)
        self.pending_sleeve = None
        self.portfolio_equity_curve = [1.0]
        self.benchmark_equity_curve = [1.0]
        self.alpha_equity_curve = [1.0]

    @staticmethod
    def calculate_sleeve_simple_ret(
        sleeve: Union[List[str], Dict[str, float]],
        stock_simple_rets: Union[pd.Series, pd.DataFrame, Dict[str, float]],
    ) -> float:
        """
        Calculates realized simple return for an active sleeve using uniform 1/K_t weighting.
        Renormalizes automatically across surviving finite returns.
        """
        if not sleeve:
            return 0.0

        tickers = list(sleeve.keys()) if isinstance(sleeve, dict) else list(sleeve)
        if not tickers:
            return 0.0

        ret_vals = []
        for t in tickers:
            ret = None
            if isinstance(stock_simple_rets, pd.DataFrame):
                if t in stock_simple_rets.columns:
                    val = stock_simple_rets[t].iloc[0]
                    if np.isfinite(val):
                        ret = float(val)
            elif isinstance(stock_simple_rets, pd.Series):
                if t in stock_simple_rets.index:
                    val = stock_simple_rets[t]
                    if np.isfinite(val):
                        ret = float(val)
            elif isinstance(stock_simple_rets, dict):
                if t in stock_simple_rets:
                    val = stock_simple_rets[t]
                    if np.isfinite(val):
                        ret = float(val)

            if ret is not None:
                ret_vals.append(ret)

        if not ret_vals:
            return 0.0

        return QuantUtils.calculate_equal_weight_return(
            np.asarray(ret_vals, dtype=float)
        )

    def step(
        self,
        selected_tickers: Union[List[str], Dict[str, float]],
        equity_exposure: float,
        active_tilt: float,
        stock_simple_rets: Union[pd.Series, pd.DataFrame, Dict[str, float]],
        bm_daily_simple_ret: float,
        cash_daily_simple_ret: float = 0.0,
        selected_weights: Optional[
            Union[Dict[str, float], List[float], np.ndarray]
        ] = None,
    ) -> MTMStepResult:
        """Executes one Mark-to-Market daily transition step."""
        # 1. State Transition: Pending sleeve from prior decision executes NOW (T Close)
        executing_sleeve = self.pending_sleeve
        if executing_sleeve is not None:
            self.active_sleeves.append(executing_sleeve)
            self.pending_sleeve = None

        # 2. Compute realized return across currently active sleeves over (T -> T+1)
        num_active = len(self.active_sleeves)
        if num_active > 0:
            sleeve_simple_returns = [
                self.calculate_sleeve_simple_ret(sleeve, stock_simple_rets)
                for sleeve in self.active_sleeves
            ]
            active_sum = float(np.sum(sleeve_simple_returns))
            unfilled_count = self.holding_period - num_active
            gross_stock_daily_simple_ret = (
                active_sum + (unfilled_count * bm_daily_simple_ret)
            ) / self.holding_period
        else:
            gross_stock_daily_simple_ret = bm_daily_simple_ret

        # 3. Dynamic Tri-Asset Weights
        has_selection = len(selected_tickers) > 0
        if num_active > 0 or has_selection:
            w_active = equity_exposure * active_tilt
            w_benchmark = equity_exposure * (1.0 - active_tilt)
            w_cash = max(0.0, 1.0 - (w_active + w_benchmark))
        else:
            w_active = 0.0
            w_benchmark = equity_exposure
            w_cash = max(0.0, 1.0 - (w_active + w_benchmark))

        # 4. Turnover Slippage Loss: applied when a pending sleeve executes
        slippage_daily_simple_loss = (
            float((self.config.slippage_rate / self.holding_period) * w_active)
            if executing_sleeve is not None
            and len(executing_sleeve) > 0
            and w_active > 0.0
            else 0.0
        )

        # 5. Portfolio Returns via QuantUtils pure math kernel
        gross_daily_simple_ret, net_daily_simple_ret = (
            QuantUtils.calculate_tri_asset_portfolio_return(
                gross_stock_ret=gross_stock_daily_simple_ret,
                bm_ret=bm_daily_simple_ret,
                cash_ret=cash_daily_simple_ret,
                w_active=w_active,
                w_benchmark=w_benchmark,
                w_cash=w_cash,
                slippage_loss=slippage_daily_simple_loss,
            )
        )

        # 6. Active Alpha Simple Return & Policy Optimization Reward Shaping
        alpha_daily_simple_ret = net_daily_simple_ret - bm_daily_simple_ret
        penalized_alpha_daily_simple_ret = QuantUtils.calculate_shaped_alpha_reward(
            alpha_ret=alpha_daily_simple_ret,
            upside_mult=getattr(self.config, "upside_alpha_mult", 1.0),
            loss_penalty=getattr(self.config, "loss_aversion_penalty", 0.0),
        )

        # 7. Log Returns & Policy Optimization Rewards
        raw_stock_daily_log_ret = (
            float(np.log1p(gross_stock_daily_simple_ret))
            if gross_stock_daily_simple_ret > -1.0
            else 0.0
        )
        net_daily_log_ret = (
            float(np.log1p(net_daily_simple_ret))
            if net_daily_simple_ret > -1.0
            else 0.0
        )
        penalized_alpha_daily_log_reward = (
            float(np.log1p(penalized_alpha_daily_simple_ret))
            if penalized_alpha_daily_simple_ret > -1.0
            else 0.0
        )

        # 8. Compounding Equity Curves (Mathematically Rigorous: Base 1.0)
        new_p_equity, new_bm_equity, new_alpha_equity = (
            QuantUtils.step_compounding_curves(
                prev_portfolio_equity=self.portfolio_equity_curve[-1],
                prev_benchmark_equity=self.benchmark_equity_curve[-1],
                net_daily_ret=net_daily_simple_ret,
                bm_daily_ret=bm_daily_simple_ret,
            )
        )

        self.portfolio_equity_curve.append(new_p_equity)
        self.benchmark_equity_curve.append(new_bm_equity)
        self.alpha_equity_curve.append(new_alpha_equity)

        # 9. Format today's selection as pending sleeve dict for next step's execution (Strict 1/K_t)
        ticker_list = (
            list(selected_tickers.keys())
            if isinstance(selected_tickers, dict)
            else list(selected_tickers)
        )
        n_sel = len(ticker_list)
        self.pending_sleeve = {t: 1.0 / n_sel for t in ticker_list} if n_sel > 0 else {}

        active_tickers_list = ticker_list

        return MTMStepResult(
            weight_active=w_active,
            weight_benchmark=w_benchmark,
            weight_cash=w_cash,
            equity_exposure=equity_exposure,
            active_tilt=active_tilt,
            gross_stock_daily_simple_ret=gross_stock_daily_simple_ret,
            bm_daily_simple_ret=bm_daily_simple_ret,
            cash_daily_simple_ret=cash_daily_simple_ret,
            slippage_daily_simple_loss=slippage_daily_simple_loss,
            gross_daily_simple_ret=gross_daily_simple_ret,
            net_daily_simple_ret=net_daily_simple_ret,
            alpha_daily_simple_ret=alpha_daily_simple_ret,
            penalized_alpha_daily_simple_ret=penalized_alpha_daily_simple_ret,
            raw_stock_daily_log_ret=raw_stock_daily_log_ret,
            net_daily_log_ret=net_daily_log_ret,
            penalized_alpha_daily_log_reward=penalized_alpha_daily_log_reward,
            agent_equity=new_p_equity,
            benchmark_equity=new_bm_equity,
            alpha_equity=new_alpha_equity,
            active_sleeves_count=len(self.active_sleeves),
            active_tickers=active_tickers_list,
        )
