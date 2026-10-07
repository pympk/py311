from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

from core.quant import QuantUtils
from core.settings import TradingConfig


@dataclass(frozen=True)
class MTMStepResult:
    """Standardized Mark-To-Market (MTM) Portfolio Accounting Result.

    All rates and returns are explicitly tagged as simple return or log return.
    """

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

    strict T+1 Next-Day Close Execution.

    Execution Lifecycle:
    1. Day T (Close): Decision made. Stored in `pending_sleeve`. Zero exposure over (T -> T+1).
    2. Day T+1 (Close): `pending_sleeve` executes and enters `active_sleeves`.
       The oldest sleeve (held for H days) is evicted.
    3. Over interval (T+1 -> T+2): Active sleeves earn realized returns.
    """

    def __init__(self, config: Optional[TradingConfig] = None):
        self.config = config or TradingConfig()
        self.holding_period = self.config.holding_period
        self.active_sleeves: deque = deque(maxlen=self.holding_period)
        self.pending_sleeve: Optional[List[str]] = None

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
    def extract_sleeve_return_array(
        sleeve: List[str],
        stock_simple_rets: Union[pd.Series, pd.DataFrame, Dict[str, float]],
    ) -> np.ndarray:
        """DataFrame Slicing Boundary: Extracts clean 1D float array of returns for a sleeve."""
        if not sleeve:
            return np.empty(0, dtype=float)

        if isinstance(stock_simple_rets, pd.DataFrame):
            row = stock_simple_rets.iloc[0]
            valid_s = row.reindex(sleeve).dropna()
            return valid_s.to_numpy(dtype=float)
        elif isinstance(stock_simple_rets, pd.Series):
            valid_s = stock_simple_rets.reindex(sleeve).dropna()
            return valid_s.to_numpy(dtype=float)
        elif isinstance(stock_simple_rets, dict):
            vals = [
                stock_simple_rets[t]
                for t in sleeve
                if t in stock_simple_rets and np.isfinite(stock_simple_rets[t])
            ]
            return np.asarray(vals, dtype=float)
        return np.empty(0, dtype=float)

    @staticmethod
    def calculate_sleeve_simple_ret(
        sleeve: List[str],
        stock_simple_rets: Union[pd.Series, pd.DataFrame, Dict[str, float]],
    ) -> float:
        """Slices the DataFrame and delegates mean return reduction to QuantUtils."""
        ret_arr = MTMPortfolioEngine.extract_sleeve_return_array(
            sleeve, stock_simple_rets
        )
        return QuantUtils.calculate_equal_weight_return(ret_arr)

    def step(
        self,
        selected_tickers: List[str],
        equity_exposure: float,
        active_tilt: float,
        stock_simple_rets: Union[pd.Series, pd.DataFrame, Dict[str, float]],
        bm_daily_simple_ret: float,
        cash_daily_simple_ret: float = 0.0,
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
        if num_active > 0 or len(selected_tickers) > 0:
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

        # 9. Store today's selection as pending order for next step's close execution
        self.pending_sleeve = list(selected_tickers)

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
            alpha_equity=new_alpha_equity,  # Strictly V_p / V_bm (never 350 again!)
            active_sleeves_count=len(self.active_sleeves),
            active_tickers=selected_tickers,
        )
