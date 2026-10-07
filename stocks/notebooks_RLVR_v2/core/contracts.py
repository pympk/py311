import pandas as pd
import numpy as np
import logging

from typing import List, Dict, Any, Optional, Callable

from core.settings import TradingConfig, QualityThresholds
from core.accounting import MTMStepResult
from dataclasses import dataclass, field

# Set up logging if not already configured
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarketObservation:
    # Dataframes
    lookback_close: pd.DataFrame
    lookback_returns: pd.DataFrame

    # Active Strategy Series (per ticker)
    atrp: pd.Series
    trp: pd.Series
    atr: pd.Series
    rsi: pd.Series
    ir_63: pd.Series
    dd_21: pd.Series
    mom_126: pd.Series
    ivol_63: pd.Series
    mom_252_21: pd.Series
    trend_r2_63: pd.Series

    # Macro Scalars
    macro_trend: float
    macro_trend_vel: float
    macro_vix_z: float
    macro_vix_ratio: float

    # Generation 16 Orthogonal Strategy Factor Slots
    res_mom_126: Optional[pd.Series] = None
    range_pos_52w: Optional[pd.Series] = None
    beta_down_63: Optional[pd.Series] = None
    er_63: Optional[pd.Series] = None

    # Legacy slots (Optional defaults to ensure clean decoupling)
    consistency: Optional[pd.Series] = None
    mom_21: Optional[pd.Series] = None
    mom_63: Optional[pd.Series] = None
    beta_63: Optional[pd.Series] = None
    semidev_63: Optional[pd.Series] = None


# -------------------------------------------------------------------------
# DATA PIPELINE CONTRACTS
# -------------------------------------------------------------------------
@dataclass(frozen=True)
class ProcessedDataBundle:
    """Standardized immutable container for aligned datasets.

    Eliminates positional unpacking order bugs.
    """

    df_ohlcv: pd.DataFrame
    macro_df: pd.DataFrame
    features_df: pd.DataFrame


@dataclass
class FilterPack:
    """The 'Saved List' and state for the second filter pass."""

    decision_date: Optional[pd.Timestamp] = None
    eligible_pool: List[str] = field(default_factory=list)  # Survivors of Stage 1
    selected_tickers: List[str] = field(default_factory=list)  # Final output

    def __repr__(self):
        return f"FilterPack(Date: {self.decision_date}, Eligible: {len(self.eligible_pool)}, Selected: {len(self.selected_tickers)})"


@dataclass
class EngineInput:
    mode: str
    decision_date: pd.Timestamp
    lookback_period: int
    holding_period: int
    metric: str
    benchmark_ticker: str
    rank_start: int = 1
    rank_end: int = 10
    # UPDATE 2: Use the QualityThresholds dataclass directly
    quality_thresholds: QualityThresholds = field(default_factory=QualityThresholds)
    manual_tickers: List[str] = field(default_factory=list)
    debug: bool = False
    universe_subset: Optional[List[str]] = None


@dataclass
class EngineOutput:
    # 1. CORE DATA (Required - No Defaults)
    portfolio_series: pd.Series
    benchmark_series: pd.Series
    normalized_plot_data: pd.DataFrame
    tickers: List[str]
    initial_weights: pd.Series
    perf_metrics: Dict[str, float]
    results_df: pd.DataFrame

    # 2. TIMELINE (Required - No Defaults)
    start_date: pd.Timestamp
    decision_date: pd.Timestamp
    buy_date: pd.Timestamp
    holding_end_date: pd.Timestamp

    # 3. OPTIONAL / AUDIT DATA (Must be at the bottom because they have defaults)
    portfolio_atrp_series: Optional[pd.Series] = None
    benchmark_atrp_series: Optional[pd.Series] = None
    portfolio_trp_series: Optional[pd.Series] = None
    benchmark_trp_series: Optional[pd.Series] = None
    error_msg: Optional[str] = None
    debug_data: Optional[Dict[str, Any]] = None
    macro_df: Optional[pd.DataFrame] = None  # <-- ADD THIS

    # 4. The Standardized Alpha Matrix
    alpha_perception: Optional[pd.DataFrame] = None


@dataclass(frozen=True)
class SelectionResult:
    tickers: List[str]
    table: pd.DataFrame
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscoveryResult:
    action_weights: Dict[str, float]
    selected_tickers: List[str]
    veritable_reward: float
    metric_values: pd.Series
    raw_alpha_matrix: pd.DataFrame  # Added for your manual verification


@dataclass(frozen=True)
class BlotterRecord:
    """
    Standardized Mark-To-Market (MTM) Blotter Record.
    Directly aligns with MTMStepResult accounting metrics alongside execution
    timeline, selection metadata, and policy telemetry.
    """

    # --- Timeline ---
    date: pd.Timestamp
    decision_date: pd.Timestamp
    buy_date: pd.Timestamp
    sell_date: pd.Timestamp

    # --- Selection & Controls ---
    tickers: List[str]
    top_3: List[str]
    universe_size: int
    offset: int
    width: int

    # --- Tri-Asset Portfolio Allocation Weights ---
    weight_active: float
    weight_benchmark: float
    weight_cash: float
    equity_exposure: float
    active_tilt: float
    max_score: float
    min_score: float

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

    # --- Compounding Equity Tracking (Base 1.0) ---
    agent_equity: float
    alpha_equity: float


@dataclass(frozen=True)
class MetricBlueprint:
    name: str
    category: str
    regime: str
    description: str
    agent_hint: str
    intervention_trigger: str
    formula: Callable[[Any], pd.Series]
    scaling_type: str = "None"

    def __call__(self, obs) -> pd.Series:
        """Returns RAW data, with strict development tracking and graceful production fallback."""
        try:
            return self.formula(obs)

        except (AttributeError, NameError, KeyError) as code_err:
            # 1. DO NOT suppress coding/structural errors.
            # If the variable or column does not exist, fail immediately to inform the developer.
            raise type(code_err)(
                f"[CRITICAL ERROR] Blueprint '{self.name}' failed due to a code/naming mismatch. "
                f"Please verify if the required fields are defined in MarketObservation and screener.py. "
                f"Original error: {code_err}"
            ) from code_err

        except Exception as data_err:
            # 2. Log and degrade gracefully for math or indexing anomalies (e.g. division by zero, empty series)
            logger.warning(
                f"[WARNING] Blueprint '{self.name}' encountered a data exception on calculation: {data_err}. "
                f"Generating fallback NaN series."
            )

            # Safe degradation index fallback
            target_index = (
                obs.rsi.index if hasattr(obs, "rsi") else obs.lookback_close.columns
            )
            return pd.Series([float("nan")] * len(target_index), index=target_index)


#
