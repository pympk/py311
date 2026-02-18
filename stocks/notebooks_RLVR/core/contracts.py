import pandas as pd

from core.settings import GLOBAL_SETTINGS
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, TypedDict


class MarketObservation(TypedDict):
    """
    The 'STATE' (Observation) in Reinforcement Learning.
    """

    # --- The Movie (Time Series) ---
    lookback_returns: pd.DataFrame
    lookback_close: pd.DataFrame

    # --- The Snapshot (Scalar values at Decision Time) ---
    atrp: pd.Series
    trp: pd.Series  # <--- RESTORED

    # NEW SENSORS
    atr: pd.Series  # <--- RESTORED (Optional, but good for debug)
    rsi: pd.Series
    consistency: pd.Series
    mom_21: pd.Series
    ir_63: pd.Series
    beta_63: pd.Series
    dd_21: pd.Series

    # MACRO CONTEXT
    macro_trend: pd.Series
    macro_trend_vel: pd.Series  # <-- ADD (Raw 21d slope)
    macro_trend_vel_z: pd.Series  # <-- ADD (Z-score normalized)
    macro_vix_z: pd.Series
    macro_vix_ratio: pd.Series


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
    start_date: pd.Timestamp
    lookback_period: int
    holding_period: int
    metric: str
    benchmark_ticker: str
    rank_start: int = 1
    rank_end: int = 10
    # Default factory pulls from Global thresholds
    quality_thresholds: Dict[str, float] = field(
        default_factory=lambda: GLOBAL_SETTINGS["thresholds"].copy()
    )
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
