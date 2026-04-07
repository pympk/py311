import pandas as pd

from core.settings import GLOBAL_SETTINGS
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable


@dataclass(frozen=True)
class MarketObservation:

    # Dataframes
    lookback_close: pd.DataFrame
    lookback_returns: pd.DataFrame

    # Series (per ticker)
    atrp: pd.Series
    trp: pd.Series

    atr: pd.Series
    rsi: pd.Series
    consistency: pd.Series
    mom_21: pd.Series
    ir_63: pd.Series
    beta_63: pd.Series
    dd_21: pd.Series

    ############
    # --- 4 NEW MICROSTRUCTURE SLOTS ---
    autocorr_15: pd.Series
    range_pos_20: pd.Series
    slope_p_5: pd.Series
    slope_v_5: pd.Series
    convexity: pd.Series  # Adding this now to prevent the next error!
    ############

    # Macro Scalars (or Series)
    macro_trend: float
    macro_trend_vel: float
    macro_vix_z: float
    macro_vix_ratio: float


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
    # start_date: pd.Timestamp
    decision_date: pd.Timestamp
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
class MetricBlueprint:
    """
    Identity Card for a Trading Signal.

    Why this matters for RL:
    - category → Feature groups for diversity constraints (don't pick 5 'Momentum' signals)
    - regime → Market state filter (disable 'Mean Reversion' in strong trends)
    - agent_hint → Curriculum learning seed (initialize policy network with semantic priors)
    """

    name: str
    category: str  # Feature family for portfolio construction
    regime: str  # Market microstructure state
    description: str  # Human-readable math intuition
    agent_hint: str  # RL policy guidance (when to weight this high)
    intervention_trigger: str  # NEW: Specific threshold/action for the agent
    formula: Callable[[Any], pd.Series]

    def __call__(self, obs) -> pd.Series:
        try:
            return self.formula(obs)
        except (AttributeError, ZeroDivisionError, TypeError) as e:
            # Senior Dev: Use rsi or atrp index to ensure we return TICKERS, not Dates
            # This prevents the 'Dates as Tickers' index bug
            target_index = (
                obs.rsi.index if hasattr(obs, "rsi") else obs.lookback_close.columns
            )
            return pd.Series([float("nan")] * len(target_index), index=target_index)
