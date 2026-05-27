import pandas as pd
import numpy as np

from typing import List, Dict, Any, Optional, Callable

from core.settings import TradingConfig, QualityThresholds
from dataclasses import dataclass, field


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

    # --- 4 NEW MICROSTRUCTURE SLOTS ---
    autocorr_15: pd.Series
    range_pos_20: pd.Series
    slope_p_5: pd.Series
    slope_v_5: pd.Series
    convexity: pd.Series  # Adding this now to prevent the next error!

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
        """Returns RAW data (For Plots/Debug)."""
        try:
            return self.formula(obs)
        except Exception:
            target_index = (
                obs.rsi.index if hasattr(obs, "rsi") else obs.lookback_close.columns
            )
            return pd.Series([float("nan")] * len(target_index), index=target_index)

    def get_agent_view(self, obs, config: "TradingConfig") -> pd.Series:
        """Returns SCALED, CLEANED, and CLIPPED data for the RL Agent."""
        from core.quant import QuantUtils

        raw = self.__call__(obs)

        # 1. PRE-CLEAN: Remove Inf before they poison the batch statistics
        # We replace inf with NaN so .mean() and .std() ignore them
        # Explicitly hint clean_raw as a Series here
        clean_raw: pd.Series = raw.replace([np.inf, -np.inf], np.nan)

        # 2. Scaling Logic (Using CLEAN data)
        if self.scaling_type == "Z-Score":
            # Cross-sectional standardization using robust helper
            scaled = QuantUtils.zscore(clean_raw)

        elif self.scaling_type == "Center":
            # Map [0, 1] to [-1, 1] (e.g. Range Position)
            scaled = (clean_raw - 0.5) * 2
        elif self.scaling_type == "RSI":
            # Map [-100, 0] to [1, -1]
            # RSI 30 (Oversold) -> -RSI -70 -> Scaled +1.0
            # RSI 70 (Overbought) -> -RSI -30 -> Scaled -1.0
            scaled = (clean_raw + 50) / 20
        else:
            scaled = clean_raw

        # Tell Pylance exactly what 'scaled' is
        scaled: pd.Series = scaled

        # 3. Final Neutralization and Clipping
        # - Missing/Inf data becomes 0 (Neutral)
        # - Outliers capped at +/- feature_zscore_clip
        # USE THE SINGLE SOURCE OF TRUTH
        clip_val = config.feature_zscore_clip

        return scaled.fillna(0).clip(-clip_val, clip_val)


#
