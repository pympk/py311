import os

from dataclasses import dataclass, field


@dataclass
class CacheConfig:
    """
    [SHARED CONSTANTS] A single source of truth for the dataset slices
    and feature lookback window.
    """

    LOOKBACK: int = int(os.getenv("CACHE_LOOKBACK", 10))
    START_DATE: str = os.getenv("CACHE_START_DATE", "2026-01-01")
    END_DATE: str = os.getenv("CACHE_END_DATE", "2023-01-01")

    @classmethod
    def get_filename(cls) -> str:
        """Generates a standardized, descriptive parquet filename."""
        import pandas as pd

        start_yr = pd.Timestamp(cls.START_DATE).strftime("%Y")
        return f"alpha_cache_{cls.LOOKBACK}d_{start_yr}.parquet"


@dataclass
class StrategyParams:
    standard_confidence: float = 1.0
    strong_confidence: float = 1.5
    extreme_confidence: float = 2.5
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    range_high: float = 0.8
    range_low: float = 0.2
    convexity_exit: float = -0.7


@dataclass
class QualityThresholds:
    min_median_dollar_volume: int = 1_000_000
    min_liquidity_percentile: float = 0.40
    max_stale_pct: float = 0.05
    max_same_vol_count: int = 10


@dataclass
class TradingConfig:
    # ENVIRONMENT
    benchmark_ticker: str = "SPY"
    calendar_ticker: str = "SPY"

    # DATA SANITIZER
    handle_zeros_as_nan: bool = True
    max_data_gap_ffill: int = 1
    nan_price_replacement: float = 0.0

    # STRATEGY & MATH
    annual_period: int = 252
    atr_period: int = 14
    rsi_period: int = 14
    range_pos_period: int = 20

    # FEATURE ENGINE WINDOWS
    win_5d: int = 5
    win_21d: int = 21
    win_63d: int = 63

    # FEATURE GUARDRAILS (CLIPS)
    feature_zscore_clip: float = 4.0
    feature_ratio_clip: float = 10.0

    # QUALITY/LIQUIDITY
    quality_window: int = 252
    quality_min_periods: int = 126

    # STRATEGY PARAMETERS & THRESHOLDS
    strategy_params: StrategyParams = field(default_factory=StrategyParams)
    thresholds: QualityThresholds = field(default_factory=QualityThresholds)
