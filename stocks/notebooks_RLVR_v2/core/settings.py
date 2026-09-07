import os
from dataclasses import dataclass, field


@dataclass
class CacheConfig:
    """
    [SHARED CONSTANTS] A single source of truth for the dataset slices
    and feature lookback window.
    """

    LOOKBACK: int = int(os.getenv("CACHE_LOOKBACK", 41))
    START_DATE: str = os.getenv("CACHE_START_DATE", "1998-01-01")
    END_DATE: str = os.getenv("CACHE_END_DATE", "2030-01-01")

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

    # ------------------------------------------------------------------------
    # RULE: SYSTEM-WIDE NaN & ZERO HANDLING POLICY
    # ------------------------------------------------------------------------
    # 1. Prices (Open, High, Low, Close) MUST remain NaN if the asset did not
    #    trade (e.g., Pre-IPO, Halted, Delisted).
    # 2. NEVER fill missing prices with 0.0. This causes division-by-zero,
    #    infinite returns, and breaks pipeline mathematics.
    # 3. Features (like ATRP, TRP, RSI) can be filled with 0.0 where appropriate
    #    to allow cross-sectional math to proceed without dropping the row.
    # 4. Use .bfill() in portfolio simulation (QuantUtils) only to align newly
    #    IPO'd stocks mid-period against the initial capital allocation.
    # ------------------------------------------------------------------------
    nan_price_replacement: float = 0.0  # DEPRECATED/UNSAFE: DO NOT USE FOR PRICES

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

    # TRAINING & SIMULATION PARAMETERS (Strategic Pivot Calibrated)
    holding_period: int = 5
    rank_max_offset_percentile: float = 1.0
    rank_max_width: int = 10
    min_basket_width: int = 5  # Diversify: 5 to 10 stocks dampens idiosyncratic drag
    max_cash_pct: float = 0.0  # 100% Gross Equity Exposure (Zero Cash)
    min_active_tilt: float = (
        0.20  # Flexibility: Allows up to 80% benchmark beta shelter
    )

    # ENVIRONMENT CONTROLS
    randomize_start: bool = False
    episode_steps: int = 0

    # INSTITUTIONAL RL REWARD PARAMETERS
    slippage_rate: float = 0.0010  # 10 bps round-trip slippage
    loss_aversion_penalty: float = 0.0  # 0.0 = Linear spread
    upside_alpha_mult: float = 1.0  # 1.0 = Strict linear spread (NO convex gambling)

    # GAE & PPO HYPERPARAMETERS
    gamma: float = 0.90
    gae_lambda: float = 0.95
    learning_rate: float = 2.0e-4
    critic_learning_rate: float = 8.0e-4

    # ENTROPY ANNEALING SCHEDULE
    entropy_coef: float = 0.0050
    entropy_coef_start: float = 0.0050  # Broader exploration
    entropy_coef_end: float = 0.0005  # Controlled convergence

    # PPO BATCHING & LOSS COEFFICIENTS
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    num_steps: int = 512  # 512 steps * 8 envs = 4,096 timesteps per epoch
    num_envs: int = 8
    mini_batch_size: int = 256

    @property
    def dynamic_gamma(self) -> float:
        """
        Returns self.gamma (0.98) to prevent accidental override
        to legacy overlapping horizon values.
        """
        return self.gamma

    @property
    def benchmark(self) -> str:
        """Dynamic alias for benchmark_ticker conforming to TradingConfig domain model."""
        return self.benchmark_ticker
