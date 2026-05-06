from pathlib import Path

GLOBAL_SETTINGS = {
    # ENVIRONMENT (The "Where")
    "benchmark_ticker": "SPY",
    "calendar_ticker": "SPY",  # Used as the "Master Clock" for trading days
    # DATA SANITIZER (The "Glitches & Gaps" Protector)
    "handle_zeros_as_nan": True,  # Convert 0.0 prices to NaN to prevent math errors
    "max_data_gap_ffill": 1,  # Max consecutive days to "Forward Fill" missing data
    # IMPLICATION OF nan_price_replacement:
    # - This defines what happens if the "Forward Fill" limit is exceeded.
    # - If set to 0.0: A permanent data gap will look like a "total loss" (-100%).
    #   The equity curve will plummet. Good for "disaster detection."
    #   Sharpe and Sharpe(ATR) drop because: return (gets smaller) / std (gets larger)
    # - If set to np.nan: A permanent gap will cause portfolio calculations to return NaN.
    #   The chart may break or show gaps. Good for "math integrity."
    "nan_price_replacement": 0.0,
    # STRATEGY & MATH
    "annual_period": 252,  # Replaces hardcoded 252 in Sharpe calculations
    "atr_period": 14,  # Used for volatility normalization
    "rsi_period": 14,  # <--- NEW: Control for RSI logic
    "range_pos_period": 20,  # Range position
    # FEATURE ENGINE WINDOWS
    "5d_window": 5,  # Replaces hardcoded 5 ("Weekly" anchor)
    "21d_window": 21,  # Replaces hardcoded 21 ("Monthly" anchor)
    "63d_window": 63,  # Replaces hardcoded 63 ("3 Monthly" anchor)
    # FEATURE GUARDRAILS (CLIPS)
    "feature_zscore_clip": 4.0,  # Replaces hardcoded 4.0 in OBV Z-Scores
    "feature_ratio_clip": 10.0,  # Replaces hardcoded 10.0 in RVol ratios
    # QUALITY/LIQUIDITY
    "quality_window": 252,  # 1 year lookback for liquidity/quality stats
    "quality_min_periods": 126,  # min period that ticker has to meet quality thresholds
    # STRATEGY PARAMETERS (The "Levers")
    "strategy_params": {
        "standard_confidence": 1.0,  # Default Z-Score trigger (1.0std)
        "strong_confidence": 1.5,  # Strong Z-Score trigger (1.5std)
        "extreme_confidence": 2.5,  # Parabolic/Extreme risk trigger (2.5std)
        "rsi_overbought": 70,  # Standard RSI Upper Bound
        "rsi_oversold": 30,  # Standard RSI Lower Bound
        "range_high": 0.8,  # Range Position Upper Bound
        "range_low": 0.2,  # Range Position Lower Bound
        "convexity_exit": -0.7,  # Deceleration/Exhaustion threshold
    },
    # QUALITY THRESHOLDS (The "Rules")
    "thresholds": {
        # HARD LIQUIDITY FLOOR
        "min_median_dollar_volume": 1_000_000,
        # DYNAMIC LIQUIDITY CUTOFF (Relative to Universe)
        "min_liquidity_percentile": 0.40,
        # PRICE/VOLUME STALENESS
        "max_stale_pct": 0.05,
        # DATA INTEGRITY (FROZEN VOLUME)
        "max_same_vol_count": 10,
    },
}

def find_notebooks_root():
    """Find notebooks_RLVR_v2 root from any location."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if parent.name == "notebooks_RLVR_v2":
            return parent
    raise RuntimeError("Could not find notebooks_RLVR_v2 directory")


NOTEBOOKS_ROOT = find_notebooks_root()
OUTPUT_DIR = NOTEBOOKS_ROOT / "output"

# Create dir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
