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
    # QUALITY THRESHOLDS (The "Rules")
    "thresholds": {
        # HARD LIQUIDITY FLOOR
        # Logic: Calculates (Adj Close * Volume) daily, then takes the ROLLING MEDIAN
        # over the quality_window (252 days). Filters out stocks where the
        # typical daily dollar turnover is below this absolute value.
        "min_median_dollar_volume": 1_000_000,
        # DYNAMIC LIQUIDITY CUTOFF (Relative to Universe)
        # Logic: On the decision date, the engine calculates the X-quantile
        # of 'RollMedDollarVol' across ALL available stocks.
        # Setting this to 0.40 calculates the 60th percentile and requires
        # stocks to be above it—effectively keeping only the TOP 60% of the market.
        "min_liquidity_percentile": 0.40,
        # PRICE/VOLUME STALENESS
        # Logic: Creates a binary flag (1 if Volume is 0 OR High equals Low).
        # It then calculates the ROLLING MEAN of this flag.
        # A value of 0.05 means the stock is rejected if it was "stale"
        # for more than 5% of the trading days in the rolling window.
        "max_stale_pct": 0.05,
        # DATA INTEGRITY (FROZEN VOLUME)
        # Logic: Checks if Volume is identical to the previous day (Volume.diff() == 0).
        # It calculates the ROLLING SUM of these occurrences over the window.
        # If the exact same volume is reported more than 10 times, the stock
        # is rejected as having "frozen" or low-quality data.
        "max_same_vol_count": 10,
    },
}
