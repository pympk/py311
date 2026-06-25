import os
import pandas as pd


def get_master_trading_calendar(df_ohlcv, calendar_ticker="SPY"):
    """Extracts a guaranteed clean trading calendar based on the config ticker."""
    if df_ohlcv.index.get_level_values("Date").tz is not None:
        df_ohlcv.index = df_ohlcv.index.set_levels(
            pd.to_datetime(df_ohlcv.index.levels[1]).tz_localize(None), level="Date"
        )
    master_dates = (
        df_ohlcv.xs(calendar_ticker, level="Ticker").index.unique().sort_values()
    )
    return pd.DatetimeIndex(master_dates)


def get_chronological_splits(
    trading_calendar, feature_cube_dates, holding_period=0, min_dates_threshold=100
):
    """
    Returns strict Train/Val/Test splits.

    If DEBUG_MODE is active in the environment, or if the available
    calendar size is too small to split safely (based on min_dates_threshold),
    it falls back to overlapping splits to allow for pipeline execution and debugging.
    """
    # Filter to only include dates present in the feature cube
    valid_cal = trading_calendar[trading_calendar.isin(feature_cube_dates)]
    valid_cal = valid_cal.sort_values()  # Ensure strict chronological order

    # 1. Check for explicit debug overrides or implicit data scarcity
    is_debug = os.getenv("DEBUG_MODE", "false").lower() in ("true", "1")
    too_small = len(valid_cal) < min_dates_threshold

    if is_debug or too_small:
        print(
            f"[INFO] Entering Debug/Scarcity mode "
            f"(DEBUG_MODE={is_debug}, Total Dates={len(valid_cal)}) less than min_dates_threshold of {min_dates_threshold}. "
            f"Identical calendars will be used across all splits."
        )
        # Return identical calendars to bypass splits and allow debugging of the loops
        return valid_cal, valid_cal, valid_cal

    # 2. Standard Proportional Slicing (e.g., 70% Train, 15% Val, 15% Test)
    total_len = len(valid_cal)
    train_end_idx = int(total_len * 0.70)
    val_end_idx = int(total_len * 0.85)

    # Slice the Index directly without .iloc
    cal_train = valid_cal[:train_end_idx]

    # 3. Apply Purge Gaps/Holding Period Adjustments if applicable
    val_start_idx = train_end_idx + holding_period
    test_start_idx = val_end_idx + holding_period

    # Safe boundary checks to prevent slicing past limits or empty slices
    if val_start_idx < val_end_idx:
        cal_val = valid_cal[val_start_idx:val_end_idx]
    else:
        # Fallback to standard slice if the holding period gap is too large for the subset
        cal_val = valid_cal[train_end_idx:val_end_idx]

    if test_start_idx < total_len:
        cal_test = valid_cal[test_start_idx:]
    else:
        # Fallback to standard slice if the holding period gap is too large for the subset
        cal_test = valid_cal[val_end_idx:]

    return cal_train, cal_val, cal_test
