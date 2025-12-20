"""
This module contains standalone functions for calculating various ranking metrics.
Each function adheres to a standard interface, accepting a dictionary of OHLCV
DataFrame slices and returning a pandas Series of scores for each ticker.
This allows for easy extension by adding new functions and registering them.
"""

import pandas as pd
import numpy as np


# --- Reusable Helper from Main Codebase ---
# This helper is now centralized and used by metric calculators.
def calculate_true_range(high, low, prev_close):
    """
    Calculates the True Range from high, low, and previous close data.
    Works for both pandas Series and DataFrames.
    """
    component1 = high - low
    component2 = abs(high - prev_close)
    component3 = abs(low - prev_close)

    # np.maximum is element-wise and works perfectly on both Series and DataFrames.
    tr = np.maximum(component1, np.maximum(component2, component3))
    return tr


# --- NEW: The single, authoritative function for ATR calculation ---
def calculate_atr_series(high_series, low_series, close_series, period=14):
    """
    Calculates the ATR for a full series of price data.
    This function correctly handles the first row convention (TR = High - Low).
    It expects full, unsliced pandas Series.
    """
    prev_close = close_series.shift(1)

    # Use the pure helper to calculate the raw TR
    tr = calculate_true_range(high_series, low_series, prev_close)

    # --- THIS IS THE CENTRALIZED FIX ---
    # The first value of a TR series is conventionally just High - Low.
    # We explicitly set it here on the full series.
    tr.iloc[0] = high_series.iloc[0] - low_series.iloc[0]
    # --- END OF FIX ---

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()

    # --- THIS IS THE FIX YOU CORRECTLY IDENTIFIED ---
    # The first ATR value is conventionally seeded with the first TR value.
    # We explicitly set it here to ensure the calculation is standard and verifiable.
    atr.iloc[0] = tr.iloc[0]
    # --- END OF FIX ---

    return atr


# --- Ranking Metric Calculation Functions ---


def calculate_price_rank(df_close, **kwargs):
    """Calculates the ranking score based on simple price change."""
    first_prices = df_close.bfill().iloc[0]
    last_prices = df_close.ffill().iloc[-1]
    scores = last_prices / first_prices
    return scores.dropna()


def calculate_sharpe_rank(df_close, **kwargs):
    """Calculates the ranking score based on the annualized Sharpe ratio."""
    daily_returns = df_close.bfill().ffill().pct_change()
    mean_returns = daily_returns.mean()
    std_returns = daily_returns.std()
    # Replace division by zero with 0, as it implies no risk and no return.
    scores = (mean_returns / std_returns * np.sqrt(252)).fillna(0)
    return scores


# def calculate_sharpe_atr_rank(df_close, df_high, df_low, **kwargs):
#     """Calculates the ranking score based on Sharpe ratio using ATR for volatility."""
#     # Dependency: Mean daily return
#     daily_returns = df_close.bfill().ffill().pct_change()
#     mean_returns = daily_returns.mean()

#     # Dependency: Mean ATR Percent (ATRP)
#     # Note: We use df_close.shift(1) here on the SLICED data, which is correct
#     # for this self-contained calculation.
#     prev_close = df_close.shift(1)

#     tr = calculate_true_range(df_high, df_low, prev_close)
#     atr = tr.ewm(alpha=1/14, adjust=False).mean()
#     atrp = (atr / df_close).mean() # Mean ATRP over the calculation period

#     scores = (mean_returns / atrp).fillna(0)
#     return scores

# def calculate_sharpe_atr_rank(df_close, df_high, df_low, df_close_full, **kwargs):
#     """
#     Calculates the ranking score based on Sharpe ratio using ATR for volatility.
#     NOTE: Requires the full, unsliced df_close_full to correctly calculate prev_close.
#     """
#     # Dependency: Mean daily return (uses the sliced data)
#     daily_returns = df_close.pct_change()
#     mean_returns = daily_returns.mean()

#     # --- THIS IS THE CORRECTED LOGIC ---
#     # 1. Shift the FULL history to get the true previous day's close.
#     prev_close_full = df_close_full.shift(1)

#     # 2. Slice this shifted series to align with our calculation period.
#     #    This correctly provides the day before the period starts for the first row.
#     prev_close_slice = prev_close_full.loc[df_close.index]

#     # 3. Now call the pure helper function with perfectly prepared data.
#     tr = calculate_true_range(df_high, df_low, prev_close_slice)

#     # The first row of TR is now correctly calculated as (High - Low) because
#     # component2 and component3 will use the real previous day's close.
#     # If the previous day's close doesn't exist (very first day of data),
#     # the TR will be NaN, which is acceptable and will be handled by fillna(0) later.
#     # To be fully robust, let's explicitly handle the first-row-ever case.
#     if tr.iloc[0].isnull().any():
#         tr.iloc[0] = df_high.iloc[0] - df_low.iloc[0]

#     atr = tr.ewm(alpha=1/14, adjust=False).mean()
#     atrp = (atr / df_close).mean()

#     scores = (mean_returns / atrp).fillna(0)
#     return scores

# def calculate_sharpe_atr_rank(df_close, df_high, df_low, df_close_full, **kwargs):
#     """Calculates the ranking score based on Sharpe ratio using ATR for volatility."""
#     daily_returns = df_close.pct_change()
#     mean_returns = daily_returns.mean()

#     # --- SIMPLIFIED LOGIC ---
#     # 1. Calculate the full ATR series for all tickers using the new authoritative function.
#     #    We apply the function to each column (ticker) of the full DataFrame.
#     full_atr_series = df_close_full.copy() # Create a placeholder DataFrame of the right shape
#     for ticker in df_close_full.columns:
#         full_atr_series[ticker] = calculate_atr_series(
#             df_high_full[ticker], df_low_full[ticker], df_close_full[ticker], period=14
#         )

#     # 2. Slice the resulting full ATR series to match our calculation period.
#     atr_slice = full_atr_series.loc[df_close.index]

#     # 3. Calculate ATRP and the final score.
#     atrp = (atr_slice / df_close).mean()
#     scores = (mean_returns / atrp).fillna(0)
#     return scores


def calculate_sharpe_atr_rank(
    df_close, df_high, df_low, df_high_full, df_low_full, df_close_full, **kwargs
):
    """Calculates the ranking score based on Sharpe ratio using ATR for volatility."""
    daily_returns = df_close.pct_change()
    mean_returns = daily_returns.mean()

    # Create a placeholder DataFrame of the right shape
    full_atr_series = df_close_full.copy()
    for ticker in df_close_full.columns:
        # --- MODIFIED --- Now these variables are correctly defined and passed in
        full_atr_series[ticker] = calculate_atr_series(
            df_high_full[ticker], df_low_full[ticker], df_close_full[ticker], period=14
        )

    # Slice the resulting full ATR series to match our calculation period.
    atr_slice = full_atr_series.loc[df_close.index]

    # Calculate ATRP and the final score.
    atrp = (atr_slice / df_close).mean()
    scores = (mean_returns / atrp).fillna(0)
    return scores


# --- The Metric Registry ---
# This dictionary maps the user-facing metric names to the functions that calculate them.
# To add a new metric, define a function above and add an entry here.

METRIC_REGISTRY = {
    "Price": calculate_price_rank,
    "Sharpe": calculate_sharpe_rank,
    "Sharpe (ATR)": calculate_sharpe_atr_rank,
}
