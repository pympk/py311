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
# def calculate_true_range(high, low, prev_close):
#     """
#     Calculates the True Range from high, low, and previous close data.
#     Works for both pandas Series and DataFrames.
#     """
#     component1 = high - low
#     component2 = abs(high - prev_close)
#     component3 = abs(low - prev_close)
    
#     # np.maximum is element-wise and works perfectly on both Series and DataFrames.
#     tr = np.maximum(component1, np.maximum(component2, component3))
#     return tr

def calculate_true_range(high, low, prev_close):
    """
    Calculates the True Range from high, low, and previous close data.
    Works for both pandas Series and DataFrames.
    This version correctly handles the first row by treating NaN in prev_close components as 0.
    """
    component1 = high - low
    
    # On the first row, prev_close is NaN. By filling the resulting NaN with 0,
    # we ensure these components don't propagate the NaN and are ignored in the max() calculation,
    # which correctly makes the first TR value equal to (high - low).
    component2 = abs(high - prev_close).fillna(0)
    component3 = abs(low - prev_close).fillna(0)
    
    tr = np.maximum(component1, np.maximum(component2, component3))
    return tr


# --- Ranking Metric Calculation Functions ---

def calculate_price_rank(df_close, **kwargs):
    """Calculates the ranking score based on simple price change."""
    first_prices = df_close.bfill().iloc[0]
    last_prices = df_close.ffill().iloc[-1]
    scores = (last_prices / first_prices)
    return scores.dropna()

def calculate_sharpe_rank(df_close, **kwargs):
    """Calculates the ranking score based on the annualized Sharpe ratio."""
    daily_returns = df_close.bfill().ffill().pct_change()
    mean_returns = daily_returns.mean()
    std_returns = daily_returns.std()
    # Replace division by zero with 0, as it implies no risk and no return.
    scores = (mean_returns / std_returns * np.sqrt(252)).fillna(0)
    return scores

def calculate_sharpe_atr_rank(df_close, df_high, df_low, **kwargs):
    """Calculates the ranking score based on Sharpe ratio using ATR for volatility."""
    # Dependency: Mean daily return
    daily_returns = df_close.bfill().ffill().pct_change()
    mean_returns = daily_returns.mean()
    
    # Dependency: Mean ATR Percent (ATRP)
    # Note: We use df_close.shift(1) here on the SLICED data, which is correct
    # for this self-contained calculation.
    prev_close = df_close.shift(1)
    
    tr = calculate_true_range(df_high, df_low, prev_close)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    atrp = (atr / df_close).mean() # Mean ATRP over the calculation period
    
    scores = (mean_returns / atrp).fillna(0)
    return scores

# --- The Metric Registry ---
# This dictionary maps the user-facing metric names to the functions that calculate them.
# To add a new metric, define a function above and add an entry here.

METRIC_REGISTRY = {
    'Price': calculate_price_rank,
    'Sharpe': calculate_sharpe_rank,
    'Sharpe (ATR)': calculate_sharpe_atr_rank,
}