import pandas as pd
import numpy as np

from typing import Union, Tuple, List
from collections import Counter  # <--- Add this import!
from core.settings import GLOBAL_SETTINGS


class QuantUtils:
    """
    MATHEMATICAL KERNEL REGISTRY: THE SINGLE SOURCE OF TRUTH.
    Handles both pd.Series (Report) and pd.DataFrame (Ranking) robustly.
    """

    @staticmethod
    def compute_returns(
        data: Union[pd.Series, pd.DataFrame],
    ) -> Union[pd.Series, pd.DataFrame]:
        return data.pct_change().replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def calculate_gain(
        data: Union[pd.Series, pd.DataFrame], min_points: int = 2
    ) -> Union[float, pd.Series]:
        """
        Calculates the logarithmic return between the first and last valid points.
        Returns 0.0 if data is empty or has fewer than min_points valid entries.
        """
        if data.empty:
            return 0.0

        # Recursive step: If input is a DataFrame, calculate gain for each column
        if isinstance(data, pd.DataFrame):
            column_gains = data.apply(
                lambda col: QuantUtils.calculate_gain(col, min_points)
            )
            # Note: 'column_gains' is actually a Series where index = DataFrame columns
            return column_gains

        # Series logic: Remove NaNs to identify first and last available values
        clean = data.dropna()

        # Validation: Ensure enough non-null points exist for a valid calculation
        if len(clean) < min_points:
            return 0.0

        if len(clean) < min_points:
            return 0.0

        # --- FIX START ---
        first_val = clean.iloc[0]
        last_val = clean.iloc[-1]

        # Logarithms require positive numbers.
        # If price is 0 (or negative), the investment is effectively a total loss (-inf).
        # We handle this to avoid the RuntimeWarning.
        if first_val <= 0 or last_val <= 0:
            # If the price dropped to 0, return a very large negative number
            # or -1.0 (representing 100% loss) depending on your preference.
            return -10.0

        # Calculate Logarithmic Gain: ln(last_price / first_price)
        # This represents the continuously compounded rate of return
        gain_val = float(np.log(clean.iloc[-1] / clean.iloc[0]))

        return gain_val

    @staticmethod
    def calculate_sharpe(
        data: Union[pd.Series, pd.DataFrame],
        periods: int = None,  # Default to None to trigger global lookup
    ) -> Union[float, pd.Series]:
        if periods is None:
            periods = GLOBAL_SETTINGS["annual_period"]
        mu, std = data.mean(), data.std()
        # Use np.maximum for universal floor (works on scalars and Series)
        res = (mu / np.maximum(std, 1e-8)) * np.sqrt(periods)

        if isinstance(res, (pd.Series, pd.DataFrame)):
            return res.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return float(res) if np.isfinite(res) else 0.0

    @staticmethod
    def calculate_sharpe_vol(
        returns: Union[pd.Series, pd.DataFrame],
        vol_data: Union[pd.Series, pd.DataFrame],
    ) -> Union[float, pd.Series]:
        """
        Aligned Reward / Risk.
        Filters out volatility observations where no return exists (e.g. Day 1 NaN).
        """
        # 1. Identify valid timestamps (Pandas .mean() skips NaNs in returns)
        # but we must manually force the volatility denominator to skip those same rows.
        mask = returns.notna()
        avg_ret = returns.mean()

        # 2. Handle Logic Branches
        if isinstance(returns, pd.DataFrame) and isinstance(vol_data, pd.Series):
            # RANKING MODE: vol_data is usually a pre-calculated snapshot Series
            avg_vol = vol_data
        else:
            # REPORT MODE (Series) or Cross-Sectional DataFrame
            # Filter vol_data to only include rows where returns exist
            avg_vol = vol_data.where(mask).mean()

        # 3. Final Division
        res = avg_ret / np.maximum(avg_vol, 1e-8)

        if isinstance(res, (pd.Series, pd.DataFrame)):
            return res.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return float(res) if np.isfinite(res) else 0.0

    @staticmethod
    def compute_portfolio_stats(
        prices: pd.DataFrame,
        atrp_matrix: pd.DataFrame,
        trp_matrix: pd.DataFrame,
        weights: pd.Series,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        MATRIX KERNEL: Calculates equity curve and weighted volatility.
        """
        # 1. Equity Curve Logic (Price-Weighted Drift)
        norm_prices = prices.div(prices.bfill().iloc[0])
        weighted_components = norm_prices.mul(weights, axis=1)
        equity_curve = weighted_components.sum(axis=1)

        # MANDATORY: Use internal compute_returns to preserve boundary NaN
        returns_WITH_BOUNDARY_NAN = QuantUtils.compute_returns(equity_curve)

        # 2. Portfolio Volatility Logic (Weighted Average)
        # We calculate current_weights (rebalanced daily by price drift)
        current_weights = weighted_components.div(equity_curve, axis=0)

        # Weighted average of ATRP and TRP
        portfolio_atrp = (current_weights * atrp_matrix).sum(axis=1, min_count=1)
        portfolio_trp = (current_weights * trp_matrix).sum(axis=1, min_count=1)

        return equity_curve, returns_WITH_BOUNDARY_NAN, portfolio_atrp, portfolio_trp


def _prepare_initial_weights(tickers: List[str]) -> pd.Series:
    """
    METADATA: Converts a list of tickers into a weight map.
    Example: ['AAPL', 'AAPL', 'TSLA'] -> {'AAPL': 0.66, 'TSLA': 0.33}
    """
    ticker_counts = Counter(tickers)
    total = len(tickers)
    return pd.Series({t: c / total for t, c in ticker_counts.items()})


# --- STANDALONE WORKFLOW FUNCTION ---
def calculate_buy_and_hold_performance(
    df_close_wide: pd.DataFrame,  # Use the WIDE version
    df_atrp_wide: pd.DataFrame,  # Use the WIDE version
    df_trp_wide: pd.DataFrame,  # <--- Added
    tickers: List[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
):
    if not tickers:
        return pd.Series(), pd.Series(), pd.Series()

    initial_weights = _prepare_initial_weights(tickers)

    # SLICE (Fix Part B)
    ticker_list = initial_weights.index.tolist()
    p_slice = df_close_wide.reindex(columns=ticker_list).loc[start_date:end_date]
    a_slice = df_atrp_wide.reindex(columns=ticker_list).loc[start_date:end_date]
    t_slice = df_trp_wide.reindex(columns=ticker_list).loc[start_date:end_date]
    # KERNEL - Pure Math
    return QuantUtils.compute_portfolio_stats(
        p_slice, a_slice, t_slice, initial_weights
    )
