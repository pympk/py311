import pandas as pd
import numpy as np

from typing import Union, Tuple
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

    @staticmethod
    def calculate_rsi(series: pd.Series, period: int) -> pd.Series:
        """Wilder's RSI logic."""
        delta = series.diff()
        up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
        # alpha=1/period is the standard Wilder's smoothing
        ma_up = up.ewm(alpha=1 / period, adjust=False).mean()
        ma_down = down.ewm(alpha=1 / period, adjust=False).mean()
        rs = ma_up / ma_down
        rsi = 100 - (100 / (1 + rs))
        return rsi.replace({np.inf: 100, -np.inf: 0}).fillna(50)

    @staticmethod
    def calculate_tr(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """True Range (TR)."""
        prev_close = close.shift(1)
        # Vectorized max of (H-L, |H-Cp|, |L-Cp|)
        tr = np.maximum(
            high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs())
        )
        return tr

    @staticmethod
    def calculate_atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int
    ) -> pd.Series:
        """Average True Range (ATR) using Wilder's Smoothing."""
        tr = QuantUtils.calculate_tr(high, low, close)  # Use the TR kernel
        return tr.ewm(alpha=1 / period, adjust=False).mean()

    @staticmethod
    def calculate_rolling_beta(
        rets: Union[pd.Series, pd.DataFrame], benchmark_rets: pd.Series, window: int
    ) -> Union[pd.Series, pd.DataFrame]:
        """Standard Rolling Beta: Cov(r, m) / Var(m)."""
        cov = rets.rolling(window).cov(benchmark_rets)
        var = benchmark_rets.rolling(window).var()

        # If rets is a DataFrame, we must specify axis=0 for the division
        if isinstance(rets, pd.DataFrame):
            return cov.div(var, axis=0).fillna(1.0)
        return (cov / var).fillna(1.0)


class TickerEngine:
    """
    The Orchestrator: Bridges MultiIndex DataFrames with QuantUtils Kernels.
    """

    @staticmethod
    def map_kernels(data, kernel_func, *args, **kwargs):
        # The standardized bridge pattern we tested
        return data.groupby(level="Ticker", group_keys=False).apply(
            lambda x: kernel_func(x, *args, **kwargs)
        )


#
