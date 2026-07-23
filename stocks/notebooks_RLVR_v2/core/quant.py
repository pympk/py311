import pandas as pd
import numpy as np
import warnings

from typing import Union, Tuple, overload, Optional, cast


class QuantUtils:
    """
    MATHEMATICAL KERNEL REGISTRY: THE SINGLE SOURCE OF TRUTH.
    Handles both pd.Series (Report) and pd.DataFrame (Ranking) robustly.
    """

    @overload
    @staticmethod
    def compute_returns(data: pd.Series) -> pd.Series: ...

    @overload
    @staticmethod
    def compute_returns(data: pd.DataFrame) -> pd.DataFrame: ...

    @staticmethod
    def compute_returns(
        data: Union[pd.Series, pd.DataFrame],
    ) -> Union[pd.Series, pd.DataFrame]:
        # We use cast here internally because Pandas methods like .replace()
        # often confuse the type checker's ability to track Series vs DataFrame
        res = data.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        return cast(Union[pd.Series, pd.DataFrame], res)

    @overload
    @staticmethod
    def calculate_gain(data: pd.Series, min_points: int = 2) -> float: ...

    @overload
    @staticmethod
    def calculate_gain(data: pd.DataFrame, min_points: int = 2) -> pd.Series: ...

    @staticmethod
    def calculate_gain(
        data: Union[pd.Series, pd.DataFrame], min_points: int = 2
    ) -> Union[float, pd.Series]:
        if data.empty:
            return 0.0

        if isinstance(data, pd.DataFrame):
            # The result of apply on a DataFrame is a Series
            return cast(
                pd.Series,
                data.apply(lambda col: QuantUtils.calculate_gain(col, min_points)),
            )

        clean = data.dropna()
        if len(clean) < min_points:
            return 0.0

        first_val = clean.iloc[0]
        last_val = clean.iloc[-1]

        if first_val <= 0 or last_val <= 0:
            return -10.0

        return float(np.log(last_val / first_val))

    @staticmethod
    def calculate_sharpe(
        data: Union[pd.Series, pd.DataFrame],
        periods: Optional[int] = None,
    ) -> Union[float, pd.Series]:
        """
        Calculates Sharpe Ratio.
        If data is a DataFrame, returns a Series of Sharpe Ratios.
        If data is a Series, returns a single float Sharpe Ratio.
        """
        if periods is None:
            periods = 252

        # CASE 1: Data is a DataFrame (Result is a Series)
        if isinstance(data, pd.DataFrame):
            mu = data.mean()  # Result: pd.Series
            std = data.std()  # Result: pd.Series
            res = (mu / np.maximum(std, 1e-8)) * np.sqrt(periods)

            # Clean and cast to Series for Pylance
            cleaned = res.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            return cast(pd.Series, cleaned)

        # CASE 2: Data is a Series (Result is a float)
        elif isinstance(data, pd.Series):
            mu = float(data.mean())  # Result: float
            std = float(data.std())  # Result: float
            res = (mu / max(std, 1e-8)) * np.sqrt(periods)

            return res if np.isfinite(res) else 0.0

        else:
            raise TypeError("Input 'data' must be a pandas Series or DataFrame.")

    @staticmethod
    def calc_sharpe_cross_section(
        returns: pd.DataFrame, vol_vector: pd.Series
    ) -> pd.Series:
        """
        [RANKING KERNEL] Calculates Sharpe ratio using a static volatility vector.

        Use Case: Ranking many tickers (DataFrame) against their current TRP/ATRP (Series).
        Logic: Mean(Returns) / Vol_Vector.
        Performance: High-speed NumPy vectorization.
        """

        # DEBUG TRAP: Ensure Tickers are in the same order
        if not (returns.columns.equals(vol_vector.index)):
            raise ValueError(
                f"Cross-section Alignment Mismatch!\n"
                f"Returns Tickers (first 3): {list(returns.columns[:3])}\n"
                f"Vol Index Tickers (first 3): {list(vol_vector.index[:3])}"
            )
        # 1. Extract strictly-typed float arrays to satisfy Pylance
        ret_arr = returns.to_numpy(dtype=float)  # Shape: (Time, Tickers)
        vol_arr = vol_vector.to_numpy(dtype=float)  # Shape: (Tickers,)

        # 2. Fast C-level math
        avg_ret = np.nanmean(ret_arr, axis=0)
        avg_vol = np.maximum(vol_arr, 1e-8)

        # 3. Calculate and clean infinites/NaNs natively in NumPy
        with np.errstate(divide="ignore", invalid="ignore"):
            res_arr = avg_ret / avg_vol

        cleaned_arr = np.nan_to_num(res_arr, nan=0.0, posinf=0.0, neginf=0.0)

        # 4. Wrap back in Series for engine compatibility
        return pd.Series(cleaned_arr, index=returns.columns)

    @staticmethod
    def calc_sharpe_multivariate_aligned(
        returns: pd.DataFrame, vol_grid: pd.DataFrame
    ) -> pd.Series:
        """
        [RESEARCH KERNEL] Calculates Sharpe ratio using dynamic time-series volatility.

        Use Case: Full-grid vectorized backtesting.
        Logic: Ensures 'Temporal Coupling'—only counts volatility on days where returns
               are non-NaN (prevents the 'Day 1 Trap').
        Performance: O(n) NumPy matrix math.
        """
        # DEBUG TRAP: Ensure columns and index are identical
        # Check if indices/columns match
        if not all(returns.columns == vol_grid.columns):
            print(
                f"Mismatch! Returns: {returns.columns[:3]}... Vol: {vol_grid.columns[:3]}..."
            )
            raise ValueError("Alignment Mismatch")

        ret_arr = returns.to_numpy(dtype=float)
        vol_arr = vol_grid.to_numpy(dtype=float)

        # Create a mask of valid return days
        valid_mask = ~np.isnan(ret_arr)

        # Apply the mask to volatility (ignoring vol on days with NaN returns)
        masked_vol = np.where(valid_mask, vol_arr, np.nan)

        # Calculate means ignoring NaNs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            avg_ret = np.nanmean(ret_arr, axis=0)
            avg_vol = np.nanmean(masked_vol, axis=0)

        avg_vol = np.maximum(avg_vol, 1e-8)

        # Math and cleanup
        with np.errstate(divide="ignore", invalid="ignore"):
            res_arr = avg_ret / avg_vol

        cleaned_arr = np.nan_to_num(res_arr, nan=0.0, posinf=0.0, neginf=0.0)

        return pd.Series(cleaned_arr, index=returns.columns)

    @staticmethod
    def calc_sharpe_univariate(returns: pd.Series, vol_series: pd.Series) -> float:
        """
        [REPORT KERNEL] Calculates a single scalar Sharpe ratio for one asset or portfolio.

        Use Case: Reporting, creating individual ticker stats, or portfolio performance.
        Logic: Standard Univariate Sharpe with NaN-masking.
        Performance: Minimal overhead float calculation.
        """
        # DEBUG TRAP: Ensure Dates are identical
        if not (returns.index.equals(vol_series.index)):
            # Optional: Try to fix it automatically if you prefer
            # returns, vol_series = returns.align(vol_series, join='inner')
            raise ValueError(
                "Univariate Temporal Alignment Mismatch: Indices do not match."
            )
        ret_arr = returns.to_numpy(dtype=float)
        vol_arr = vol_series.to_numpy(dtype=float)

        # Find valid indices (where returns are not NaN)
        valid_mask = ~np.isnan(ret_arr)

        if not np.any(valid_mask):
            return 0.0

        avg_ret = np.mean(ret_arr[valid_mask])
        avg_vol = np.mean(vol_arr[valid_mask])

        res = float(avg_ret / max(avg_vol, 1e-8))

        return res if np.isfinite(res) else 0.0

    @staticmethod
    def compute_portfolio_stats(
        prices: pd.DataFrame,
        atrp_matrix: pd.DataFrame,
        trp_matrix: pd.DataFrame,
        weights: pd.Series,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        norm_prices = prices.div(prices.bfill().iloc[0])
        weighted_components = norm_prices.mul(weights, axis=1)
        equity_curve = weighted_components.sum(axis=1)

        returns_WITH_BOUNDARY_NAN = QuantUtils.compute_returns(equity_curve)
        current_weights = weighted_components.div(equity_curve, axis=0)

        portfolio_atrp = (current_weights * atrp_matrix).sum(axis=1, min_count=1)
        portfolio_trp = (current_weights * trp_matrix).sum(axis=1, min_count=1)

        return equity_curve, returns_WITH_BOUNDARY_NAN, portfolio_atrp, portfolio_trp

    @staticmethod
    def calculate_rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
        ma_up = up.ewm(alpha=1 / period, adjust=False).mean()
        ma_down = down.ewm(alpha=1 / period, adjust=False).mean()
        rs = ma_up / ma_down
        rsi = 100 - (100 / (1 + rs))
        return rsi.replace({np.inf: 100, -np.inf: 0}).fillna(50)

    @staticmethod
    def calculate_tr(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        # 1. Extract underlying NumPy arrays and strictly enforce float dtype
        # This prevents the "cannot convert float NaN to integer" error
        h_arr = high.to_numpy(dtype=float)
        l_arr = low.to_numpy(dtype=float)
        c_arr = close.to_numpy(dtype=float)

        # Handle edge case: empty series
        if len(c_arr) == 0:
            return pd.Series(dtype=float, index=high.index)

        # 2. Shift close by 1 manually (much faster than pd.Series.shift)
        prev_c = np.empty_like(c_arr)
        prev_c[0] = np.nan
        prev_c[1:] = c_arr[:-1]

        # 3. Calculate components
        tr1 = h_arr - l_arr
        tr2 = np.abs(h_arr - prev_c)
        tr3 = np.abs(l_arr - prev_c)

        # 4. np.maximum evaluates element-wise and naturally propagates NaNs.
        tr_arr = np.maximum(tr1, np.maximum(tr2, tr3))

        # 5. Re-wrap as pandas Series to maintain pipeline compatibility
        return pd.Series(tr_arr, index=high.index)

    @staticmethod
    def calculate_atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int
    ) -> pd.Series:
        tr = QuantUtils.calculate_tr(high, low, close)
        return tr.ewm(alpha=1 / period, adjust=False).mean()

    @staticmethod
    def calculate_rolling_beta(
        rets: Union[pd.Series, pd.DataFrame], benchmark_rets: pd.Series, window: int
    ) -> Union[pd.Series, pd.DataFrame]:
        """Standard Rolling Beta: Cov(r, m) / Var(m)."""
        # Safely align benchmark to the exact shape and index of rets
        dates = (
            rets.index.get_level_values("Date")
            if isinstance(rets.index, pd.MultiIndex)
            else rets.index
        )
        aligned_bench = pd.Series(
            benchmark_rets.reindex(dates).values, index=rets.index
        )

        cov = rets.rolling(window).cov(aligned_bench)
        var = aligned_bench.rolling(window).var()

        if isinstance(rets, pd.DataFrame):
            return cov.div(var, axis=0).fillna(1.0)
        return (cov / var).fillna(1.0)

    @staticmethod
    def calculate_rolling_ir(
        rets: pd.Series, benchmark_rets: pd.Series, window: int
    ) -> pd.Series:
        """Information Ratio: Mean(Active Ret) / Std(Active Ret)."""
        # Safely align benchmark to the exact shape and index of rets
        dates = (
            rets.index.get_level_values("Date")
            if isinstance(rets.index, pd.MultiIndex)
            else rets.index
        )
        aligned_bench = pd.Series(
            benchmark_rets.reindex(dates).values, index=rets.index
        )

        active_ret = rets - aligned_bench
        mu = active_ret.rolling(window).mean()
        sigma = active_ret.rolling(window).std()
        return mu / np.maximum(sigma, 1e-8)

    @staticmethod
    def calculate_rolling_sharpe(rets: pd.Series, window: int) -> pd.Series:
        mu = rets.rolling(window).mean()
        sigma = rets.rolling(window).std()
        return mu / np.maximum(sigma, 1e-8)

    @staticmethod
    def calculate_autocorr(
        rets: pd.Series, lag: int = 1, window: int = 15
    ) -> pd.Series:
        return rets.rolling(window=window).corr(rets.shift(lag)).fillna(0.0)

    @staticmethod
    def calculate_range_pos(
        high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20
    ) -> pd.Series:
        roll_min = low.rolling(window=window).min()
        roll_max = high.rolling(window=window).max()
        denom = (roll_max - roll_min).replace(0, 1e-8)
        return (close - roll_min) / denom

    @staticmethod
    def calculate_rolling_slope_5d_fast(series: pd.Series) -> pd.Series:
        return (
            2 * series
            + 1 * series.shift(1)
            + 0 * series.shift(2)
            + -1 * series.shift(3)
            + -2 * series.shift(4)
        ) / 10.0

    @staticmethod
    def calculate_obv_fast(close: pd.Series, volume: pd.Series) -> pd.Series:
        # Pylance sees np.sign as returning an ndarray
        direction = np.sign(close.diff().fillna(0))

        # Multiplying ndarray * Series returns a Series at runtime,
        # but we cast it here so Pylance knows for sure.
        obv = (direction * volume).cumsum()

        return cast(pd.Series, obv)

    @staticmethod
    def calculate_convexity_5d_fast(slope_series: pd.Series) -> pd.Series:
        return slope_series.diff(2).fillna(0)

    # --- OVERLOAD 1: If input is a Series, output is a Series ---
    @overload
    @staticmethod
    def zscore(data: pd.Series) -> pd.Series: ...  # <--- Literally three dots

    # --- OVERLOAD 2: If input is a DataFrame, output is a DataFrame ---
    @overload
    @staticmethod
    def zscore(data: pd.DataFrame) -> pd.DataFrame: ...  # <--- Literally three dots

    # --- THE ACTUAL IMPLEMENTATION ---
    # This is the only one with real code.
    # Note: No @overload decorator here.

    @staticmethod
    def zscore(data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        if data.empty:
            return data

        # CASE 1: Data is a DataFrame
        if isinstance(data, pd.DataFrame):
            m = data.mean()  # Result: pd.Series
            s = data.std()  # Result: pd.Series

            # FIX 1: Use Pandas .where() instead of np.where() to keep it a Series.
            # FIX 2: Use s.notna() to fix the strikethrough.
            denom = s.where((s != 0) & s.notna(), 1.0)

            res = (data - m) / denom
            return cast(pd.DataFrame, res)

        # CASE 2: Data is a Series
        elif isinstance(data, pd.Series):
            m = float(data.mean())  # Result: float
            s = float(data.std())  # Result: float

            # Simple scalar logic (no NumPy arrays needed)
            denom = s if (s != 0 and not np.isnan(s)) else 1.0

            res = (data - m) / denom
            return cast(pd.Series, res)

        else:
            raise TypeError("Input must be a pandas Series or DataFrame.")


class TickerEngine:
    @staticmethod
    def map_kernels(data, kernel_func, *args, **kwargs):
        return data.groupby(level="Ticker", group_keys=False).apply(
            lambda x: kernel_func(x, *args, **kwargs)
        )
