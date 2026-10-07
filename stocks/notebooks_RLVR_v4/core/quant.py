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

    @overload
    @staticmethod
    def calculate_sharpe(data: pd.Series, periods: Optional[int] = None) -> float: ...

    @overload
    @staticmethod
    def calculate_sharpe(
        data: pd.DataFrame, periods: Optional[int] = None
    ) -> pd.Series: ...

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

        if isinstance(data, pd.DataFrame):
            mu = data.mean()
            std = data.std()
            res = (mu / np.maximum(std, 1e-8)) * np.sqrt(periods)
            cleaned = res.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            return cast(pd.Series, cleaned)

        elif isinstance(data, pd.Series):
            mu = float(data.mean())
            std = float(data.std())
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
        # --- FIX: Suppress the RuntimeWarning for columns that are entirely NaN ---
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
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
        valid_prices = prices.dropna(how="all", axis=1)
        if valid_prices.empty:
            empty_s = pd.Series(0.0, index=prices.index)
            return empty_s, empty_s, empty_s, empty_s

        base_prices = valid_prices.bfill().iloc[0]
        valid_cols = base_prices.dropna().index
        if len(valid_cols) == 0:
            empty_s = pd.Series(0.0, index=prices.index)
            return empty_s, empty_s, empty_s, empty_s

        valid_prices = valid_prices[valid_cols]
        base_prices = base_prices[valid_cols]

        valid_weights = weights.reindex(valid_cols)
        valid_weights = valid_weights / valid_weights.sum()

        norm_prices = valid_prices.div(base_prices)
        weighted_components = norm_prices.mul(valid_weights, axis=1)
        equity_curve = weighted_components.sum(axis=1)

        returns_WITH_BOUNDARY_NAN = QuantUtils.compute_returns(equity_curve)
        current_weights = weighted_components.div(equity_curve, axis=0)

        sub_atrp = atrp_matrix.reindex(columns=valid_cols)
        sub_trp = trp_matrix.reindex(columns=valid_cols)

        portfolio_atrp = (current_weights * sub_atrp).sum(axis=1, min_count=1)
        portfolio_trp = (current_weights * sub_trp).sum(axis=1, min_count=1)

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
    def calculate_information_ratio(
        rets: pd.Series,
        benchmark_rets: pd.Series,
        periods: Optional[int] = None,
    ) -> float:
        """
        [REPORT KERNEL] Calculates full-period annualized Information Ratio (IR).
        IR = (Mean(Active Return) / Std(Active Return)) * sqrt(periods)
        """
        if periods is None:
            periods = 252

        dates = (
            rets.index.get_level_values("Date")
            if isinstance(rets.index, pd.MultiIndex)
            else rets.index
        )
        aligned_bench = pd.Series(
            benchmark_rets.reindex(dates).values, index=rets.index
        )

        active_ret = (rets - aligned_bench).dropna()
        if len(active_ret) < 2:
            return 0.0

        mu = float(active_ret.mean())
        std = float(active_ret.std())
        if std <= 1e-8 or not np.isfinite(std):
            return 0.0

        ir = (mu / std) * np.sqrt(periods)
        return float(ir) if np.isfinite(ir) else 0.0

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
    def calculate_momentum(series: pd.Series, window: int) -> pd.Series:
        """Vectorized cumulative return over rolling window: (P_t - P_{t-w}) / P_{t-w}."""
        return series.pct_change(window, fill_method=None).fillna(0.0)

    @staticmethod
    def calculate_rolling_ivol(
        rets: Union[pd.Series, pd.DataFrame], benchmark_rets: pd.Series, window: int
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Idiosyncratic / Residual Volatility: Rolling Std of CAPM residuals.
        e_{i, t} = r_{i, t} - beta_{i, t} * r_{bm, t}
        """
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
        var_safe = var.replace(0, np.nan)

        if isinstance(rets, pd.DataFrame):
            beta = cov.div(var_safe, axis=0).fillna(1.0)
            residuals = rets - beta.mul(aligned_bench, axis=0)
        else:
            beta = (cov / var_safe).fillna(1.0)
            residuals = rets - (beta * aligned_bench)

        return residuals.rolling(window).std().fillna(0.0)

    @overload
    @staticmethod
    def zscore(data: pd.Series) -> pd.Series: ...

    @overload
    @staticmethod
    def zscore(data: pd.DataFrame) -> pd.DataFrame: ...

    @staticmethod
    def zscore(data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        if data.empty:
            return data

        if isinstance(data, pd.DataFrame):
            m = data.mean()
            s = data.std()
            denom = s.where((s != 0) & s.notna(), 1.0)
            res = (data - m) / denom
            return cast(pd.DataFrame, res)

        elif isinstance(data, pd.Series):
            m = float(data.mean())
            s = float(data.std())
            denom = s if (s != 0 and not np.isnan(s)) else 1.0
            res = (data - m) / denom
            return cast(pd.Series, res)

        else:
            raise TypeError("Input must be a pandas Series or DataFrame.")

    @staticmethod
    def build_forward_return_matrix(
        df_close: pd.DataFrame, horizon: int = 1
    ) -> pd.DataFrame:
        """Computes forward simple returns: R_{t -> t+h} = (P_{t+h} - P_t) / P_t

        Tagged with strict temporal metadata to prevent accidental ingestion of
        backward returns.
        """
        fwd_ret = df_close.pct_change(horizon, fill_method=None).shift(-horizon)
        if "CASH" not in fwd_ret.columns:
            fwd_ret["CASH"] = 0.0
        fwd_ret.attrs["temporal_alignment"] = f"forward_{horizon}d"
        fwd_ret.attrs["is_forward_looking"] = True
        return fwd_ret

    # -------------------------------------------------------------------------
    # PURE NUMPY MTM & ALPHA KERNELS (STATELESS)
    # -------------------------------------------------------------------------
    @staticmethod
    def calculate_equal_weight_return(ret_array: np.ndarray) -> float:
        """Pure C-level NumPy mean return for a 1D array of asset returns."""
        if ret_array.size == 0:
            return 0.0
        finite_mask = np.isfinite(ret_array)
        if not np.any(finite_mask):
            return 0.0
        return float(np.mean(ret_array[finite_mask]))

    @staticmethod
    def calculate_tri_asset_portfolio_return(
        gross_stock_ret: float,
        bm_ret: float,
        cash_ret: float,
        w_active: float,
        w_benchmark: float,
        w_cash: float,
        slippage_loss: float = 0.0,
    ) -> Tuple[float, float]:
        """Calculates (gross_daily_ret, net_daily_ret) using linear dot product."""
        gross = w_active * gross_stock_ret + w_benchmark * bm_ret + w_cash * cash_ret
        net = gross - slippage_loss
        return float(gross), float(net)

    @staticmethod
    def calculate_shaped_alpha_reward(
        alpha_ret: float,
        upside_mult: float = 1.0,
        loss_penalty: float = 0.0,
    ) -> float:
        """Pure mathematical reward shaping for RL policy optimization.

        NEVER to be compounded as an equity curve.
        """
        if not np.isfinite(alpha_ret):
            return 0.0
        if alpha_ret > 0.0:
            return float(alpha_ret * max(1.0, upside_mult))
        elif alpha_ret < 0.0:
            return float(alpha_ret * (1.0 + max(0.0, loss_penalty)))
        return 0.0

    @staticmethod
    def step_compounding_curves(
        prev_portfolio_equity: float,
        prev_benchmark_equity: float,
        net_daily_ret: float,
        bm_daily_ret: float,
    ) -> Tuple[float, float, float]:
        """Calculates next step (portfolio_equity, benchmark_equity, alpha_multiplier).

        Alpha Multiplier is strictly V_p(t) / V_bm(t) (Base 1.0).
        """
        new_p_equity = prev_portfolio_equity * (1.0 + net_daily_ret)
        new_bm_equity = prev_benchmark_equity * (1.0 + bm_daily_ret)
        alpha_multiplier = new_p_equity / max(new_bm_equity, 1e-8)
        return (
            float(new_p_equity),
            float(new_bm_equity),
            float(alpha_multiplier),
        )

    @staticmethod
    def calculate_momentum_skip(
        series: pd.Series, window_total: int = 252, window_skip: int = 21
    ) -> pd.Series:
        """
        Fama-French 12-1 Momentum: Cumulative return from t-252 to t-21,
        excluding the most recent 21 sessions to eliminate short-term reversal drag.
        """
        p_skip = series.shift(window_skip)
        p_base = series.shift(window_total)
        mom = (p_skip / p_base.replace(0, np.nan)) - 1.0
        return mom.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def calculate_downside_semideviation(
        rets: Union[pd.Series, pd.DataFrame], window: int = 63
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Downside Semi-Deviation: sqrt( E[ min(r, 0)^2 ] ).
        Isolates left-tail downside volatility without penalizing upside windfalls.
        """
        downside = rets.clip(upper=0.0)
        downside_sq = downside**2
        roll_mean_sq = downside_sq.rolling(
            window=window, min_periods=max(1, window // 2)
        ).mean()
        # Native pandas .pow(0.5) preserves Series/DataFrame type for Pylance
        return roll_mean_sq.pow(0.5).fillna(0.0)

    @staticmethod
    def calculate_trend_quality(series: pd.Series, window: int = 63) -> pd.Series:
        """
        Trend Quality / Smoothness: Pearson correlation between log(Price) and time [0, 1, ..., W-1].
        Bounded in [-1.0, 1.0]. +1 = pure monotonic exponential trend, 0 = random walk, -1 = smooth crash.
        """
        clean_p = series.replace(0, np.nan).ffill()
        # Explicit pd.Series construction guarantees .rolling() is recognized by Pylance
        log_p = pd.Series(
            np.log(np.where(clean_p > 0, clean_p, np.nan)), index=series.index
        )
        time_idx = pd.Series(np.arange(len(series), dtype=float), index=series.index)
        trend_r = (
            log_p.rolling(window=window, min_periods=max(2, window // 2))
            .corr(time_idx)
            .clip(-1.0, 1.0)
            .fillna(0.0)
        )
        return trend_r

    @staticmethod
    def compute_composite_fitness(
        excess_return: float,
        information_ratio: float,
        ir_floor: float = 0.05,
    ) -> float:
        """
        [CHECKPOINT KERNEL] Principled Benchmark-Relative Validation Fitness.
        Fitness = Excess_Return * max(ir_floor, Information_Ratio) if Excess >= 0
        Fitness = Excess_Return * (1.0 / ir_floor)                 if Excess < 0

        Properties:
        1. Invariant to absolute benchmark beta (eliminates SPY-sheltering bias).
        2. Penalizes micro-alpha tracking-error gaming (requires meaningful excess return).
        3. Strictly negative for underperforming strategies (excess < 0).
        """
        if not (np.isfinite(excess_return) and np.isfinite(information_ratio)):
            return -10.0

        if excess_return >= 0.0:
            # Rewards simultaneous scale of excess wealth and consistency of active spread
            return float(excess_return * max(ir_floor, information_ratio))
        else:
            # Steep linear penalty for lagging the benchmark
            return float(excess_return * (1.0 / ir_floor))


class TickerEngine:
    @staticmethod
    def map_kernels(data, kernel_func, *args, **kwargs):
        return data.groupby(level="Ticker", group_keys=False).apply(
            lambda x: kernel_func(x, *args, **kwargs)
        )
