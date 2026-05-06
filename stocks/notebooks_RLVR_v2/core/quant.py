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
        if data.empty:
            return 0.0

        if isinstance(data, pd.DataFrame):
            return data.apply(lambda col: QuantUtils.calculate_gain(col, min_points))

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
        periods: int = None,
    ) -> Union[float, pd.Series]:
        if periods is None:
            periods = GLOBAL_SETTINGS["annual_period"]
        mu, std = data.mean(), data.std()
        res = (mu / np.maximum(std, 1e-8)) * np.sqrt(periods)

        if isinstance(res, (pd.Series, pd.DataFrame)):
            return res.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return float(res) if np.isfinite(res) else 0.0

    @staticmethod
    def calculate_sharpe_vol(
        returns: Union[pd.Series, pd.DataFrame],
        vol_data: Union[pd.Series, pd.DataFrame],
    ) -> Union[float, pd.Series]:
        mask = returns.notna()
        avg_ret = returns.mean()

        if isinstance(returns, pd.DataFrame) and isinstance(vol_data, pd.Series):
            avg_vol = vol_data
        else:
            avg_vol = vol_data.where(mask).mean()

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
        prev_close = close.shift(1)
        tr = np.maximum(
            high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs())
        )
        return tr

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
        direction = np.sign(close.diff().fillna(0))
        return (direction * volume).cumsum()

    @staticmethod
    def calculate_convexity_5d_fast(slope_series: pd.Series) -> pd.Series:
        return slope_series.diff(2).fillna(0)

    @staticmethod
    def zscore(data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        if data.empty:
            return data

        m = data.mean()
        s = data.std()
        denom = np.where((s != 0) & (~pd.isna(s)), s, 1.0)
        return (data - m) / denom


class TickerEngine:
    @staticmethod
    def map_kernels(data, kernel_func, *args, **kwargs):
        return data.groupby(level="Ticker", group_keys=False).apply(
            lambda x: kernel_func(x, *args, **kwargs)
        )
