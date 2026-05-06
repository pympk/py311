import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from collections import Counter

from core.config import GLOBAL_SETTINGS

@dataclass(frozen=True)
class TaskResult:
    ok: bool
    msg: str = ""
    val: Any = None

@dataclass(frozen=True)
class MarketObservation:
    # Dataframes
    lookback_close: pd.DataFrame
    lookback_returns: pd.DataFrame

    # Series (per ticker)
    atrp: pd.Series
    trp: pd.Series

    atr: pd.Series
    rsi: pd.Series
    consistency: pd.Series
    mom_21: pd.Series
    ir_63: pd.Series
    beta_63: pd.Series
    dd_21: pd.Series

    ############
    # --- 4 NEW MICROSTRUCTURE SLOTS ---
    autocorr_15: pd.Series
    range_pos_20: pd.Series
    slope_p_5: pd.Series
    slope_v_5: pd.Series
    convexity: pd.Series
    ############

    # Macro Scalars (or Series)
    macro_trend: float
    macro_trend_vel: float
    macro_vix_z: float
    macro_vix_ratio: float

@dataclass
class FilterPack:
    """The 'Saved List' and state for the second filter pass."""
    decision_date: Optional[pd.Timestamp] = None
    eligible_pool: List[str] = field(default_factory=list)  # Survivors of Stage 1
    selected_tickers: List[str] = field(default_factory=list)  # Final output

    def __repr__(self):
        return f"FilterPack(Date: {self.decision_date}, Eligible: {len(self.eligible_pool)}, Selected: {len(self.selected_tickers)})"

@dataclass
class EngineInput:
    mode: str
    decision_date: pd.Timestamp
    lookback_period: int
    holding_period: int
    metric: str
    benchmark_ticker: str
    rank_start: int = 1
    rank_end: int = 10
    # Default factory pulls from Global thresholds
    quality_thresholds: Dict[str, float] = field(
        default_factory=lambda: GLOBAL_SETTINGS["thresholds"].copy()
    )
    manual_tickers: List[str] = field(default_factory=list)
    debug: bool = False
    universe_subset: Optional[List[str]] = None

@dataclass
class EngineOutput:
    # 1. CORE DATA (Required - No Defaults)
    portfolio_series: pd.Series
    benchmark_series: pd.Series
    normalized_plot_data: pd.DataFrame
    tickers: List[str]
    initial_weights: pd.Series
    perf_metrics: Dict[str, float]
    results_df: pd.DataFrame

    # 2. TIMELINE (Required - No Defaults)
    start_date: pd.Timestamp
    decision_date: pd.Timestamp
    buy_date: pd.Timestamp
    holding_end_date: pd.Timestamp

    # 3. OPTIONAL / AUDIT DATA (Must be at the bottom because they have defaults)
    portfolio_atrp_series: Optional[pd.Series] = None
    benchmark_atrp_series: Optional[pd.Series] = None
    portfolio_trp_series: Optional[pd.Series] = None
    benchmark_trp_series: Optional[pd.Series] = None
    error_msg: Optional[str] = None
    debug_data: Optional[Dict[str, Any]] = None
    macro_df: Optional[pd.DataFrame] = None

    # 4. The Standardized Alpha Matrix
    alpha_perception: pd.DataFrame = None

@dataclass(frozen=True)
class SelectionResult:
    tickers: List[str]
    table: pd.DataFrame
    debug: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DiscoveryResult:
    action_weights: Dict[str, float]
    selected_tickers: List[str]
    veritable_reward: float
    metric_values: pd.Series
    raw_alpha_matrix: pd.DataFrame

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

        if isinstance(data, pd.DataFrame):
            column_gains = data.apply(
                lambda col: QuantUtils.calculate_gain(col, min_points)
            )
            return column_gains

        clean = data.dropna()
        if len(clean) < min_points:
            return 0.0

        first_val = clean.iloc[0]
        last_val = clean.iloc[-1]

        if first_val <= 0 or last_val <= 0:
            return -10.0

        gain_val = float(np.log(clean.iloc[-1] / clean.iloc[0]))
        return gain_val

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
        cov = rets.rolling(window).cov(benchmark_rets)
        var = benchmark_rets.rolling(window).var()

        if isinstance(rets, pd.DataFrame):
            return cov.div(var, axis=0).fillna(1.0)
        return (cov / var).fillna(1.0)

    @staticmethod
    def calculate_rolling_ir(
        rets: pd.Series, benchmark_rets: pd.Series, window: int
    ) -> pd.Series:
        active_ret = rets - benchmark_rets
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

@dataclass(frozen=True)
class MetricBlueprint:
    name: str
    category: str
    regime: str
    description: str
    agent_hint: str
    intervention_trigger: str
    formula: Callable[[Any], pd.Series]
    scaling_type: str = "None"

    def __call__(self, obs) -> pd.Series:
        try:
            return self.formula(obs)
        except Exception:
            target_index = (
                obs.rsi.index if hasattr(obs, "rsi") else obs.lookback_close.columns
            )
            return pd.Series([float("nan")] * len(target_index), index=target_index)

    def get_agent_view(self, obs) -> pd.Series:
        raw = self.__call__(obs)
        clean_raw = raw.replace([np.inf, -np.inf], np.nan)

        if self.scaling_type == "Z-Score":
            scaled = QuantUtils.zscore(clean_raw)
        elif self.scaling_type == "Center":
            scaled = (clean_raw - 0.5) * 2
        elif self.scaling_type == "RSI":
            scaled = (clean_raw + 50) / 20
        else:
            scaled = clean_raw

        clip_val = GLOBAL_SETTINGS.get("feature_zscore_clip", 4.0)
        return scaled.fillna(0).clip(-clip_val, clip_val)

class TickerEngine:
    """
    The Orchestrator: Bridges MultiIndex DataFrames with QuantUtils Kernels.
    """
    @staticmethod
    def map_kernels(data, kernel_func, *args, **kwargs):
        """
        The Orchestrator: Bridges MultiIndex DataFrames with QuantUtils Kernels.
        Ensures consistent behavior even when only one Ticker is present.
        """
        # 1. Group by Ticker
        grouped = data.groupby(level="Ticker", group_keys=False)
        
        # 2. Use apply
        res = grouped.apply(lambda x: kernel_func(x, *args, **kwargs))

        # 3. ROBUSTNESS: Handle pandas 'apply' inconsistency for single-group cases.
        # If there's only one ticker, 'apply' sometimes returns the result transposed or as a row.
        # We ensure the result matches the input index length and structure.
        if hasattr(data, "index") and isinstance(data.index, pd.MultiIndex):
            unique_tickers = data.index.get_level_values("Ticker").unique()
            if len(unique_tickers) == 1:
                # If we got a DataFrame with 1 row but input had multiple rows, it's transposed
                if isinstance(res, pd.DataFrame) and len(res) == 1 and len(data) > 1:
                    # Detect if it's a single-row DF where columns match expected dates
                    # We just stack it to get a Series and force the index.
                    res = res.stack()
                    res.index = data.index
                # If we got a Series but it lost the Ticker level
                elif isinstance(res, pd.Series) and not isinstance(res.index, pd.MultiIndex):
                    res.index = data.index

        return res

def _prepare_initial_weights(tickers: List[str]) -> pd.Series:
    ticker_counts = Counter(tickers)
    total = len(tickers)
    return pd.Series({t: c / total for t, c in ticker_counts.items()})

def calculate_buy_and_hold_performance(
    df_close_wide: pd.DataFrame,
    df_atrp_wide: pd.DataFrame,
    df_trp_wide: pd.DataFrame,
    tickers: List[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
):
    if not tickers:
        return pd.Series(), pd.Series(), pd.Series()

    initial_weights = _prepare_initial_weights(tickers)
    ticker_list = initial_weights.index.tolist()
    p_slice = df_close_wide.reindex(columns=ticker_list).loc[start_date:end_date]
    a_slice = df_atrp_wide.reindex(columns=ticker_list).loc[start_date:end_date]
    t_slice = df_trp_wide.reindex(columns=ticker_list).loc[start_date:end_date]

    return QuantUtils.compute_portfolio_stats(
        p_slice, a_slice, t_slice, initial_weights
    )

class PerformanceCalculator:
    @staticmethod
    def calculate_period_metrics(
        full_data: tuple, hold_data: tuple, decision_date: pd.Timestamp, prefix: str
    ):
        f_val, f_ret, f_atrp, f_trp = full_data
        h_val, h_ret, h_atrp, h_trp = hold_data

        lb_val = f_val.loc[:decision_date]
        lb_ret = f_ret.loc[:decision_date]
        lb_atrp = f_atrp.loc[:decision_date]
        lb_trp = f_trp.loc[:decision_date]

        m = {
            f"full_{prefix}_gain": QuantUtils.calculate_gain(f_val),
            f"full_{prefix}_sharpe": QuantUtils.calculate_sharpe(f_ret),
            f"full_{prefix}_sharpe_atrp": QuantUtils.calculate_sharpe_vol(
                f_ret, f_atrp
            ),
            f"full_{prefix}_sharpe_trp": QuantUtils.calculate_sharpe_vol(f_ret, f_trp),
            f"lookback_{prefix}_gain": QuantUtils.calculate_gain(lb_val),
            f"lookback_{prefix}_sharpe": QuantUtils.calculate_sharpe(lb_ret),
            f"lookback_{prefix}_sharpe_atrp": QuantUtils.calculate_sharpe_vol(
                lb_ret, lb_atrp
            ),
            f"lookback_{prefix}_sharpe_trp": QuantUtils.calculate_sharpe_vol(
                lb_ret, lb_trp
            ),
            f"holding_{prefix}_gain": QuantUtils.calculate_gain(h_val),
            f"holding_{prefix}_sharpe": QuantUtils.calculate_sharpe(h_ret),
            f"holding_{prefix}_sharpe_atrp": QuantUtils.calculate_sharpe_vol(
                h_ret, h_atrp
            ),
            f"holding_{prefix}_sharpe_trp": QuantUtils.calculate_sharpe_vol(
                h_ret, h_trp
            ),
        }
        return m, {"val": f_val, "ret": f_ret, "atrp": f_atrp, "trp": f_trp}

class HeadlessReporter:
    @staticmethod
    def get_metadata(res: EngineOutput) -> Dict[str, Any]:
        return {
            "start": res.start_date.date(),
            "decision": res.decision_date.date(),
            "entry": res.buy_date.date(),
            "end": res.holding_end_date.date(),
            "tickers": res.tickers,
            "ticker_count": len(res.tickers),
        }

    @staticmethod
    def get_metrics_table(res: EngineOutput) -> pd.DataFrame:
        m = res.perf_metrics
        rows = []
        metric_types = [
            ("Gain", "gain"),
            ("Sharpe", "sharpe"),
            ("Sharpe (ATRP)", "sharpe_atrp"),
            ("Sharpe (TRP)", "sharpe_trp"),
        ]

        for label, key in metric_types:
            p_row = {
                "Metric": f"Group {label}",
                "Full": m.get(f"full_p_{key}"),
                "Lookback": m.get(f"lookback_p_{key}"),
                "Holding": m.get(f"holding_p_{key}"),
            }
            b_row = {
                "Metric": f"Benchmark {label}",
                "Full": m.get(f"full_b_{key}"),
                "Lookback": m.get(f"lookback_b_{key}"),
                "Holding": m.get(f"holding_b_{key}"),
            }

            d_row = {"Metric": f"== {label} Delta"}
            for col in ["Full", "Lookback", "Holding"]:
                p_val, b_val = p_row[col] or 0.0, b_row[col] or 0.0
                d_row[col] = p_val - b_val
            rows.extend([p_row, b_row, d_row])

        return pd.DataFrame(rows).set_index("Metric")
