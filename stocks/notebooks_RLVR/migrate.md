Yes, these 5 principles are absolutely correct and represent the gold standard for maintaining a codebase in the AI-coding era. AI coding assistants thrive on context; if your modules are deep, your language ubiquitous, and your tests clear, the AI generates accurate, bug-free code. If the code is tangled, the AI will hallucinate and generate spaghetti.

Here is a concise evaluation of your current codebase against these principles, followed by a concrete migration plan.

### 1. Codebase Evaluation

**1. Ubiquitous Language (Pass with minor issues)**
*   *Good:* You use excellent, descriptive domain terms: `FeatureCube`, `MarketObservation`, `AlphaEngine`, `MetricBlueprint`. This gives the AI fantastic context.
*   *Bad:* Some generic variable names still exist (`df_ohlcv`, `res`, `m`, `obs`).

**2. Deep Modules (Mixed)**
*   *Good:* `QuantUtils` and `TickerEngine` are perfect deep modules. Simple interfaces (`calculate_sharpe`), hiding complex vectorized math.
*   *Bad:* `AlphaEngine` and `generate_features` are "God Modules". They are shallow and broad. `AlphaEngine.run()` handles data validation, timeline calculation, strategy routing, and performance metrics all at once.

**3. Test-Driven Development (Fail)**
*   *Good:* You have `auditor.py`, which shows a strong commitment to mathematical integrity.
*   *Bad:* These are runtime tripwires, not TDD. They rely on `print()` statements and manual execution in notebooks. AI tools operate best with standard `pytest` suites acting as specifications.

**4. Vertical Slices (Fail)**
*   *Bad:* The architecture is horizontally layered. If you want to add a new execution mode, you currently have to touch `contracts.py` -> `engine.py` -> `performance.py` -> `analyzer.py`. 

**5. Iterative Refinement (Ongoing)**
*   *Good:* Your comments (e.g., "MANDATORY FIX", "Refactored for DRY") show you are already doing this. The codebase is ripe for the next major refactoring loop.

---

### 2. Migration Plan

We will migrate iteratively so you don't break the current working system.

#### Phase 1: Establish True TDD (The Safety Net)
Before modifying the engine, we must convert `auditor.py` into standard unit tests. AI uses `pytest` outputs to fix its own code.
*   **Action:** Create a `tests/` directory.
*   **Action:** Migrate `verify_math_integrity`, `verify_ranking_integrity`, and `verify_feature_engineering_integrity` into `tests/test_quant.py` and `tests/test_features.py`.
*   **Action:** Replace `TaskResult(ok=False)` with standard `assert` statements.

#### Phase 2: Deepen the Modules (Kill the God Classes)
`AlphaEngine` and `generate_features` bloat the AI's context window. We must break them down.
*   **Action:** Extract `_validate_timeline`, `_filter_universe`, and `_build_observation` out of `AlphaEngine` into a dedicated `UniverseScreener` class. `AlphaEngine` should only take an observation and run it.
*   **Action:** Break `generate_features` (currently ~180 lines) into smaller sub-pipelines: `MacroFeaturePipeline`, `MicroFeaturePipeline`, `QualityFilterPipeline`.

#### Phase 3: Vertical Slices (Feature-based grouping)
Instead of grouping files by technical layer (`core/analyzer.py`, `core/engine.py`), group them by the *domain feature*.
*   **Action:** Restructure the directories. For example:
    *   `features/walk_forward/` (Contains its own UI, engine hook, and performance metrics).
    *   `features/rl_discovery/` (Contains the RL environment, discovery logic).
    *   `features/data_pipeline/` (Contains feature cube builders and cache logic).
*   *Why?* When you ask the AI to "Update the Walk Forward logic", it only needs to read the `walk_forward` folder, saving context tokens and preventing side-effects in the RL environment.

#### Phase 4: Enforce Ubiquitous Language via Strict Typing
*   **Action:** Eliminate `GLOBAL_SETTINGS` dict. AI struggles with global dictionaries because the keys are invisible to static analysis.
*   **Action:** Convert `GLOBAL_SETTINGS` into a strongly typed Dataclass or Pydantic `BaseModel` (e.g., `TradingConfig`). Pass this config explicitly into your classes.

### Next Steps
Which phase would you like to tackle first? I recommend starting with **Phase 1 (TDD)** so we have a safety net, or **Phase 4 (Config/Typing)** as it's the easiest quick-win to clear up context. Let me know what you want to execute.

---  
***

Let's execute Phase 1. By moving these runtime tripwires into a standard `pytest` suite, the AI (and you) will have a rapid, automated way to verify that core math doesn't break during future refactoring.

Create the following directory structure in your new `notebooks_RLVR_v2` folder:

```text
notebooks_RLVR_v2/
├── core/             # (Copy your existing core/ files here for now)
├── strategy/         # (Copy your existing strategy/ files here for now)
└── tests/            # (NEW)
    ├── __init__.py   # (Empty file)
    ├── test_quant.py
    └── test_features.py
```

Here is the exact code to convert your `auditor.py` logic into standard `pytest` specs.

### File: `notebooks_RLVR_v2/tests/test_quant.py`
This covers your math boundaries, ranking integrity, and temporal coupling checks.

```python
import pytest
import pandas as pd
import numpy as np

# Adjust imports based on your structure
from core.quant import QuantUtils

def test_compute_returns_boundary_integrity():
    """Validates math kernels before execution (Leading NaNs)."""
    # Test 1: Series Boundary
    mock_s = pd.Series([100.0, 102.0, 101.0])
    rets_s = QuantUtils.compute_returns(mock_s)
    assert pd.isna(rets_s.iloc[0]), "Math Integrity: Series Leading NaN missing"

    # Test 2: DataFrame Boundary
    mock_df = pd.DataFrame({"A": [100, 101], "B": [200, 202]})
    rets_df = QuantUtils.compute_returns(mock_df)
    assert rets_df.iloc[0].isna().all(), "Math Integrity: DF Leading NaN missing"

def test_ranking_integrity_sharpe_vol():
    """
    Prevents 'Momentum Collapse' in Volatility-Adjusted Ranking.
    Ensures Sharpe(Vol) distinguishes between High-Vol and Low-Vol stocks.
    """
    # VOLATILE: 10% ret / 10% Vol = 1.0 Sharpe
    # STABLE:   2% ret / 1% Vol   = 2.0 Sharpe (Winner)
    data = {"VOLATILE": [1.0, 1.10], "STABLE": [1.0, 1.02]}
    df_returns = pd.DataFrame(data).pct_change().dropna()
    vol_series = pd.Series({"VOLATILE": 0.10, "STABLE": 0.01})

    results = QuantUtils.calculate_sharpe_vol(df_returns, vol_series)

    assert not np.isclose(results["VOLATILE"], results["STABLE"]), "RANKING COLLAPSE: No differentiation"
    assert results["STABLE"] > results["VOLATILE"], "MOMENTUM REGRESSION: Volatility ignored"
    assert np.isclose(results["STABLE"], 2.0), f"MATH ERROR: Expected 2.0, got {results['STABLE']}"

def test_volatility_alignment_temporal_coupling():
    """
    Verifies Temporal Coupling between Returns and Volatility.
    Ensures denominator only counts days where a valid return exists.
    """
    # Day 1: NaN Return, 0.90 Vol
    # Day 2: 0.10 Return, 0.10 Vol
    rets_s = pd.Series([np.nan, 0.10])
    vol_s = pd.Series([0.90, 0.10])

    res_series = QuantUtils.calculate_sharpe_vol(rets_s, vol_s)
    assert np.isclose(res_series, 1.0), f"DENOMINATOR MISMATCH: Series {res_series:.2f} != 1.0"

    rets_df = pd.DataFrame({"A": [np.nan, 0.10], "B": [np.nan, 0.20]})
    vol_df = pd.DataFrame({"A": [0.90, 0.10], "B": [0.05, 0.20]})
    
    res_df = QuantUtils.calculate_sharpe_vol(rets_df, vol_df)
    assert np.isclose(res_df["A"], 1.0) and np.isclose(res_df["B"], 1.0), "VECTORIZED MISMATCH: Column alignment failed"
```

### File: `notebooks_RLVR_v2/tests/test_features.py`
This converts your feature engineering Wilder's smoothing and initialization checks.

```python
import pytest
import pandas as pd
import numpy as np

from core.features import generate_features

def test_feature_engineering_wilders_atr():
    """
    Validates Feature Engineering Logic.
    Enforces: Day 1 ATR must be NaN, Initialization, and Wilder's Smoothing.
    """
    # 1. Create Synthetic Data (3 Days)
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    idx = pd.MultiIndex.from_product([["TEST"], dates], names=["Ticker", "Date"])

    df_mock = pd.DataFrame({
        "Adj Open": [100, 110, 110],
        "Adj High": [110, 130, 120],
        "Adj Low": [100, 110, 110],
        "Adj Close": [105, 120, 115],  # PrevClose: NaN, 105, 120
        "Volume": [1000, 1000, 1000],
    }, index=idx)

    # 2. Run the Generator (Period=2 means Alpha = 1/2 = 0.5)
    feats_df, _ = generate_features(
        df_mock, 
        atr_period=2, 
        rsi_period=2, 
        quality_min_periods=1
    )

    atr_series = feats_df["ATR"]

    # 3. Assertions
    # Check Day 1 (No PrevClose)
    assert np.isnan(atr_series.iloc[0]), f"Day 1 Regression: Expected NaN, got {atr_series.iloc[0]}"

    # Check Day 2 (Initialization: H-L=20, |130-105|=25, |110-105|=5 -> TR=25)
    assert np.isclose(atr_series.iloc[1], 25.0), f"Initialization Regression: Expected 25.0, got {atr_series.iloc[1]}"

    # Check Day 3 (Recursion: TR=10. ATR_3 = (10 * 0.5) + (25 * 0.5) = 17.5)
    assert np.isclose(atr_series.iloc[2], 17.5), f"Wilder's Logic Regression: Expected 17.5, got {atr_series.iloc[2]}"
```

### Next Action
Open your terminal, navigate to the `notebooks_RLVR_v2` directory, and run:
`pytest tests/ -v`

If this passes, we have successfully established our TDD safety net for the core math. 

Once confirmed, do you want to move to **Phase 4 (Enforce Ubiquitous Language via Strict Typing for Configs)** to remove the `GLOBAL_SETTINGS` dict, or **Phase 2 (Deepen Modules)** to start breaking apart `AlphaEngine`?
---  
---  
The error `ValueError: Data must be 1-dimensional, got ndarray of shape (1, 3) instead` is a classic Pandas 2.x alignment issue. 

When calculating `ir_63` and `beta_63`, we are performing operations between a MultiIndex Series `(Ticker, Date)` and a Single Index Series `(Date)`. In older versions of Pandas, it broadcasted automatically. In newer versions, misaligned indexes during `.cov()` or subtraction can result in cross-product DataFrames (hence the `(1, 3)` 2D array slipping into the features dictionary).

To permanently fix this, we must align the `benchmark_rets` directly to the `rets` index inside the math kernels. This is the exact same robust alignment you previously used in `auditor.py`.

Here is the patched **`core/quant.py`**. Replace your existing file with this:

### File: `notebooks_RLVR_v2/core/quant.py`

```python
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
        dates = rets.index.get_level_values("Date") if isinstance(rets.index, pd.MultiIndex) else rets.index
        aligned_bench = pd.Series(benchmark_rets.reindex(dates).values, index=rets.index)

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
        dates = rets.index.get_level_values("Date") if isinstance(rets.index, pd.MultiIndex) else rets.index
        aligned_bench = pd.Series(benchmark_rets.reindex(dates).values, index=rets.index)

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
```

Run `poetry run pytest tests/ -v` again. It should now cleanly pass! If it does, we can knock out Phase 4 (Config Typing).
---  
---  
The error occurred because of a known issue in Pandas 2.x when mixing MultiIndex `(Ticker, Date)` Series with Single Index `(Date)` Series during `pd.DataFrame` construction. 

In `features.py`, older operations like `consistency = ...groupby().rolling().mean().reset_index(drop=True)` were dropping the Ticker index. When Pandas 2.x tried to combine that with the other MultiIndex features, it panicked and broadcasted it into a 2D `(1, 3)` numpy array.

The solution is to use our `TickerEngine.map_kernels` for **everything**. It guarantees index alignment and is completely immune to Pandas version changes. It also removes the dangerous `.values` assignment at the bottom of the file.

Here is the fully patched `features.py`. Replace your file with this code and run the test again.

### File: `notebooks_RLVR_v2/core/features.py`

```python
import pandas as pd
import numpy as np

from typing import Tuple
from core.quant import QuantUtils, TickerEngine
from core.settings import GLOBAL_SETTINGS


def generate_features(
    df_ohlcv: pd.DataFrame,
    df_indices: pd.DataFrame = None,
    df_fed: pd.DataFrame = None,
    benchmark_ticker: str = GLOBAL_SETTINGS["benchmark_ticker"],
    atr_period: int = GLOBAL_SETTINGS["atr_period"],
    rsi_period: int = GLOBAL_SETTINGS["rsi_period"],
    win_5d: int = GLOBAL_SETTINGS["5d_window"],
    win_21d: int = GLOBAL_SETTINGS["21d_window"],
    win_63d: int = GLOBAL_SETTINGS["63d_window"],
    feature_zscore_clip: float = GLOBAL_SETTINGS["feature_zscore_clip"],
    quality_window: int = GLOBAL_SETTINGS["quality_window"],
    quality_min_periods: int = GLOBAL_SETTINGS["quality_min_periods"],
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    print(f"⚡ Generating Decoupled Features (Benchmark: {benchmark_ticker})...")

    # --- 0. PREP ---
    df_ohlcv = df_ohlcv.sort_index(level=["Ticker", "Date"])
    all_dates = df_ohlcv.index.get_level_values("Date").unique().sort_values()

    # --- 1. MACRO ENGINE ---
    macro_df = pd.DataFrame(index=all_dates)
    if benchmark_ticker in df_ohlcv.index.get_level_values("Ticker"):
        mkt_close = (
            df_ohlcv.xs(benchmark_ticker, level="Ticker")["Adj Close"]
            .reindex(all_dates)
            .ffill()
        )
        macro_df["Mkt_Ret"] = mkt_close.pct_change().fillna(0.0)
        macro_df["Macro_Trend"] = (mkt_close / mkt_close.rolling(200).mean()) - 1.0
    else:
        macro_df["Mkt_Ret"] = 0.0
        macro_df["Macro_Trend"] = 0.0

    # --- 1.2 FED Data Integration ---
    if df_fed is not None:
        fed_data = df_fed.reindex(all_dates).ffill().bfill()
        macro_df["High_Yield_Spread"] = fed_data["High_Yield_Spread"]
        macro_df["Yield_Curve_10Y2Y"] = fed_data["Yield_Curve_10Y2Y"]

        for col in ["High_Yield_Spread", "Yield_Curve_10Y2Y"]:
            roll_mean = macro_df[col].rolling(252, min_periods=60).mean()
            roll_std = macro_df[col].rolling(252, min_periods=60).std()
            macro_df[f"{col}_Z"] = (
                ((macro_df[col] - roll_mean) / roll_std)
                .clip(-feature_zscore_clip, feature_zscore_clip)
                .fillna(0.0)
            )
    else:
        macro_df["High_Yield_Spread"] = 0.0
        macro_df["Yield_Curve_10Y2Y"] = 0.0
        macro_df["High_Yield_Spread_Z"] = 0.0
        macro_df["Yield_Curve_10Y2Y_Z"] = 0.0

    # --- TREND VELOCITY & MOMENTUM ---
    macro_df["Macro_Trend_Vel"] = macro_df["Macro_Trend"].diff(win_21d)
    macro_df["Macro_Trend_Vel_Z"] = (
        macro_df["Macro_Trend_Vel"] / macro_df["Macro_Trend"].rolling(win_63d).std()
    ).clip(-feature_zscore_clip, feature_zscore_clip)
    macro_df["Macro_Trend_Mom"] = (
        np.sign(macro_df["Macro_Trend"])
        * np.sign(macro_df["Macro_Trend_Vel"])
        * np.abs(macro_df["Macro_Trend_Vel"])
    ).fillna(0)

    # VIX Extraction
    macro_df["Macro_Vix_Z"] = 0.0
    macro_df["Macro_Vix_Ratio"] = 1.0
    if df_indices is not None:
        idx_names = df_indices.index.get_level_values(0).unique()
        if "^VIX" in idx_names:
            v = df_indices.xs("^VIX", level=0)["Adj Close"].reindex(all_dates).ffill()
            macro_df["Macro_Vix_Z"] = (
                (v - v.rolling(63).mean()) / v.rolling(63).std()
            ).clip(-feature_zscore_clip, feature_zscore_clip)
        if "^VIX" in idx_names and "^VIX3M" in idx_names:
            v3 = (
                df_indices.xs("^VIX3M", level=0)["Adj Close"].reindex(all_dates).ffill()
            )
            macro_df["Macro_Vix_Ratio"] = (v / v3).fillna(1.0)
    macro_df.fillna(0.0, inplace=True)

    # --- 2. TICKER ENGINE ---
    # STEP 1: Returns via TickerEngine Orchestrator
    rets = TickerEngine.map_kernels(df_ohlcv["Adj Close"], QuantUtils.compute_returns)
    autocorr_15 = TickerEngine.map_kernels(
        rets, QuantUtils.calculate_autocorr, lag=1, window=15
    )
    
    mkt_ret_series = macro_df["Mkt_Ret"]

    # A. Hybrid Metrics (Beta & IR)
    ir_63 = TickerEngine.map_kernels(
        rets,
        QuantUtils.calculate_rolling_ir,
        benchmark_rets=mkt_ret_series,
        window=win_63d,
    )

    beta_63 = TickerEngine.map_kernels(
        rets,
        QuantUtils.calculate_rolling_beta,
        benchmark_rets=mkt_ret_series,
        window=win_63d,
    )

    # B. Volatility (ATR / TRP)
    def get_ticker_vol(df_slice):
        h, l, c = df_slice["Adj High"], df_slice["Adj Low"], df_slice["Adj Close"]
        return pd.DataFrame(
            {
                "TR_Raw": QuantUtils.calculate_tr(h, l, c),
                "ATR_Smooth": QuantUtils.calculate_atr(h, l, c, atr_period),
            },
            index=df_slice.index,
        )

    vol_bundle = TickerEngine.map_kernels(df_ohlcv, get_ticker_vol)
    atr = vol_bundle["ATR_Smooth"]
    natr = (atr / df_ohlcv["Adj Close"]).fillna(0)
    trp = (vol_bundle["TR_Raw"] / df_ohlcv["Adj Close"]).fillna(0)

    # C. Momentum & Consistency (Fixed to use TickerEngine to guarantee MultiIndex)
    mom_21 = TickerEngine.map_kernels(
        df_ohlcv["Adj Close"], lambda x: x.pct_change(win_21d)
    )
    
    consistency = TickerEngine.map_kernels(
        rets, lambda x: (x > 0).astype(float).rolling(win_5d).mean()
    )
    
    dd_21 = TickerEngine.map_kernels(
        df_ohlcv["Adj Close"], lambda x: (x / x.rolling(win_21d).max()) - 1.0
    )

    # STEP 2: RSI
    rsi = TickerEngine.map_kernels(
        df_ohlcv["Adj Close"], QuantUtils.calculate_rsi, period=rsi_period
    )

    def get_range_pos_kernel(df_slice):
        return QuantUtils.calculate_range_pos(
            df_slice["Adj High"],
            df_slice["Adj Low"],
            df_slice["Adj Close"],
            window=GLOBAL_SETTINGS.get("range_pos_period", 20),
        )

    range_pos_20 = TickerEngine.map_kernels(df_ohlcv, get_range_pos_kernel)

    def get_obv_kernel(df_slice):
        v = df_slice["Volume"]
        v_baseline = v.rolling(window=win_63d, min_periods=1).mean().replace(0, 1e-8)
        v_rel = v / v_baseline
        return QuantUtils.calculate_obv_fast(df_slice["Adj Close"], v_rel)

    obv = TickerEngine.map_kernels(df_ohlcv, get_obv_kernel)

    log_price = np.log(df_ohlcv["Adj Close"].replace(0, 1e-8))
    slope_p = TickerEngine.map_kernels(log_price, QuantUtils.calculate_rolling_slope_5d_fast)
    slope_v = TickerEngine.map_kernels(obv, QuantUtils.calculate_rolling_slope_5d_fast)

    convexity = TickerEngine.map_kernels(slope_p, QuantUtils.calculate_convexity_5d_fast)

    # E. Assemble Features
    features_df = pd.DataFrame(
        {
            "ATR": atr,
            "ATRP": natr,
            "TRP": trp,
            "RSI": rsi,
            "Mom_21": mom_21,
            "Consistency": consistency,
            "IR_63": ir_63,
            "Beta_63": beta_63,
            "DD_21": dd_21.fillna(0),
            "AutoCorr_15": autocorr_15,
            "Ret_1d": rets,
            "Range_Pos_20": range_pos_20,
            "Slope_P_5": slope_p,
            "Slope_V_5": slope_v,
            "Convexity": convexity,
        }
    )

    # F. Quality (Universe Filtering) - Fixed to use safe alignment
    quality_temp = pd.DataFrame(
        {
            "IsStale": np.where(
                (df_ohlcv["Volume"] == 0) | (df_ohlcv["Adj High"] == df_ohlcv["Adj Low"]), 1, 0
            ),
            "DollarVolume": df_ohlcv["Adj Close"] * df_ohlcv["Volume"],
        },
        index=df_ohlcv.index,
    )
    
    # Calculate HasSameVolume safely using TickerEngine
    quality_temp["HasSameVolume"] = TickerEngine.map_kernels(
        df_ohlcv["Volume"], lambda x: (x.diff() == 0).astype(int)
    )

    def get_quality(slice_df):
        return pd.DataFrame(
            {
                "RollingStalePct": slice_df["IsStale"].rolling(window=quality_window, min_periods=quality_min_periods).mean(),
                "RollMedDollarVol": slice_df["DollarVolume"].rolling(window=quality_window, min_periods=quality_min_periods).median(),
                "RollingSameVolCount": slice_df["HasSameVolume"].rolling(window=quality_window, min_periods=quality_min_periods).sum(),
            },
            index=slice_df.index
        )

    rolling_quality = TickerEngine.map_kernels(quality_temp, get_quality)

    return pd.concat([features_df, rolling_quality], axis=1).sort_index(), macro_df

```
---  
---  
You are absolutely right. Instead of guessing which Pandas operation is mutating the dimension, let's follow the principle: **add debug statements to capture the error and test assumptions.**

The error says one of the variables being passed into `pd.DataFrame({...})` has become a 2D array of shape `(1, 3)` instead of a 1D Pandas Series. Since our test mock data has exactly 3 days and 1 ticker, `(1, 3)` means a Series got accidentally pivoted or returned as a DataFrame.

Let's inject a diagnostic block right before it crashes to catch the exact culprit.

### Action: Update `core/features.py`

Open `notebooks_RLVR_v2/core/features.py`. Scroll down to around **line 175**, right *before* the `features_df = pd.DataFrame(...)` assembly block, and paste this debug code:

```python
    # ... previous code (convexity = ...) ...

    # ==========================================
    # DEBUG BLOCK: CAPTURE THE (1, 3) CULPRIT
    # ==========================================
    debug_vars = {
        "ATR": atr, "ATRP": natr, "TRP": trp, "RSI": rsi,
        "Mom_21": mom_21, "Consistency": consistency, 
        "IR_63": ir_63, "Beta_63": beta_63, "DD_21": dd_21,
        "AutoCorr_15": autocorr_15, "Ret_1d": rets, 
        "Range_Pos_20": range_pos_20, "Slope_P_5": slope_p, 
        "Slope_V_5": slope_v, "Convexity": convexity
    }
    
    print("\n\n" + "="*50)
    print("🕵️ DEBUG: FEATURE SHAPES AND TYPES")
    print("="*50)
    for name, var in debug_vars.items():
        var_type = type(var)
        var_shape = getattr(var, "shape", "NO_SHAPE")
        var_ndim = getattr(var, "ndim", "NO_NDIM")
        print(f"{name:<15}: Type={var_type.__name__:<15}, Shape={var_shape}, ndim={var_ndim}")
        
        if var_ndim != 1 and var_ndim != "NO_NDIM":
            print(f"  >>> 🚨 CULPRIT FOUND: {name} is not 1-dimensional! <<<")
            print(f"  >>> Value:\n{var}")
    print("="*50 + "\n")
    # ==========================================

    # E. Assemble Features
    features_df = pd.DataFrame(
        {
            "ATR": atr,
# ... rest of the file ...
```

### Action: Run the test with Output Capture

Run the test again, but this time add the `-s` flag. The `-s` flag tells `pytest` not to swallow our `print()` statements so we can read the diagnostic output.

```powershell
poetry run pytest tests/test_features.py -s -v
```

Please paste the output of the debug block here. It will tell us exactly which variable (IR_63, Beta_63, Consistency, etc.) is breaking the assembly!
---  
---  
This is a perfect example of why debugging statements are better than guessing! The output tells us exactly what went wrong:

*   `Range_Pos_20`: `Type=DataFrame, Shape=(1, 3)`
*   `Slope_V_5`: `Type=DataFrame, Shape=(1, 3)`
*   Every other feature is a `Series` with `Shape=(3,)`.

### The Root Cause: A Pandas `.apply()` Quirk
When you use `.apply()` on a **Series**, and your function returns a Series, Pandas stacks them vertically (which is what we want).
But when you use `.apply()` on a **DataFrame** (like `df_ohlcv`), and your function returns a Series, Pandas assumes you want to pivot the data (turning the dates into columns). 

Because `Range_Pos_20` and `obv` needed multiple columns (`High`, `Low`, `Close`, `Volume`), we passed the whole DataFrame to `TickerEngine`. Because they returned a single Series, Pandas pivoted them into `(1, 3)` DataFrames. Then `obv` infected `Slope_V_5`.

### The Fix
To prevent Pandas from pivoting, we just need the kernel to return a `pd.DataFrame`, and then we extract the column we want.

Please open `notebooks_RLVR_v2/core/features.py`. **Remove the Debug Block** you pasted earlier, and update the `Range_Pos` and `obv` sections (around lines 145-165) to look exactly like this:

```python
    # ... previous code (rsi = ...) ...

    def get_range_pos_kernel(df_slice):
        rp = QuantUtils.calculate_range_pos(
            df_slice["Adj High"],
            df_slice["Adj Low"],
            df_slice["Adj Close"],
            window=GLOBAL_SETTINGS.get("range_pos_period", 20),
        )
        # Wrap in DataFrame to prevent Pandas from pivoting the output
        return pd.DataFrame({"RP": rp})

    range_pos_20 = TickerEngine.map_kernels(df_ohlcv, get_range_pos_kernel)["RP"]

    def get_obv_kernel(df_slice):
        v = df_slice["Volume"]
        v_baseline = v.rolling(window=win_63d, min_periods=1).mean().replace(0, 1e-8)
        v_rel = v / v_baseline
        obv_val = QuantUtils.calculate_obv_fast(df_slice["Adj Close"], v_rel)
        # Wrap in DataFrame to prevent Pandas from pivoting the output
        return pd.DataFrame({"OBV": obv_val})

    obv = TickerEngine.map_kernels(df_ohlcv, get_obv_kernel)["OBV"]

    log_price = np.log(df_ohlcv["Adj Close"].replace(0, 1e-8))
    
    # ... rest of the code (slope_p = ...) ...
```

Run `poetry run pytest tests/test_features.py -v` one more time. It should pass perfectly. Once it passes, Phase 1 (TDD) is officially complete and we have a strict math boundary! Let me know when you get the green light.
---  
---  
Awesome! Getting those tests to pass is a huge milestone. You now have a permanent mathematical safety net protecting your core logic.

Since we are at ~123k tokens, **we should definitely wrap up this chat.** Pushing closer to the 128k/200k limit will slow down response times and could cause the AI to lose track of details during the upcoming major architectural changes.

### Wrap-Up & Next Steps

We have successfully completed **Phase 1: True TDD**. 

For our next session, we will tackle **Phase 4 (Strict Config/Typing)** and **Phase 2 (Deepening the Modules - Refactoring AlphaEngine)**. 

To start the next chat smoothly, please bundle your updated `notebooks_RLVR_v2` directory using `code2prompt` just like you did before. 

When you open the new chat, you can use this exact prompt to get us right back on track:

> **Prompt for New Chat:**
> *"Here is the updated codebase for notebooks_RLVR_v2. In our previous session, we successfully completed Phase 1 (TDD) by implementing `pytest` suites to verify mathematical integrity. We are now ready to execute **Phase 4: Enforce Ubiquitous Language via Strict Typing**. I want to eliminate the `GLOBAL_SETTINGS` dictionary in `settings.py` and convert it into a strongly typed Dataclass, then pass it explicitly to our classes. Let's do this step-by-step."*

Excellent work today! See you in the next chat when you are ready.