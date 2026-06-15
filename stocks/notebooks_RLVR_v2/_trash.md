Based on the structure of `UniverseScreener` and `get_strategy_registry`, we can formulate a precise plan to test and verify `AlphaCache`'s calculations independently.

---

### 1. Verification Strategy

The goal is to verify that `AlphaCache` correctly:
1. Aligns the trading calendar indices for different lookback periods.
2. Filters out candidate tickers based on thresholds.
3. Packages the correct historical subsets into the `MarketObservation`.
4. Executes the blueprints from the registry and returns them in a structured MultiIndex DataFrame (`[Date, Ticker]`).

Because `UniverseScreener.build_observation` and the registry strategies rely on several specific DataFrame columns, we can run a deterministic verification using **mocked/synthetic inputs** rather than live or heavy data.

---

### 2. Mock Data Requirements

To test this without running into `KeyError` or schema mismatch errors, we will design minimal, synthetic DataFrames with the following columns:

* **`trading_calendar`**: A sequential `pd.DatetimeIndex` of 10 consecutive trading days (e.g., `2026-01-01` to `2026-01-10`).
* **`df_close`**: A wide-format DataFrame (Dates $\times$ Tickers) containing simple numbers (e.g., `100.0`, `101.0`...) for mock tickers like `"AAA"` and `"BBB"`.
* **`features_df`**: A MultiIndex DataFrame indexed by `[Ticker, Date]` containing:
  * *Screener columns*: `RollMedDollarVol`, `RollingStalePct`, `RollingSameVolCount`.
  * *Observation columns*: `ATRP`, `TRP`, `ATR`, `RSI`, `Consistency`, `Mom_21`, `IR_63`, `Beta_63`, `DD_21`, `AutoCorr_15`, `Range_Pos_20`, `Slope_P_5`, `Slope_V_5`, `Convexity`.
* **`macro_df`**: A DataFrame indexed by `Date` containing: `Macro_Trend`, `Macro_Trend_Vel`, `Macro_Vix_Z`, `Macro_Vix_Ratio`.
* **`TradingConfig`**: A configuration instance set with low thresholds so our mock tickers always pass the screener.

---

### 3. Step-by-Step Test Plan

#### Step A: Mock the Strategy Registry
By default, `AlphaCache` retrieves the entire production strategy registry via `get_strategy_registry(self.config)`. To make validation predictable, we will mock/patch `data_pipeline.cache.get_strategy_registry` during the test.
* We will return a dictionary of simplified dummy blueprints (e.g., a single blueprint named `"Test_Factor"` whose formula simply returns `obs.mom_21`).
* This decouples our testing of the caching wrapper from the mathematical correctness of individual quant utilities, which should be tested separately.

#### Step B: Verify Single-Day Lookback and Alignment (`compute_alpha_ensemble`)
We will select a specific `decision_date` (e.g., index 5 in our calendar) and a lookback period (e.g., `2` days). 
1. **Manual Math**: 
   * The lookback start index is `5 - 2 = 3`. 
   * The lookback start date is the timestamp at index 3.
   * We inspect our synthetic `features_df` on index 5 for `"Mom_21"` (since our mock blueprint returns `obs.mom_21`). 
   * Let's say `"Mom_21"` is `1.5` for `"AAA"` and `2.5` for `"BBB"`.
2. **Execution**: Call `cache.compute_alpha_ensemble(decision_date, lookback_periods=[2])`.
3. **Assertions**:
   * Verify the columns of the returned DataFrame are exactly `["2d_Test_Factor"]`.
   * Verify the index matches `["AAA", "BBB"]`.
   * Verify that the values align exactly with our manual math (`1.5` and `2.5`).

#### Step C: Verify Cache Construction (`build`)
We will call `cache.build(start_date)` using a target start date from our synthetic calendar.
1. **Assertions**:
   * Verify that `cache.feature_cube` is structured with a `pd.MultiIndex` named `["Date", "Ticker"]`.
   * Verify that passing a date to `cache.get_vision(date)` correctly extracts the cross-section of features for that specific day.
   * Verify that if a ticker gets filtered out on a specific day (by setting its `RollingStalePct` to a high value in our synthetic features DataFrame), that ticker is absent from the `feature_cube` on that specific date, but present on others.

#### Step D: Verify Edge Cases
* **Lookback Out-of-Bounds**: Provide a lookback period larger than the index of the `decision_date`. Verify that the exception is caught, a warning is printed, and the program continues gracefully without crashing.
* **Empty Candidates**: Mock a day where no tickers pass the volume/stale thresholds. Verify that `compute_alpha_ensemble` returns an empty DataFrame and doesn't crash on concatenation.

---

Does this plan cover all of your validation objectives, or should we refine how the strategy blueprints are patched and mocked for the verification script?