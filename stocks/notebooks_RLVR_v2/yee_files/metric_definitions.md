## **STRATEGY REGISTRY and MACRO DASHBOARD DOCUMENTATION**
---   

### **STRATEGY REGISTRY DOCUMENTATION**

---  
### **PILLAR 1: THE TREND ENGINE (DIRECTION)**
---  
### Metric: Portfolio Log Gain (Price-Drift)

| Field | Specification |
| :--- | :--- |
| **Identity** | **Category:** Returns \| **Regime:** Trend |
| **Description** | Continuously compounded return of a price-drifted, equal-weighted portfolio over the evaluation window. |
| **Logic Tag** | `PortfolioEngine.calculate_log_gain` |

#### 1. Mathematical Logic
Calculates the growth of an $N$-asset portfolio where weights are allowed to drift based on relative performance.

1.  **Price Normalization:**
    Rebase all constituent prices to $1.0$ at the window start ($t_0$).
    $$P_{norm,i}(t) = \frac{P_i(t)}{P_i(t_0)}$$
2.  **Equity Curve Construction:**
    The portfolio value is the arithmetic mean of normalized prices (equal-weighted at $t_0$).
    $$E(t) = \frac{1}{N}\sum_{i=1}^{N} P_{norm,i}(t)$$
3.  **Log Gain Calculation:**
    $$\text{Log Gain} = \ln(E(t_{end}))$$
    *Note: Since $E(t_0) = 1.0$, $\ln(E(t_{end}) / E(t_0)) = \ln(E(t_{end}))$.*

#### 2. Strategy Interpretation
*   **Aggregate Growth Filter:** Measures the total compounded growth of the *portfolio* (not an individual ticker) over the Full, Lookback, or Holding period.
*   **Intervention Trigger:** Used to validate that the portfolio-level equity curve achieved positive drift during the holding period. `LONG` bias if Log Gain $> 0$ and supported by trend confirmation.
#### 3. Data Scaling (Agent View)
*   **Scaling Type:** `None` (Raw Log Return).
*   **Unit:** Continuously compounded return (e.g., `+0.10` = ~10.5% total return).
*   **Clipping:** Not applicable; derived directly from the terminal equity value.

---  

### Metric: Sharpe (TRP)

| Field | Specification |
| :--- | :--- |
| **Identity** | **Category:** Risk-Adjusted \| **Regime:** Efficiency |
| **Description** | Risk-adjusted efficiency of the Total Return Premium. |
| **Logic Tag** | `QuantUtils.calc_sharpe_cross_section` |

#### 1. Mathematical Logic
Calculates a cross-sectional risk-adjusted performance metric for $N$ assets using historical returns and a static risk/volatility proxy vector.

1.  **TRP Vector Calculation (Risk Denominator):**
    For each asset $i$, compute the average daily Total Return Premium ($\text{TRP}_{i,t}$) over the active evaluation window of $T$ days:
    $$\text{TRP}_i = \frac{1}{T}\sum_{t=1}^{T} \text{TRP}_{i,t}$$

2.  **Expected Return Calculation:**
    Calculate the arithmetic mean of returns for each asset $i$ over the active window, ignoring $NaN$ values:
    $$\bar{R}_i = \frac{1}{T_i}\sum_{t=1}^{T} R_{i,t}$$
    *Where $T_i$ is the count of valid non-NaN returns for asset $i$.*

3.  **Denominator Floor Protection:**
    To prevent division-by-zero or near-zero errors, apply a safety floor to the mean TRP value:
    $$\hat{\sigma}_{TRP, i} = \max(\text{TRP}_i, 10^{-8})$$

4.  **Ratio Calculation:**
    $$\text{Sharpe (TRP)}_i = \frac{\bar{R}_i}{\hat{\sigma}_{TRP, i}}$$
    *Note: Downstream cleaning converts any resulting $NaN$, positive infinite, or negative infinite values to $0.0$.*

#### 2. Strategy Interpretation
*   **Quality Dial:** Serves as a measure of trend stability and quality. High positive values indicate stable, low-volatility, institutional-led trends relative to the asset's risk premium.
*   **Intervention Trigger:** Directly influences position sizing.
    $$\text{Size Factor} = \frac{\min(\max(\text{Sharpe (TRP)}, 0), 3)}{2.0}$$
    If $\text{Sharpe (TRP)} < 0.5$, the position size is reduced by 50%.

#### 3. Data Scaling (Agent View)
*   **Scaling Type:** `None` (Raw Ratio).
*   **Unit:** Dimensionless (average return per unit of TRP).
*   **Clipping:** Raw values can be unbounded but are capped at downstream execution (e.g., clipped to $[0, 3]$ for position sizing calculations).  


---  
### Metric: Momentum (21d)  

| Field | Specification |
| :--- | :--- |
| **Identity** | **Category:** Momentum \| **Regime:** Trend |
| **Description** | Standard 21-trading-day (1-month) point-to-point percentage return using adjusted close prices. |
| **Logic Tag** | `obs.mom_21` |

#### 1. Mathematical Logic
Measures the simple percentage change over a 1-month trading window.

1.  **Window Definition:**
    Identify the current price ($P_t$) and the price 21 trading bars prior ($P_{t-21}$).
2.  **Percentage Change:**
    $$\text{Mom}_{21} = \frac{P_{t}}{P_{t-21}} - 1$$
3.  **Handling Gaps:**
    Utilizes forward-filled adjusted close prices to ensure continuity.

#### 2. Strategy Interpretation
*   **Asset Ranking:** Primary factor for cross-sectional ranking.
*   **Intervention Trigger:** `CONFIRM LONG` if 21d momentum $>$ 63d mean momentum. `AVOID` if value exceeds the `extreme_confidence` standard-deviation threshold (identifies parabolic risk).

#### 3. Data Scaling (Agent View)
*   **Scaling Type:** `None`.
*   **Unit:** Raw percentage change (e.g., `+0.15` = 15% gain).
*   **Clipping:** Capped at $\pm 4.0$ (400%).

---  
### Metric: Info Ratio (63d)
| Field | Specification |
| :--- | :--- |
| **Identity** | **Category:** Alpha \| **Regime:** Trend Quality |
| **Description** | Alpha consistency over a quarterly window. |
| **Logic Tag** | `QuantUtils.calculate_rolling_ir` |

#### 1. Mathematical Logic
Measures the risk-adjusted excess return (Alpha) relative to the benchmark.

1.  **Active Return:**
    Calculate daily returns for the asset ($R_{asset}$) and the benchmark ($R_{mkt}$).
    $$R_{active, t} = R_{asset, t} - R_{mkt, t}$$
2.  **Rolling Statistics (63d):**
    Compute the mean and standard deviation of active returns over the last 63 trading days.
3.  **Information Ratio:**
    $$\text{IR}_{63} = \frac{\text{Mean}(R_{active})}{\text{Std}(R_{active})}$$

#### 2. Strategy Interpretation
*   **Intervention Trigger:** `GATING`: Only allow 'Trend' Pillar weight $> 0.2$ if Info Ratio $> 0.5$.
*   **Agent Hint:** The 'Gatekeeper'. If IR is low, the trend is likely noise/random walk.

#### 3. Data Scaling (Agent View)
*   **Scaling Type:** `None`.

---  
### **PILLAR 2: MEAN REVERSION (THE RUBBER BAND)**

---  
### Metric: Oversold (-RSI)
| Field | Specification |
| :--- | :--- |
| **Identity** | **Category:** Mean Reversion \| **Regime:** Contrarian |
| **Description** | Inverse RSI(14) transformed into a "Pressure" gauge. |
| **Logic Tag** | `QuantUtils.calculate_rsi` |

#### 1. Mathematical Logic
The calculation follows J. Welles Wilder's original methodology, utilizing alpha-based smoothing rather than a simple moving average.

1.  **Price Change:**
    $$\Delta P_t = P_t - P_{t-1}$$
2.  **Separate Gains ($U$) and Losses ($D$):**
    $$U_t = \max(\Delta P_t, 0), \quad D_t = \max(-\Delta P_t, 0)$$
3.  **Wilder's Smoothing (Averaging):**
    Uses an Exponential Moving Average with $\alpha = 1/14$ and no initial adjustment period (`adjust=False`).
    $$\text{AvgU}_t = \alpha U_t + (1-\alpha) \text{AvgU}_{t-1}$$
    $$\text{AvgD}_t = \alpha D_t + (1-\alpha) \text{AvgD}_{t-1}$$
4.  **Relative Strength ($RS$):**
    $$\text{RS}_t = \frac{\text{AvgU}_t}{\text{AvgD}_t}$$
5.  **Relative Strength Index ($RSI$):**
    $$\text{RSI}_{14} = 100 - \frac{100}{1 + \text{RS}_t}$$
6.  **Pressure Transform:**
    $$\text{Pressure} = -RSI_{14}$$
    *Note: Higher values (e.g., -20) indicate deeper oversold conditions than lower values (e.g., -80).*

#### 2. Strategy Interpretation
*   **Intervention Trigger:** `BUY` if Value $> 100-S\_PARAMS['rsi\_oversold']$ AND Convexity $> 0.2$.
*   **Agent Hint:** Higher is more oversold. Look for the 'Hook' (Convexity $> 0$) to time entry.

#### 3. Data Scaling (Agent View)
*   **Scaling Type:** `RSI` Transform: $(Value + 50) / 20$.

---  
### Metric: Dip Buyer (-dd_21)
| Field | Specification |
| :--- | :--- |
| **Identity** | **Category:** Mean Reversion \| **Regime:** Contrarian |
| **Description** | Inverse 21-day drawdown. High = Deep pullback. |
| **Logic Tag** | `obs.dd_21` |

#### 1. Mathematical Logic
Measures the current distance from the local peak to identify oversold opportunities.

1.  **Rolling Peak:**
    Identify the maximum adjusted close price over the last 21 trading bars.
    $$P_{max, 21} = \max(P_{t}, P_{t-1}, \dots, P_{t-20})$$
2.  **Drawdown:**
    Calculate the percentage drop from that peak.
    $$\text{DD}_{21} = \frac{P_t}{P_{max, 21}} - 1$$
3.  **Inverse Transform (Dip):**
    $$\text{Dip} = -\text{DD}_{21}$$
    *Note: A 10% drawdown results in a Dip value of +0.10.*

#### 2. Strategy Interpretation
*   **Intervention Trigger:** `BUY DIP` if Value $> S\_PARAMS['strong\_confidence']$ std AND Autocorr $> 0.2$.
*   **Agent Hint:** Best used when the structural trend is still positive (Autocorr $> 0.15$).

#### 3. Data Scaling (Agent View)
*   **Scaling Type:** `None`.

---  
### Metric: Range Position (20d)
| Field | Specification |
| :--- | :--- |
| **Identity** | **Category:** Mean Reversion \| **Regime:** Boundary |
| **Description** | Price location in 20-day High/Low range ($0.0$ to $1.0$). |
| **Logic Tag** | `QuantUtils.calculate_range_pos` |

#### 1. Mathematical Logic
Determines where the current price sits relative to its recent trading channel.

1.  **Identify Boundaries:**
    Extract the highest high ($H_{20}$) and lowest low ($L_{20}$) from the last 20 trading days.
2.  **Linear Interpolation:**
    $$\text{Range Pos} = \frac{P_{close, t} - L_{20}}{H_{20} - L_{20}}$$
3.  **Agent Transform (Center):**
    The engine centers this value to range from -1.0 to +1.0 for the agent.
    $$\text{Agent View} = (\text{Range Pos} - 0.5) \times 2$$

#### 2. Strategy Interpretation
*   **Intervention Trigger:** `Value > Range High`: LONG only if OBV $> 1.0$ std; `Value < Range Low`: LONG only if OBV $< -1.0$ std.
*   **Agent Hint:** The 'Decision Fork'. Breakout at $0.8+$, Support at $0.2-$.

#### 3. Data Scaling (Agent View)
*   **Scaling Type:** `Center` Transform: $(Value - 0.5) \times 2$.

---  
### **PILLAR 3: REGIME DETECTION (THE MASTER SWITCH)**

---  
### Metric: Return Autocorr (15d)
| Field | Specification |
| :--- | :--- |
| **Identity** | **Category:** Regime \| **Regime:** Market State |
| **Description** | Price memory (Persistence vs. Mean Reversion). |
| **Logic Tag** | `QuantUtils.calculate_autocorr` |

#### 1. Mathematical Logic
Quantifies the relationship between today's return and yesterday's return to detect trend persistence.

1.  **Daily Returns:**
    Calculate standard percentage returns ($R_t$).
2.  **Lagged Returns:**
    Align returns with a 1-day lag ($R_{t-1}$).
3.  **Rolling Correlation (15d):**
    Compute the Pearson correlation coefficient over a 15-trading-day sliding window.
    $$\rho_{15} = \frac{\sum (R_t - \bar{R}_t)(R_{t-1} - \bar{R}_{t-1})}{\sqrt{\sum (R_t - \bar{R}_t)^2 \sum (R_{t-1} - \bar{R}_{t-1})^2}}$$
    *   **Value > 0.15:** Persistence (Trend Following is favored).
    *   **Value < -0.15:** Anti-persistence (Mean Reversion is favored).

#### 2. Strategy Interpretation
*   **Intervention Trigger:** Bias 'Trend' if $> 0.15$; Bias 'Reversion' if $< -0.15$; Else prioritize 'Cash'.
*   **Agent Hint:** THE MASTER SWITCH. Determines which other features to trust.

#### 3. Data Scaling (Agent View)
*   **Scaling Type:** `None`.

---  
### **PILLAR 4: VOLATILITY (RISK MANAGER)**

---  
### Metric: Low Volatility (-ATRP)
| Field | Specification |
| :--- | :--- |
| **Identity** | **Category:** Volatility \| **Regime:** Risk Filter |
| **Description** | Inverse ATR Percentage. High = Quiet market. |
| **Logic Tag** | `QuantUtils.calculate_atr` |

#### 1. Mathematical Logic
Measures normalized volatility relative to price levels.

1.  **True Range (TR):**
    Identify the max of High-Low, High-PrevClose, and Low-PrevClose.
    $$TR_t = \max[(H_t - L_t), |H_t - C_{t-1}|, |L_t - C_{t-1}|]$$
2.  **ATR Calculation (Wilder's):**
    Apply Wilder's smoothing with $\alpha = 1/14$.
    $$ATR_t = \alpha TR_t + (1-\alpha) ATR_{t-1}$$
3.  **ATR Percentage (ATRP):**
    Normalize by the current price.
    $$\text{ATRP} = \frac{ATR_t}{P_t}$$
4.  **Inverse Transform (Low Vol):**
    $$\text{Low Vol} = -\text{ATRP}$$

#### 2. Strategy Interpretation
*   **Intervention Trigger:** `RISK OFF` if Value $< -2.0$ std; `BREAKOUT WATCH` if Value $> S\_PARAMS['strong\_confidence']$ std.
*   **Agent Hint:** Standardized volatility. $0$ = Market Average.

#### 3. Data Scaling (Agent View)
*   **Scaling Type:** `Z-Score`.

---  
### **PILLAR 5: VOLUME FLOW (THE LIE DETECTOR)**

---  
### Metric: OBV Divergence (5d) TODO: NEED TO UPDATE TO MATCH REGISTRY 
| Field | Specification |
| :--- | :--- |
| **Identity** | **Category:** Volume/Fuel \| **Regime:** Confirmation |
| **Description** | Z-scored gap between relative volume flow and price trend. |
| **Logic Tag** | `calculate_obv_fast` + `slope_5d_fast` |

#### 1. Mathematical Logic
Detects "hidden" strength or weakness by comparing price trends to volume flow.

1.  **Relative Volume:**
    Scale volume by its 63-day average.
    $$V_{rel, t} = \frac{Volume_t}{\text{Mean}(Volume_{t \dots t-62})}$$
2.  **Relative OBV:**
    Accumulate volume scaled by price direction.
    $$\text{OBV}_{rel} = \sum (\text{sgn}(\Delta P_t) \cdot V_{rel, t})$$
3.  **Slope Extraction:**
    Calculate 5-day OLS slopes for both Relative OBV and Log Price. This uses the **Trend Velocity Kernel** (see Core Calculation Kernels) to extract the rate of change for each series.
    $$S_{v, t} = \text{Kernel}(\text{OBV}_{rel, t \dots t-4})$$
    $$S_{p, t} = \text{Kernel}(\ln P_{t \dots t-4})$$
4.  **Raw Divergence (Double Z):**
    Subtract the cross-sectional Z-score of price trend from the volume trend.
    $$\text{Div}_{raw} = Zscore(S_v) - Zscore(S_p)$$
5.  **Agent View:**
    The engine Z-scores this difference *again* across the universe.
    $$\text{Agent View} = Zscore(\text{Div}_{raw})$$

#### 2. Strategy Interpretation
*   **Intervention Trigger:** `INVALIDATE` Longs if Price Trend (+) but Divergence $< -1.0$ std.
*   **Agent Hint:** Detects smart money accumulation/distribution. 

#### 3. Data Scaling (Agent View)
*   **Scaling Type:** `Z-Score`.

---  
### **PILLAR 6: PHYSICS (EXHAUSTION DETECTOR)**

---  
### Metric: Convexity
| Field | Specification |
| :--- | :--- |
| **Identity** | **Category:** Physics \| **Regime:** Acceleration |
| **Description** | Second derivative of price trend. Curvature of the move. |
| **Logic Tag** | `calculate_convexity_5d_fast` |

#### 1. Mathematical Logic
Measures the acceleration or deceleration of the price trend.

1.  **Primary Slope:**
    Calculate the 5-day OLS slope of log-prices ($S_{p,t}$) using the **Trend Velocity Kernel**.
2.  **Lagged Slope:**
    Identify the slope from 2 trading days ago ($S_{p,t-2}$).
3.  **Acceleration (Convexity):**
    Calculate the rate of change in the slope.
    $$\text{Convexity} = S_{p,t} - S_{p,t-2}$$
    *   **Value > 0:** Trend is accelerating (Parabolic).
    *   **Value < 0:** Trend is flattening (Exhaustion).

#### 2. Strategy Interpretation
*   **Intervention Trigger:** `EXIT LONG` if Value $< -0.7$ (Deceleration).
*   **Agent Hint:** The 'Golden Exit'. Trend is healthy when $> 0$, exhausting when $< 0$.

#### 3. Data Scaling (Agent View)
*   **Scaling Type:** `Z-Score`.

---   

### **MACRO DASHBOARD DOCUMENTATION**

---  
### Subplot 1: Market Regime (200d MA Deviation)
Determines the structural "health" of the broad market.

1.  **Metric Calculation:**
    $$\text{Regime} = \frac{Price_{Mkt}}{SMA_{200}} - 1$$
2.  **Visualization:**
    *   **Baseline (0%):** Broad market is trading at its long-term average.
    *   **Green Shading:** Price > $SMA_{200}$ (Structural Bull).
    *   **Interpretation:** Filter for long-only strategies; avoid aggressive positions when Regime < 0.

---  
### Subplot 2: Market Momentum (Trend Velocity Z)
Measures the acceleration of the broad market regime.

1.  **Velocity Calculation:**
    Calculate the 21-day percentage change in the Regime value.
    $$\text{Vel} = \frac{\text{Regime}_t}{\text{Regime}_{t-21}} - 1$$
2.  **Z-Score Normalization:**
    Apply a 252-day rolling Z-score to the velocity.
    $$\text{Vel}_Z = Zscore(\text{Vel}, \text{window}=252)$$
3.  **Thresholds:**
    *   **Red Dash (+2.0):** Parabolic risk (Overbought).
    *   **Green Dash (-2.0):** Mean Reversion watch (Oversold).

---  
### Subplot 3: Volatility Regime (VIX Structure)
Detects systemic fear and market stress levels.

1.  **Fear Intensity ($VIX_Z$):**
    Apply a 14-day rolling Z-score to the VIX Index.
    $$VIX_Z = Zscore(VIX_{14})$$
2.  **VIX Term Structure (Ratio):**
    $$\text{Ratio} = \frac{VIX}{VIX3M}$$
3.  **Regime Classification:**
    *   **Green (STABILITY):** Ratio < 0.9. Contango (Normal fear curve).
    *   **Gray (TRANSITION):** Ratio 0.9 - 1.0. Flattening curve.
    *   **Red (SYSTEMIC SHOCK):** Ratio > 1.0. Backwardation (Extreme stress).

---  
### Subplot 4: Credit & Rates (Z-Score)
Identifies structural divergences between price action and macroeconomic fundamentals.

1.  **High Yield Spread (HYS):**
    Measures corporate credit risk. Normalized using a 252-day Z-score.
    $$HYS_Z = Zscore(\text{HYS}, \text{window}=252)$$
2.  **Yield Curve (10Y-2Y):**
    Measures recession risk/economic expectations.
    $$YC_Z = Zscore(\text{10Y2Y}, \text{window}=252)$$
3.  **Interpretation:**
    Divergences (e.g., Price rising while $HYS_Z$ spikes) signal "Lies" in the price action and suggest reducing risk.

---

### **CORE CALCULATION KERNELS**

#### 1. 5-Day Weighted Slope (Trend Velocity Kernel)
Used by **Convexity**, **OBV Divergence**, and **Slope_P_5**. Replicates an Ordinary Least Squares (OLS) linear regression slope over a 5-bar window ($t, t-1, t-2, t-3, t-4$).

*   **Derivation:** 
    For a time series with $x = [-2, -1, 0, 1, 2]$ (centered at $x=0$):
    *   $\sum x = 0$
    *   $\sum x^2 = (-2)^2 + (-1)^2 + 0^2 + 1^2 + 2^2 = 10$
    *   $\beta = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2} = \frac{\sum x_i y_i}{10}$

*   **Explicit Expansion:**
    $$\text{Slope}_t = \frac{2y_t + 1y_{t-1} + 0y_{t-2} - 1y_{t-3} - 2y_{t-4}}{10}$$

*   **Weights:** $W = [2, 1, 0, -1, -2]$ (applied to $y_{t \dots t-4}$)
*   **Logic:** This kernel provides a fast, stable estimate of short-term growth rates or momentum without the computational overhead of a full regression library.

#### 2. Wilder’s Smoothing (Signal Stability)
Used by **RSI** and **ATR**.

*   **Logic:** An Exponential Moving Average (EMA) where $\alpha = 1 / \text{period}$.
*   **Difference:** Unlike standard EMAs which often use $2 / (\text{period} + 1)$, Wilder's smoothing responds more slowly to recent changes, providing the "sturdy" signals required for regime-based filtering.
