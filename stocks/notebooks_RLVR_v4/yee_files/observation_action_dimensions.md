Here is the detailed breakdown of the calculations and the exact dimension names for both the observation space and the action space in the updated setup.

---

### 1. Observation Dimensions (State Space)

#### **The Calculation**
$$\text{Total Observation Dimensions} = (\text{Cross-Sectional Features} \times 2) + \text{Macro Features}$$

$$\text{Total} = (12 \times 2) + 11 = 24 + 11 = 35 \text{ dimensions}$$

* **Why multiply by 2?** For each of the 12 features, the environment adapter computes two cross-sectional stats across all valid candidate tickers on that day: the **Mean** (representing average market alignment) and the **Standard Deviation** (representing cross-sectional dispersion).

#### **Names of each of the 35 Observation Dimensions**

| Dim Index | Dimension Name | Description |
| :--- | :--- | :--- |
| **0** | `Mean: Log Price Gain (Z)` | Average momentum across candidates |
| **1** | `Mean: Sharpe (TRP) (S)` | Average risk-adjusted trend quality across candidates |
| **2** | `Mean: Momentum (21d) (S)` | Average 1-month momentum across candidates |
| **3** | `Mean: Info Ratio (63d) (S)` | Average consistency of alpha across candidates |
| **4** | `Mean: Oversold (-RSI) (S)` | Average contrarian indicator across candidates |
| **5** | `Mean: Dip Buyer (-dd_21) (S)` | Average drawdown depth across candidates |
| **6** | `Mean: Range Position (20d) (S)` | Average position within high/low boundaries |
| **7** | `Mean: Return Autocorr (15d) (S)` | Average price memory/persistence across candidates |
| **8** | `Mean: Low Volatility (-ATRP) (Z)` | Average inverse volatility across candidates |
| **9** | `Mean: Slope_P_5_Z (Z)` | Average short-term price velocity across candidates |
| **10** | `Mean: Slope_V_5_Z (Z)` | Average short-term volume velocity across candidates |
| **11** | `Mean: Convexity (Z)` | Average trend curvature across candidates |
| **12** | `Std: Log Price Gain (Z)` | Dispersion of momentum across candidates |
| **13** | `Std: Sharpe (TRP) (S)` | Dispersion of risk-adjusted trend quality across candidates |
| **14** | `Std: Momentum (21d) (S)` | Dispersion of 1-month momentum across candidates |
| **15** | `Std: Info Ratio (63d) (S)` | Dispersion of alpha consistency across candidates |
| **16** | `Std: Oversold (-RSI) (S)` | Dispersion of contrarian indicators across candidates |
| **17** | `Std: Dip Buyer (-dd_21) (S)` | Dispersion of drawdown depth across candidates |
| **18** | `Std: Range Position (20d) (S)` | Dispersion of boundary positions |
| **19** | `Std: Return Autocorr (15d) (S)` | Dispersion of price memory across candidates |
| **20** | `Std: Low Volatility (-ATRP) (Z)` | Dispersion of inverse volatility across candidates |
| **21** | `Std: Slope_P_5_Z (Z)` | Dispersion of price velocity across candidates |
| **22** | `Std: Slope_V_5_Z (Z)` | Dispersion of volume velocity across candidates |
| **23** | `Std: Convexity (Z)` | Dispersion of curvature across candidates |
| **24** | `Mkt_Ret` | Raw benchmark (SPY) return over the step |
| **25** | `Mkt_Ret_Z` | Standardized benchmark return |
| **26** | `Macro_Trend` | Trend indicator of the overall market |
| **27** | `Macro_Trend_Z` | Standardized trend indicator of the overall market |
| **28** | `High_Yield_Spread_Z` | Credit risk indicator (High Yield corporate bond spread) |
| **29** | `Yield_Curve_10Y2Y_Z` | Economic health indicator (10Y minus 2Y Treasury spread) |
| **30** | `Macro_Trend_Vel_Z` | Speed of the macro trend |
| **31** | `Macro_Trend_Mom` | Momentum of the macro trend |
| **32** | `Macro_Vix_Z` | Standardized market volatility (VIX level) |
| **33** | `Macro_Vix_Ratio` | VIX ratio (current vs. historic baseline) |
| **34** | `Mkt_Vol_63d_Z` | Benchmark historical volatility over 3 months |

---

### 2. Action Dimensions (Action Space)

#### **The Calculation**
$$\text{Total Action Dimensions} = \text{Strategy Feature Weights} + \text{Rank Parameters}$$

$$\text{Total} = 12 + 2 = 14 \text{ dimensions}$$

* **Why add 2?** The first 12 dimensions output by the neural network policy are continuous weights assigned to each of your active strategies. The final 2 dimensions represent control decisions used to map the scored tickers into a specific buy window.

#### **Names of each of the 14 Action Dimensions**

| Dim Index | Action Name | Category | Description |
| :--- | :--- | :--- | :--- |
| **0** | `Log Price Gain` | Strategy Weight | Weight assigned to raw log-return |
| **1** | `Sharpe (TRP)` | Strategy Weight | Weight assigned to risk-adjusted performance |
| **2** | `Momentum (21d)` | Strategy Weight | Weight assigned to 1-month trend |
| **3** | `Info Ratio (63d)` | Strategy Weight | Weight assigned to alpha consistency |
| **4** | `Oversold (-RSI)` | Strategy Weight | Weight assigned to contrarian strength |
| **5** | `Dip Buyer (-dd_21)` | Strategy Weight | Weight assigned to pullback depth |
| **6** | `Range Position (20d)` | Strategy Weight | Weight assigned to range boundaries |
| **7** | `Return Autocorr (15d)` | Strategy Weight | Weight assigned to price persistence |
| **8** | `Low Volatility (-ATRP)` | Strategy Weight | Weight assigned to inverse volatility |
| **9** | `Slope_P_5_Z` | Strategy Weight | Weight assigned to price velocity |
| **10** | `Slope_V_5_Z` | Strategy Weight | Weight assigned to volume velocity |
| **11** | `Convexity` | Strategy Weight | Weight assigned to trend curvature |
| **12** | `Rank Offset (Start Pos)` | Control Parameter | Decoded to set the selection starting rank ($[0, \text{max\_offset}]$) |
| **13** | `Rank Width (0 = Cash)` | Control Parameter | Decoded to set the number of assets to purchase ($[0, \text{max\_width}]$) |