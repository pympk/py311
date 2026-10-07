pytest -v -ra tests/test_accounting_provenance.py tests/test_scoring_provenance.py

Nugget of Wisdom (Pytest Fast Feedback):
When testing multi-gigabyte feature arrays and parquet caches, use pytest -x --lf --tb=short (-x stops on first failure, --lf runs only the last failed tests first). To isolate any provenance drift across the entire suite without running unneeded slow data generation, run pytest -k "provenance or invariants" -q.


### High-Leverage Quant Insight: The GAE Horizon Trap

> **Nugget of Wisdom:** When designing GAE for staggered multi-sleeve MTM execution, never set $\gamma$ based on standard RL rules of thumb ($\gamma = 0.99$). In an overlapping sleeve framework of holding period $H$, any credit assigned beyond horizon $H$ induces *spurious correlation* with secular market drift.

The theoretical upper bound for $\gamma$ in staggered execution should satisfy:

$$\gamma^H \le 0.50 \implies \gamma \le (0.50)^{1/H}$$

For $H = 5$, $\gamma_{\max} \approx (0.50)^{0.2} \approx \mathbf{0.87}$. Sweeps running $\gamma = 0.90$ or $0.95$ force the critic to predict macro market trends over which the 5-day rebalancing agent has zero predictive agency.

> **Nugget of Wisdom (Save Compute, Don't Train Blind):** Never tune an RL agent on a feature set until you have computed the *Information Coefficient (IC) Decay Curve* of the raw factors.
> 
> In Python, compute the Spearman rank correlation between factor values at day $T$ and forward returns at $T + 2 \rightarrow T + 2 + H$:
> 
> ```python
> # Vectorized 1-liner to check if a factor has any juice AFTER T+1 lag
> ic = (
>     df.groupby(level="date")
>     .apply(
>         lambda x: x["feature_val"].corr(
>             x["forward_ret_h5_lag1"], method="spearman"
>         )
>     )
>     .mean()
> )
> print(f"Mean Rank IC: {ic:.4f}")
> ```
> 
> If the mean IC after a 1-day lag is between $-0.01$ and $+0.01$, no reinforcement learning algorithm on earth (PPO, SAC, or Deep Q) can extract positive excess return from it. RL optimizes decision policies; it cannot manufacture signal out of random noise.

> **Nugget of Wisdom (The Transfer Coefficient Reality):** Grinold & Kahn's Fundamental Law of Active Management states:
> 
> $$\text{IR} \approx \text{TC} \times \text{IC} \times \sqrt{\text{Breadth}}$$
> 
> where $\text{TC}$ is the **Transfer Coefficient** (how much of your raw signal actually survives execution latency, turnover limits, and slippage).
> 
> When you drop $H$ from $5$ to $2$, you might theoretically increase raw rebalance trials ($\text{Breadth}$), but $\text{TC}$ collapses toward zero because slippage and execution lag consume the entire expected return.

> **Nugget of Wisdom:** When refactoring feature cubes or Parquet caches that span multiple years, use `pytest --durations=5` to instantly identify any bottleneck in rolling matrix calculations. Notice how computing `calculate_rolling_ivol` via pure NumPy array operations within `QuantUtils` keeps runtime virtually identical to a standard rolling standard deviation.

> **Nugget of Wisdom (Developer Velocity):** Run `pytest tests/test_alpha_cache.py --lf -vv` to immediately re-run only the two previously failed tests without executing the other 76 passing tests. Once green, run `pytest tests/test_mtm_invariants.py tests/test_accounting_provenance.py tests/test_rl_metrics.py tests/test_alpha_cache.py` to confirm total system integrity.

```
[00_RLVR_data_process] 
   └── MicroFeaturePipeline generates features_df 
       (Must contain Mom_252_21, SemiDev_63, Trend_R2_63)
          │
          ▼
[01_RLVR_Part1_AlphaCache] 
   └── AlphaCache runs get_strategy_registry() 
       (Bakes new 12-factor cube into alpha_cache_*.parquet)
          │
          ▼
[02_RLVR_Part2_Training] 
   └── PPOTrainer / DiscoveryEnv trains policy on new cube
          │
          ▼
[03a/03b Auto/Manual Analysis]
   └── Evaluates OOS blotters and logs to Grid_sweep.csv
```

> **Nugget of Wisdom:** In multi-factor RL ranking architectures, cross-sectional factor correlation acts like hidden policy entropy reduction. If two factors have $\rho > 0.70$, the effective degrees of freedom in the actor's Dirichlet/Softmax/L2 weighting space collapse, triggering early critic gradient plateauing ($\text{explained\_variance} < 0.20$) and premature validation early stopping.

> **Nugget of Wisdom:** When using continuous actor-critic architectures (Gaussian PPO) for factor weighting, an unconstrained $[-1,1]$ action space allows the agent to discover adversarial arbitrage against its own factor definitions. If a factor is labeled "Downside Risk" and designed to be avoided, an agent can negate it to harvest high-beta risk premium during bull regimes, creating an unhedged catastrophic tail risk in out-of-sample stress regimes. Directional factors (momentum) can be bidirectional $[-1,1]$, but risk/quality filters must often be constrained to non-negative $[0,1]$ or penalized if inverted.

> **Nugget of Wisdom:** In multi-factor scoring, a linear combination $S = \sum w_i f_i$ is only mathematically valid if all $f_i$ have identical cross-sectional variances ($\sigma_i = 1$). If unstandardized factors are combined, the implicit risk contribution of factor $i$ is proportional to $w_i \times \sigma(f_i)$, not $w_i$. A factor with 50x higher variance completely hijacks the portfolio's ordinal rank sorting.

> **Nugget of Wisdom:** In factor-ranking RL environments, unconstrained continuous action spaces allow agents to discover "factor inversion exploits"—such as shorting Downside Risk or Oversold indicators—to artificially boost beta exposure. Normalization runtime fixes must be combined with hard non-negative bounds $[0, 1.0]$ or explicit loss-aversion penalization in the reward function to prevent the policy from harvesting unhedged tail risk under the guise of alpha generation.

```
[Phase 1] Fix Normalization Runtime
 └─> Insert vectorized C-level Z-score + clip in SelectionLogic.apply_action()
 └─> Keep AlphaCache and ObservationAdapter 100% untouched.

[Phase 2] Single-Seed Validation Test (Run 132 Clone)
 └─> Run identical hyperparameters (Scale25, Gamma88, Tilt15, Width5, Seed 42).
 └─> Action space kept unconstrained [-1.0, 1.0].
 └─> Metric Gate: Inspect new Personality Plot.
      ├─> Did Downside Risk weight flip from -0.185 to >= 0?
      └─> Did OOS Sharpe rebound from 0.615 back toward 0.80+?

[Phase 3] Decision Gate:
 ├─> If Yes: Scale distortion was the sole culprit. Proceed with full 9-run grid sweep.
 └─> If No (still shorts risk): Activate `loss_aversion_penalty = 0.5` across the sweep
     to penalize downside alpha spread mathematically in the reward function.
```

### [1] Master Design Plan: The 3-Phase Surgical Correction

To maintain strict scientific rigor, we execute **one phase at a time** and run a full 9-model grid sweep after each intervention. This eliminates attribution confounding and verifies each hypothesis empirically.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 3-PHASE EXECUTION ROADMAP                                       │
├────────────────────────────────┬────────────────────────────────┬──────────────────────────────┤
│ PHASE 1: Scale Equalization    │ PHASE 2: Risk Discipline       │ PHASE 3: Horizon Decoupling  │
│ (Immediate Implementation)     │ (Conditional on Phase 1 Gate)  │ (Conditional on Phase 2 Gate)│
├────────────────────────────────┼────────────────────────────────┼──────────────────────────────┤
│ • Vectorized C-level Z-score   │ • Evaluate Personality Plot:   │ • Check for multi-horizon    │
│   in SelectionLogic.apply_action Did Downside Risk weight flip  │   momentum crowding.         │
│ • Leave AlphaCache untouched     from -0.185 to >= 0.0?         │ • If agent still constructs  │
│ • Action space kept [-1.0, 1.0]│ • If NO: Activate asymmetric   │   deceleration spreads       │
│ • Equalizes variance (125:1      loss aversion in reward        │   (+Mom_252 vs -Mom_21/126), │
│   scale blowout resolved)        (loss_aversion_penalty = 0.5)  │   prune Mom_252_21.          │
│ • Re-run 9-model Grid Sweep    │ • Re-run 9-model Grid Sweep    │ • Re-run 9-model Grid Sweep  │
│ • Gate: Avg Sharpe >= 0.70     │ • Gate: Sharpe >= 0.75,        │ • Gate: Group Sharpe > 0.78, │
│                                │   Drawdown < -20.0%            │   Ceiling Sharpe > 0.85      │
└────────────────────────────────┴────────────────────────────────┴──────────────────────────────┘
```

---

#### Phase 1: Cross-Sectional Factor Equalization (Runtime Vectorization)
- **Problem Solved:** `Mom_252_21` ($\sigma \approx 0.50$) had $125\times$ the variance of `SemiDev_63` ($\sigma \approx 0.004$), hijacking the linear ranking sort.
- **Implementation Boundary:** Inside `SelectionLogic.apply_action()`, standardize the candidate slice cross-sectionally to $\mu = 0.0, \sigma = 1.0$, clipped at $\pm 4.0$ immediately before computing `@ weights`.
- **Invariants Preserved:**
  1. `AlphaCache` remains raw, saving 50 minutes of rebuild time.
  2. `ObservationAdapter` continues to receive raw cross-sectional data, preserving the breadth (`strat_mean`) and dispersion (`strat_std`) in observation dimensions `[0:24]`.
  3. Action space remains unconstrained $[-1.0, 1.0]$ to test if scale parity alone stops the agent from shorting downside risk.
- **Verification:** Run 9-model grid sweep (Runs 137–145). Benchmark against Gen 04 (Runs 128–136).

#### Phase 2: Downside Asymmetry Enforcement (Reward Loss Aversion)
- **Problem Solved:** Linear active spread ($r_p - r_{\text{bm}}$) encourages shorting risk filters during bull training regimes.
- **Trigger Gate:** If Phase 1's sweep still shows negative weights on `Downside Risk` or `Trend Quality`.
- **Implementation Boundary:** Activate `loss_aversion_penalty = 0.5` or `1.0` in the baseline training configuration. Downside alpha loss is amplified by $1.5\times$ to $2.0\times$ in the step reward, training the Critic to heavily penalize left-tail crash risk without artificially clipping actor action bounds.
- **Verification:** Re-run 9-model grid sweep (Runs 146–154).

#### Phase 3: Factor De-Crowding & Horizon Alignment
- **Problem Solved:** 6 out of 12 factors measuring directional trend; synthetic momentum deceleration spread trades ($+MOM_{252-21}$ vs $-MOM_{126} / -MOM_{21}$).
- **Trigger Gate:** If Phase 1/2 shows persistent eigenvalue collapse ($\lambda_1 > 60\%$) or momentum whipsaw in OOS drawdowns.
- **Implementation Boundary:** Prune $MOM_{252-21}$ (120-day half-life incompatible with $H=5$ staggered holding) and replace with an orthogonal quality or volume metric (e.g., standardized turnover consistency or liquidity quality).
- **Verification:** Re-run final 9-model grid sweep targeting group Sharpe $>0.78$ and peak Sharpe $>0.85$.

---

### [2] Next Session Hand-off Package

#### Files to Upload for the Next Session:
1. `AI_CONTEXT.md` (provided below)
2. `core/logic.py` (target file for runtime Z-scoring)
3. `tests/test_scoring_provenance.py` (unit test verification)
4. `tests/test_rl_metrics.py` (action-to-scoring verification)
5. `Grid_sweep.csv` (to append new sweep results)

#### First Chat Message for Next Session:
```text
We are resuming the RLVR Staggered Sleeve PPO Alpha Engine project. 

We have completed the forensic post-mortem of the Generation 04 regression (Runs 128–136). The empirical regression was attributed to an unstandardized 125:1 variance blowout between Mom_252_21 (sigma ~0.50) and SemiDev_63 (sigma ~0.004) in SelectionLogic.apply_action(), combined with adversarial risk filter inversion under linear reward shaping.

We are now executing Phase 1 of our surgical correction roadmap:
1. Implement microsecond C-level cross-sectional Z-scoring (clip at +/- 4.0) inside SelectionLogic.apply_action() immediately before `@ weights`.
2. Keep AlphaCache and ObservationAdapter 100% untouched to preserve the 46-dimensional observation contract and avoid cache rebuilding.
3. Keep the action space unconstrained [-1.0, 1.0].
4. Run pytest verification on tests/test_scoring_provenance.py and tests/test_rl_metrics.py.
5. Prepare the execution runner for the Phase 1 9-model grid sweep (Runs 137–145).

I have uploaded:
- AI_CONTEXT.md
- core/logic.py
- tests/test_scoring_provenance.py
- tests/test_rl_metrics.py
- Grid_sweep.csv

Please review the living context and execute the exact edit to core/logic.py.
```

---

### [3] Updated `AI_CONTEXT.md`

```markdown
# AI_CONTEXT.md — Living Project Memory & State Contracts

## 1. Domain Model: Staggered Sleeve MTM Trading Engine
- **Portfolio Architecture:** Overlapping Mark-to-Market (MTM) Multi-Sleeve portfolio.
- **Capital Partitioning:** Capital is split into H equal sleeves ($W_{\text{sleeve}} = 1/H$, where $H = \text{config.holding\_period}$, default = 5).
- **Execution Convention:** Strictly Next-Day Close (T+1 Close) to eliminate lookahead bias:  
  - Day T (Close): Agent observes state $S_T$, computes action $A_T$. Stored as `pending_sleeve`. Exposure over ($T \to T+1$) is identically ZERO.
  - Day T+1 (Close): `pending_sleeve` executes at $P_{T+1}$ and becomes ACTIVE. Oldest sleeve (held H days) liquidates and drops out.
  - Day T+1 $\to$ T+H+1: Active MTM holding for H consecutive sessions.
- **Daily Portfolio Return:** Realized MTM return across all currently active sleeves:
  $$r_{p,t} = w_{\text{active}} \cdot \left(\frac{1}{H} \sum \text{sleeve\_returns}\right) + w_{\text{benchmark}} \cdot r_{\text{bm},t} + w_{\text{cash}} \cdot r_{\text{cash},t} - \text{slippage}_t$$
- **Capital Mandate:** 100% Gross Equity Exposure (`max_cash_pct = 0.0`). Cash hoarding is forbidden. Downside defense is governed via cross-sectional stock selection and dynamic benchmark beta allocation ($w_{\text{benchmark}} = 1.0 - w_{\text{active}}$).

---

## 2. Execution Horizon & Truncation Invariant Contracts
- **`episode_steps = 0` (Strict Unbounded Evaluation):**
  - Signifies unbounded, full-calendar execution. Step truncation (`steps_done`) is disabled when `episode_steps == 0`.
  - Mandatory for all validation (`env_val`), out-of-sample evaluation (`env_test`), backtesting, and production inference.
  - `calendar_done` occurs at `len(calendar) - 2` sessions. Any blotter shorter than `len(calendar) - 10` is classified as truncated/stale and must be auto-healed.
- **`episode_steps = NUM_STEPS` (Rollout Workers Only):**
  - Reserved strictly for vectorized training workers in `make_train_env()` with `randomize_start = True`.
- **Validation Regime Integrity:**
  - `env_val` must execute across all 1,568 sessions of `cal_val` (`2016-01-04` to `2022-03-24`) without step truncation. Early stopping evaluates policy survivability across bull and bear regimes.
- **Temporal Alignment in Evaluation:**
  - Benchmark metrics (`bm_sub_cum`, `excess_daily`, beta, IR) must be computed over the identical trading dates recorded in the agent's blotter, never across disparate calendar lengths.

---

## 3. State, Action & Model Contracts

### Observation Space (46 Dimensions)
- `[0:12]` Cross-Sectional Mean (12 dims): Normalized feature means across universe. Computed dynamically in `ObservationAdapter` from raw `ensemble.mean(axis=0)`.
- `[12:24]` Cross-Sectional Std (12 dims): Normalized feature standard deviations across universe. Computed dynamically in `ObservationAdapter` from raw `ensemble.std(axis=0)`.
- `[24:36]` Benchmark Vector (12 dims): Trend, volatility, momentum of `config.benchmark`.
- `[36:46]` Stationary Macro Vector (10 dims): Yield curve slope, credit spreads, macro indicators.
*Invariant:* The `ensemble` DataFrame in `AlphaCache` must preserve raw units so dims `[0:24]` reflect true cross-sectional market drift and dispersion.

### 12 Strategy Factors (Active Registry Contract)
1. `Momentum (12-1m)` (Fama-French Structural Trend, $t-252 \to t-21$, $\tau \approx 120\text{d}$)
2. `Sharpe (TRP)` (Risk-Adjusted Efficiency over 41d lookback)
3. `Momentum (21d)` (1-Month Tactical Trend)
4. `Info Ratio (63d)` (Quarterly Benchmark Alpha Consistency)
5. `Oversold (-RSI)` (Mean Reversion, bounded $[-1, 1]$)
6. `Dip Buyer (-dd_21)` (Pullback Quality, bounded $[0, 1]$)
7. `Trend Quality (63d)` (Price Linearity $r(t, \ln P)$, bounded $[-1, 1]$)
8. `Downside Risk (-SemiDev_63)` (Inverse Downside Semi-Deviation relative to 0.0)
9. `Low Volatility (-ATRP)` (Cross-Sectional Absolute Volatility Quality)
10. `Momentum (63d)` (Quarterly Trend Anchor, $\tau \approx 35\text{d}$)
11. `Momentum (126d)` (Semi-Annual Structural Momentum, $\tau \approx 70\text{d}$)
12. `Residual Low-Vol (63d)` (Inverse CAPM Idiosyncratic Risk, $\tau \approx 45\text{d}$)

### Action Space (16 Dimensions, Tanh bounded [-1, 1])
- `[0:12]` Alpha Weights: L2-normalized feature scoring weights for the 12 registry factors.
- `[12]` Offset: Cross-sectional rank start percentile `[0, rank_max_offset]`.
- `[13]` Width: Basket size mapped to `[min_basket_width, rank_max_width]` (locked to min_width = 5).
- `[14]` Equity Exposure: Mapped to `[1.0 - max_cash_pct, 1.0]` (locked to 1.0 under Zero Cash mandate).
- `[15]` Active Tilt: Blending ratio between active stock basket and benchmark beta shelter `[min_active_tilt, 1.0]` (floor: 0.20).

---

## 4. Mathematical & Accounting Conventions
- **Pure Math Separation:** NumPy vector reductions and compounding logic reside strictly in `core/quant.py` (`QuantUtils`). DataFrame slicing and FIFO sleeve tracking reside in `core/accounting.py` (`MTMPortfolioEngine`).
- **Factor Variance Equalization Rule:** All strategy factor columns must be cross-sectionally standardized:
  $$Z_{i, t} = \text{clip}\left(\frac{f_{i, t} - \mu_{i, t}}{\max(\sigma_{i, t}, 10^{-8})}, -4.0, 4.0\right)$$
  inside `SelectionLogic.apply_action()` immediately prior to matrix multiplication with action weights. Linear combination across unstandardized factors is strictly prohibited.
- **Compounding Wealth Curves (Base 1.0):**
  $$V_p(t) = V_p(t-1) \cdot (1 + r_{\text{net},t}), \quad V_{\text{bm}}(t) = V_{\text{bm}}(t-1) \cdot (1 + r_{\text{bm},t})$$
- **Alpha Multiplier (Base 1.0): Strictly Geometric Ratio:**
  $$\text{Alpha Multiplier}_t = \frac{V_p(t)}{V_{\text{bm}}(t)}$$
- **Banned Math:** Compounding $(1 + \text{reward})$ or $(1 + r_p - r_{\text{bm}})$ is strictly prohibited.
- **RL Step Reward:** Linear spread $r_t = r_{p,t} - r_{\text{bm},t}$ when `upside_alpha_mult = 1.0` and `loss_aversion_penalty = 0.0`. Under `loss_aversion_penalty > 0`, negative alpha is scaled by $(1.0 + \text{penalty})$.

---

## 5. Artifacts, Serialization & Checkpoint Contracts
- **Checkpoint Path:** `output/model_checkpoints/model_{ID}_s{seed}_ep_{epoch}.pt`
- **Catalog Registry:** Single source of truth catalog at `output/model_catalog.parquet`.
- **Leaderboard Log:** `Grid_sweep.csv` records all OOS evaluated checkpoints with strict deduplication on model stem.
- **Required Artifact Outputs per Model:**
  1. `output/model_checkpoints/model_{STEM}.pt`
  2. `output/blotter_df_{STEM}.parquet` (Strict geometric alpha_equity)
  3. `output/results_{STEM}.pkl` (Full telemetry dictionary: equity curves, metrics, blotter records)

---

## 6. Verification & Tripwire Invariants
- **Gate:** `pytest tests/test_mtm_invariants.py tests/test_accounting_provenance.py tests/test_rl_metrics.py tests/test_scoring_provenance.py`
- **Dynamic Config:** Never hardcode `'SPY'` or fixed holding periods. Always consume `config.benchmark` and `config.holding_period`.
- **Purge Gap:** An $H$-day purge gap must be enforced before train/val and val/test split boundaries.

---

## 7. Empirical Findings & Meta-Analysis History

### Generation 01–02: Fast Technicals Ceiling (Runs 101–118)
- 5-day technical features (`Slope_P_5_Z`, `Slope_V_5_Z`, `Convexity`) under $T+1$ execution latency hit an empirical ceiling of ~75%–78% OOS return vs SPY +79.9% (Sharpe ~0.640, Beta ~0.99, IR ~ -0.626). Short-term signals decayed before sleeve execution.

### Generation 03: Intermediate Horizon Breakthrough (Runs 119–127)
- Replaced the 3 fast technicals with `Mom_63`, `Mom_126`, and `Residual Low-Vol (63d)`.
- Mean OOS Sharpe jumped to **0.703 (+9.8%)**, cross-seed standard deviation fell to **0.085 (-28.0%)**.
- **Frontier Peak:** Run 123 (`P5_Scale25_Gamma88_Tilt15_Width5_s42`) achieved **79.27% Return, 0.853 Sharpe, 1.204 Sortino, -1.59% Excess, and -0.038 Information Ratio**.

### Generation 04: The Scale Distortion Regression (Runs 128–136)
- Attempted factor orthogonalization: pruned `Log Price Gain`, `Autocorr_15`, `Range_Pos_20`; added `Mom_252_21`, `SemiDev_63`, `Trend_R2_63`.
- **Empirical Regression:** Mean Sharpe collapsed from 0.703 to 0.650 (-7.5%), IR decayed from -0.584 to -0.754, Champion Run 123 equivalent (Run 132) collapsed to 0.615 Sharpe.
- **Forensic Root Causes Identified:**
  1. *The 125:1 Variance Disparity:* Raw `Mom_252_21` ($\sigma \approx 0.50$) completely drowned out `SemiDev_63` ($\sigma \approx 0.004$) by a factor of 125x because `SelectionLogic.apply_action()` lacked cross-sectional standardization.
  2. *Adversarial Risk Inversion:* Unconstrained $[-1, 1]$ action bounds under linear spread rewards allowed the agent to output $-0.185$ on `Downside Risk`, actively buying high-crash-risk stocks to chase bull-market upside spread, which blew up out of sample.
  3. *Multi-Horizon Trend Collinearity:* 6 of 12 factors measured directional drift. The agent constructed a synthetic momentum deceleration spread trade ($+Mom_{252-21}$ vs $-Mom_{126} / -Mom_{21}$), buying stale rolling-over winners.

---  

## 8. Surgical Correction Plan (Generations 05+)  

### Phase 1: Runtime Scale Equalization (Current Focus)   
- Insert C-level vectorized Z-score `(x - mu) / sigma` clipped at $\pm 4.0$ in `SelectionLogic.apply_action()` prior to scoring.
- Keep `AlphaCache` and `ObservationAdapter` untouched.
- Action space remains unconstrained $[-1.0, 1.0]$ to isolate the scale hypothesis.
- Run 9-model grid sweep (Runs 137–145).

### Phase 2: Downside Asymmetry Enforcement (Reward-Level Discipline)
- If Phase 1 still outputs negative weights on `Downside Risk` or `Trend Quality`, activate `loss_aversion_penalty = 0.5` or `1.0` in the reward function.
- Disciplines the Critic to penalize left-tail risk without restricting actor action dimensionality.
- Re-run 9-model grid sweep (Runs 146–154).

### Phase 3: Horizon Decoupling
- If multi-horizon deceleration spreads persist, prune `Mom_252_21` ($\tau \approx 120\text{d}$) to align with the $H=5$ staggered holding period.
- Final grid sweep targeting group Sharpe $>0.78$ and frontier Sharpe $>0.85$.
```

Quant Nugget of Wisdom: When evaluating policies trained under asymmetric rewards ($\lambda > 0$), expect `avg_reward` in the diagnostics plot to remain structurally negative throughout training. The true gauge of Critic accuracy is `explained_variance` rising into $[0.25, 0.50]$ alongside a declining `clip_fraction`. Do not increase `REWARD_SCALE` simply to make the step reward positive, as that inflates policy loss gradients and causes PPO instability.

Quant Nugget of Wisdom: In multi-sleeve staggered architectures, set $\gamma$ such that the half-life of discounting $t_{1/2} = \frac{\ln(0.5)}{\ln(\gamma)}$ closely matches the sleeve holding duration $H$. For $H=5$, $\gamma = 0.869$ yields $t_{1/2} \approx 4.93$ sessions—an almost perfect physical match to your 5-day liquidation boundary.

```
========================================================================================================================
PHASE 3 EXPERIMENTAL MATRIX (P8 SERIES — 9 CONFIGURATIONS)
========================================================================================================================
Slice   Run ID                                           Scale   Gamma   λ (LossAv)   Upside   Tilt    Target Hypothesis
------------------------------------------------------------------------------------------------------------------------
1       P8_Scale25_Gamma88_LossAv030_Up11_Tilt15_Width5  25.0    0.88    0.30         1.10     0.15    Mild penalty + upside push
1       P8_Scale25_Gamma88_LossAv035_Up11_Tilt15_Width5  25.0    0.88    0.35         1.10     0.15    Run 148 Champion Evolution
1       P8_Scale25_Gamma88_LossAv040_Up11_Tilt15_Width5  25.0    0.88    0.40         1.10     0.15    Strong penalty + upside push
------------------------------------------------------------------------------------------------------------------------
2       P8_Scale25_Gamma88_LossAv035_Up115_Tilt15_Width5 25.0    0.88    0.35         1.15     0.15    Aggressive alpha incentive
2       P8_Scale25_Gamma88_LossAv035_Up11_Tilt10_Width5  25.0    0.88    0.35         1.10     0.10    Expanded shelter room (90%)
2       P8_Scale25_Gamma88_LossAv035_Up115_Tilt10_Width5 25.0    0.88    0.35         1.15     0.10    High shelter + high upside
------------------------------------------------------------------------------------------------------------------------
3       P8_Scale25_Gamma90_LossAv035_Up11_Tilt15_Width5  25.0    0.90    0.35         1.10     0.15    2x H-cycle credit assignment
3       P8_Scale25_Gamma90_LossAv035_Up115_Tilt10_Width5 25.0    0.90    0.35         1.15     0.10    Full multi-objective coupling
3       P8_Scale30_Gamma88_LossAv035_Up11_Tilt15_Width5  30.0    0.88    0.35         1.10     0.15    High gradient SNR test
========================================================================================================================
```
```
High-Leverage Nugget of Wisdom: When running multi-seed sweeps in Ray or Gym vectorized loops, always append the seed directly to the tracking tag (f"{ID}_s{seed}") before evaluating .completed_* sentinel markers. This avoids collision issues across distributed instances and ensures incomplete runs can be resumed cleanly with python -m pytest --lf or an automated checkpoint recovery guard.
```

💡 Nugget of Wisdom: Vectorized Welford Scaler in Vectorized Envs
In multi-worker vector environments (SyncVectorEnv with 8 workers), each worker maintains an independent ObservationScaler that sees only 
1
/
N
1/N
 of the transitions. In production RL, you can achieve global normalization stability across all 8 parallel workers with zero IPC overhead by performing a vectorized all-reduce reduction on the Welford accumulators:
code
Python
# Aggregate running statistics across all N parallel envs in one step:
total_count = sum(e.scaler.count for e in envs.envs)
global_mean = sum(e.scaler.mean * e.scaler.count for e in envs.envs) / total_count
Synchronizing the global scaler into gym_val ensures that the validation policy operates on the true distribution mean of the multi-worker training manifold. 

💡 Nugget of Wisdom: Immutability Boundaries in Quant RL
In production quant architectures, separate data preprocessing from RL training using an Immutability Contract:
Upstream (Part 1 / AlphaCache): Produces deterministic, read-only feature tensors keyed by deterministic content hashes (lookback, universe_hash, registry_hash).
Downstream (Part 2 / Agent): Consumes the cached parquet as an immutable read-only memory map.
Unless you change a formula in strategy/registry.py, alter features_df, or modify CACHE_LOOKBACK, the cache file is mathematically permanent and should never be invalidated by agent, trainer, or environment changes.

```
Safety Feature: Restoring Anchors Anytime
If an exploratory script or notebook ever wipes the continuous blotters from output/, you can restore them instantly with this one-liner in PowerShell:
code
Powershell
Copy-Item output\canonical_anchors\*.parquet output\
```

> **High-Leverage Quant Nugget:**  
> When verifying stateful MTM execution queues across walk-forward splits, always account for **warmup boundary conditions**. A rolling FIFO queue of length $H$ that resets at walk-forward boundaries will produce different returns than a continuous queue if unallocated slots are filled with benchmark beta rather than omitted. Explicitly simulating engine resets on boundary dates guarantees bit-level parity across historical and live execution environments.

> **High-Leverage Quant Rule:**  
> Never fine-tune an RL financial agent on new data with standard unconstrained gradient steps. Without policy-space anchors ($\beta \cdot D_{\text{KL}}$) and stratified replay sampling, the policy will overfit to the most recent quarter's macro regime and lose generalizable factor-timing capabilities.

### 1. "As long as it's picking good tickers, do we care?"

**Yes, you must care deeply. In quantitative finance, failing this check is the #1 cause of "live-trading blowups after a brilliant backtest."**

Here is why:

#### A. The Blotter is History; The Checkpoint is Your Future
The blotter is just a historical text file recording what happened during the simulation. 
Tomorrow morning at 9:30 AM, **the blotter does not exist**. The only thing you can run in production is the PyTorch checkpoint file on disk:
`model_chunk3_s42_champion.pt`

If you haven't verified that loading that `.pt` file and passing state $S_T$ outputs the exact same action:
* **What if the checkpoint saved was the wrong epoch?** (e.g. the last epoch instead of the best validation epoch).
* **What if action decoding was stochastic during evaluation?** (e.g. the blotter recorded a lucky exploration sample $\sim \mathcal{N}(\mu, \sigma)$, but deterministic inference in live trading uses $\mu$, which might pick completely different, losing stocks).
* **What if observation normalization state (mean/std) wasn't saved with the weights?** If the model expects input $z$-scored with 2022 stats, but you feed it raw 2026 data, the neural network will output garbage numbers.

> **Institutional Axiom:** If your PyTorch `.pt` file cannot reproduce yesterday's blotter row from yesterday's observation tensor, **you do not own a strategy—you own a simulation artifact.**

---

### 2. Policy-Space Anchors ($\beta \cdot D_{\text{KL}}$) Explained Like You're a Student

Imagine you are training a high school student to become a master chef.

#### The Problem: Catastrophic Forgetting
1. **Chunk 0 (High School, 4 Years):** You spend 4 years teaching the student the universal fundamentals of French cooking: knife skills, mother sauces, baking chemistry, temperature control. They become a well-rounded chef.
2. **Chunk 1 (Internship, 1 Year):** You send the student to a trendy Korean BBQ pop-up for 1 year. 
3. **The Disaster:** A standard neural network is like a student with severe amnesia. When fine-tuned *only* on the Korean BBQ menu, its brain rewires itself completely. By month 6, it can make incredible bulgogi, but it has **completely forgotten how to make a roux, temper chocolate, or bake bread**. 

In machine learning, this is called **Catastrophic Forgetting**. When an RL agent trains on the 2023–2024 AI bull market, it unlearns how to survive a 2022 interest-rate crash.

---

#### The Traditional (Flawed) Solution: Weight Decay ($L_2$ Regularization)
Teachers used to say: *"Don't let any brain synapses change too much."* (Penalize $\sum w_i^2$).
This fails because neural networks are wildly non-linear:
* You can change 1,000 weight values by $5\%$ and the chef’s cooking doesn't change at all.
* Or you can change **one single critical weight** by $0.001\%$, and the chef suddenly pours salt instead of sugar into every dessert.

---

#### The Master Solution: The Policy-Space Anchor ($\beta \cdot D_{\text{KL}}$)
Instead of policing the **synapses (weights)**, we police the **behavior (decisions)**.

1. **The Frozen Clone (The Anchor):** Before the chef goes to the new restaurant, we create a frozen holographic copy of their brain: $\pi_{\text{anchor}}$.
2. **The Leash (Kullback-Leibler Divergence, $D_{\text{KL}}$):** 
   $D_{\text{KL}}(\pi_{\text{new}} \parallel \pi_{\text{anchor}})$ is a mathematical tape measure that measures:
   *"How different are the new chef's taste choices from the old master chef's choices across all possible dishes?"*
3. **The Loss Function:**
   $$\text{Total Loss} = \text{PPO Reward Loss} + \beta \cdot D_{\text{KL}}(\pi_{\text{new}} \parallel \pi_{\text{anchor}})$$
   * **PPO Reward Loss** says: *"Adapt to the new year! Find the new alpha!"*
   * **$\beta \cdot D_{\text{KL}}$ (The Bungee Cord)** says: *"If you change your core trading instincts too radically from the master chef, I will penalize you severely."*
   * **$\beta = 0.05$** is the stiffness of the bungee cord: elastic enough to learn the new regime, but stiff enough to prevent amnesia.

---

### 3. How to Load the Models and Get Tomorrow's Ticker Picks

Here is the complete, runnable production inference script. It:
1. Loads the latest market data and computes the current 46-dimensional observation $S_{\text{today}}$.
2. Loads all 3 champion seed models (`s42`, `s101`, `s777`).
3. Runs deterministic policy inference (`agent.get_deterministic_action()`).
4. Generates discrete stock baskets, SPY beta shelter weights, and nets orders for execution.

```python
import os
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import torch

from core.contracts import ProcessedDataBundle
from core.logic import SelectionLogic
from core.paths import OUTPUT_DIR
from core.settings import TradingConfig
from data_pipeline.cache import AlphaCache
from data_pipeline.loader import load_processed_data
from data_pipeline.screener import UniverseScreener
from rl_discovery.adapter import ObservationAdapter, ObservationScaler
from rl_discovery.agent import AbsoluteZeroAgent


def generate_tomorrow_orders(
    portfolio_equity: float = 100_000.0,
    chunk_id: int = 3,
    seeds: List[int] = [42, 101, 777],
) -> Dict[str, Any]:
    config = TradingConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print(f"🚀 GENERATING PRODUCTION ORDERS: TOTAL EQUITY = ${portfolio_equity:,.2f}")
    print("=" * 80)

    # 1. Ingest Latest Market & Macro State
    bundle: ProcessedDataBundle = load_processed_data()
    alpha_cache = AlphaCache(config)
    screener = UniverseScreener(config)

    # Extract the most recent available trading date
    all_dates = bundle.features_df.index.get_level_values("Date").unique().sort_values()
    latest_date = all_dates[-1]
    print(f"📅 Ingested Market State for Decision Date: {latest_date.strftime('%Y-%m-%d')}")

    # Build 46-dim Observation Vector
    adapter = ObservationAdapter(config)
    obs_raw = adapter.process(
        bundle=bundle,
        target_date=latest_date,
        screener=screener,
        alpha_cache=alpha_cache,
    )
    obs_tensor = torch.as_tensor(obs_raw, dtype=torch.float32, device=device).unsqueeze(0)

    # Sleeve capital allocation (1/3 per seed, 1/H per sleeve)
    seed_capital = portfolio_equity / len(seeds)
    sleeve_capital = seed_capital / config.holding_period

    production_orders = {}

    for seed in seeds:
        # 2. Locate and Load Champion Checkpoint
        ckpt_name = f"model_chunk{chunk_id}_s{seed}_champion.pt"
        ckpt_path = OUTPUT_DIR / "model_checkpoints" / ckpt_name
        if not ckpt_path.exists():
            matches = list(OUTPUT_DIR.rglob(ckpt_name))
            if not matches:
                raise FileNotFoundError(f"❌ Checkpoint not found: {ckpt_name}")
            ckpt_path = matches[0]

        agent = AbsoluteZeroAgent(obs_dim=46, action_dim=16).to(device)
        checkpoint = torch.load(ckpt_path, map_location=device)
        agent.load_state_dict(checkpoint["model_state_dict"])
        agent.eval()

        # 3. Deterministic Policy Forward Pass (No exploration noise)
        with torch.no_grad():
            action = agent.get_deterministic_action(obs_tensor).squeeze(0).cpu().numpy()

        # 4. Map Continuous Action to Discrete Basket via SelectionLogic
        ensemble = alpha_cache.get_vision(latest_date)
        selection = SelectionLogic.apply_action(
            ensemble=ensemble,
            action=action,
            config=config,
            benchmark=config.benchmark,
        )

        active_dollars = sleeve_capital * selection.weight_active
        bm_dollars = sleeve_capital * selection.weight_benchmark
        dollars_per_stock = active_dollars / len(selection.selected_tickers)

        production_orders[seed] = {
            "tickers": selection.selected_tickers,
            "w_active": selection.weight_active,
            "w_benchmark": selection.weight_benchmark,
            "dollars_per_stock": dollars_per_stock,
            "benchmark_dollars": bm_dollars,
        }

        print(f"\n🌱 Seed {seed:03d} Action Output:")
        print(f"  • Active Tilt / Benchmark Split : {selection.weight_active*100:.1f}% Active | {selection.weight_benchmark*100:.1f}% {config.benchmark}")
        print(f"  • Selected Basket ({len(selection.selected_tickers)} stocks)  : {selection.selected_tickers}")
        print(f"  • Dollar Allocation per Stock   : ${dollars_per_stock:,.2f}")
        print(f"  • Dollar Allocation into {config.benchmark} : ${bm_dollars:,.2f}")

    # 5. Aggregate Net Target Dollars across Seeds for Tomorrow's MOC execution
    net_ticker_dollars: Dict[str, float] = {}
    for seed, data in production_orders.items():
        for tkr in data["tickers"]:
            net_ticker_dollars[tkr] = net_ticker_dollars.get(tkr, 0.0) + data["dollars_per_stock"]
        net_ticker_dollars[config.benchmark] = (
            net_ticker_dollars.get(config.benchmark, 0.0) + data["benchmark_dollars"]
        )

    print("\n" + "=" * 80)
    print(f"📋 NET MOC BUY ORDERS TO PLACE FOR TOMORROW'S CLOSE (T+1)")
    print("=" * 80)
    for tkr, dollars in sorted(net_ticker_dollars.items(), key=lambda x: x[1], reverse=True):
        print(f"  • BUY {tkr:<6} : ${dollars:>9,.2f} MOC")
    print("=" * 80)
    print("⚠️ REMINDER: Also liquidate the oldest expiring sleeve (held 5 days) at tomorrow's close.")
    
    return production_orders


# Execute when ready:
# orders = generate_tomorrow_orders(portfolio_equity=100_000.0)
```
```
💡 Nugget of Wisdom (The Fine-Tuning Plateau Principle):
In Reinforcement Learning for financial time series, fine-tuning is an exercise in regularized transfer learning, not reward maximization. If an RL agent fails to find a better policy within 5–10 epochs of warm-started fine-tuning, training longer will only accelerate memorization of noise. When fine-tuning, early stopping patience should never exceed 15 epochs.
```