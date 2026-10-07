# AI_CONTEXT.md — Living Project Memory & State Contracts

## 1. Domain Model: Staggered Sleeve MTM Trading Engine
- **Portfolio Architecture:** Overlapping Mark-to-Market (MTM) Multi-Sleeve portfolio.
- **Capital Partitioning:** Capital is split into $H$ equal sleeves ($W_{\text{sleeve}} = 1/H$, where $H = \text{config.holding\_period}$, default = 5).
- **Intra-Sleeve Weighting (Strict Invariant):** Uniform equal weighting ($1/K_t$). Inverse-volatility weighting ($1/\text{ATRP}$) is permanently revoked following Gen 15 empirical refutation. All active sleeves allocate capital uniformly across selected assets.
- **Execution Convention:** Strictly Next-Day Close ($T+1$ Close) to eliminate lookahead bias:  
  - Day $T$ (Close): Agent observes state $S_T$, outputs action $A_T$. Stored as `pending_sleeve`. Exposure over ($T \to T+1$) is identically ZERO.
  - Day $T+1$ (Close): `pending_sleeve` executes at $P_{T+1}$ and becomes ACTIVE. Oldest sleeve (held $H$ days) liquidates and drops out.
  - Day $T+1 \to T+H+1$: Active MTM holding for $H$ consecutive sessions.
- **Daily Portfolio Return:** Realized MTM return across all currently active sleeves:
  $$r_{p,t} = w_{\text{active}} \cdot \left(\sum_{s=1}^H \frac{1}{H} r_{\text{sleeve}, s, t}\right) + w_{\text{benchmark}} \cdot r_{\text{bm},t} + w_{\text{cash}} \cdot r_{\text{cash},t} - \text{slippage}_t$$
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
  - `env_val` must execute across all sessions of `cal_val` (`2016-01-04` to `2022-03-24`, 1,568 steps) without step truncation. Early stopping evaluates policy survivability across bull and bear regimes.
- **Temporal Alignment in Evaluation:**
  - Benchmark metrics (`bm_sub_cum`, `excess_daily`, beta, IR) must be computed over the identical trading dates recorded in the agent's blotter, never across disparate calendar lengths.

---

## 3. State, Action & Model Contracts

### Observation Space Dynamic Formula (3K + 10 Dimensions)
- `[0 : K]` Cross-Sectional Mean ($K$ dims): Mean of the $K$ strategy factors across universe.
- `[K : 2K]` Cross-Sectional Std ($K$ dims): Standard deviation of the $K$ strategy factors across universe.
- `[2K : 3K]` Benchmark Factor Vector ($K$ dims): Benchmark's (`config.benchmark`) own scores on the $K$ strategy factors.
- `[3K : 3K+10]` Stationary Macro Vector (10 dims): Yield curve slope, credit spreads, macro velocity, VIX ratios.
*Registry Invariant:* Under locked $K = 12$, State Dimension = $3(12) + 10 = \mathbf{46\text{ dims}}$.

### Generation 16 Strategy Registry Architecture ($K = 12$ Orthogonal Factors)
| Index | Factor Name | Category | Theoretical Function | Mathematical Formulation |
| :--- | :--- | :--- | :--- | :--- |
| **#1** | `Momentum (12-1m)` | Momentum | Secular Trend Anchor | Fama-French $t-252 \to t-21$ cumulative return |
| **#2** | `Residual Momentum (126d)` | Idiosyncratic Alpha | Beta-Decontaminated Alpha | $\sum_{\tau=t-126}^t (r_{i,\tau} - \beta_i r_{\text{bm},\tau}) / (\sigma_\epsilon \sqrt{126})$ |
| **#3** | `Range Position (52w High)` | Anchoring Quality | Overhead Supply Clearance | $P_{i,t} / \max_{\tau \in [t-252, t]} P_{i,\tau} \in (0, 1]$ |
| **#4** | `Info Ratio (63d)` | Alpha Consistency | Alpha Quality / Stability | Quarterly annualized excess return / tracking error |
| **#5** | `Oversold (-RSI)` | Mean Reversion | Contrarian Pullback | Bounded inverse RSI(14), scaled $[-1, 1]$ |
| **#6** | `Dip Buyer (-dd_21)` | Mean Reversion | Tactical Drawdown Depth | Inverse 21-day drawdown from peak |
| **#7** | `Trend Quality (63d)` | Trend Linearity | Structural March ($R^2$) | Pearson correlation $r(t, \ln P_t)$ over 63 days |
| **#8** | `Downside Beta (-Beta_Down_63)`| Asymmetric Tail Defense| Systematic Crash Filter | $-\text{Cov}(r_i, r_{\text{bm}} \mid r_{\text{bm}} < 0) / \text{Var}(r_{\text{bm}} \mid r_{\text{bm}} < 0)$ |
| **#9** | `Low Volatility (-ATRP)` | Volatility Quality | Absolute Noise Dampener | Inverse Average True Range Percentage ($-\text{ATRP}$) |
| **#10** | `Efficiency Ratio (ER_63)` | Fractal Efficiency | Signal-to-Noise Ratio | $|P_t - P_{t-63}| / \sum_{\tau=t-62}^t \|P_\tau - P_{\tau-1}\| \in [0, 1]$ |
| **#11** | `Momentum (126d)` | Momentum | Intermediate Trend Anchor | Semi-annual cumulative return ($t-126 \to t$) |
| **#12** | `Residual Low-Vol (63d)` | Risk-Adjusted | Idiosyncratic Quality Shelter| Inverse CAPM residual volatility ($-\sigma_\epsilon$ over 63d) |

### Action Space Structure ($K + 4 = 16$ Dimensions)
- `[0 : 12]` Alpha Weights: L2-normalized feature scoring weights for the 12 registry factors.
- `[12]` Offset: Cross-sectional rank start percentile mapped to $[0, \text{rank\_max\_offset\_percentile}]$.
- `[13]` Width: Basket size mapped to $[\text{min\_basket\_width}, \text{rank\_max\_width}]$ (locked to min_width = 5).
- `[14]` Equity Exposure: Mapped to $[1.0 - \text{max\_cash\_pct}, 1.0]$ (locked to 1.0 under Zero Cash mandate).
- `[15]` Active Tilt: Blending ratio between active stock basket and benchmark beta shelter $[\text{min\_active\_tilt}, 1.0]$ (optimal floor: 0.10).

### Deterministic Action & Architecture Invariant
- `AbsoluteZeroAgent.forward(x)` and `AbsoluteZeroAgent.get_deterministic_action(x)` MUST apply `torch.clamp(1.05 * torch.tanh(self.actor_mean(x)), -1.0, 1.0)`.
- Deterministic inference must match the Gaussian distribution mean trained by PPO.
- **Actor Output Bias Invariant:** `actor_mean` must initialize with unbiased orthogonal/linear weights (`std=0.01, bias=0.0`). Injecting arbitrary manual factor biases is strictly forbidden.

---

## 4. Mathematical & Accounting Conventions
- **Factor Variance Equalization Rule:** All strategy factor columns must be cross-sectionally standardized:
  $$Z_{i, t} = \text{clip}\left(\frac{f_{i, t} - \mu_{i, t}}{\max(\sigma_{i, t}, 10^{-8})}, -4.0, 4.0\right)$$
  inside `SelectionLogic.apply_action()` immediately prior to matrix multiplication with action weights.
- **Compounding Wealth Curves (Base 1.0):**
  $$V_p(t) = V_p(t-1) \cdot (1 + r_{\text{net},t}), \quad V_{\text{bm}}(t) = V_{\text{bm}}(t-1) \cdot (1 + r_{\text{bm},t})$$
- **Alpha Multiplier (Base 1.0): Strictly Geometric Ratio:**
  $$\text{Alpha Multiplier}_t = \frac{V_p(t)}{V_{\text{bm}}(t)}$$
- **Composite Validation Fitness:**
  Pure benchmark-relative active alpha evaluation in `QuantUtils.compute_composite_fitness`:
  $$\text{Fitness} = \begin{cases} \text{excess\_return} \cdot \max(\text{ir\_floor}, \text{information\_ratio}) & \text{if excess\_return} > 0 \\ \text{excess\_return} \cdot \frac{1.0}{\text{ir\_floor}} & \text{if excess\_return} \le 0 \end{cases}$$

---

## 5. Empirical History & Benchmark Baseline

### Generation 12 Breakthrough: Pure Relative Alpha (Runs 209–217)
- **Frontier Champion (Run 217):**
  - Model: `model_P9_Scale25_Gamma88_LossAv035_Up118_Tilt10_Width5_s42_ep_78.pt`
  - **OOS Return: 87.76% (+6.90% Excess vs SPY +80.2%) | Sharpe: 0.882 | Sortino: 1.239 | Max DD: -20.03% | IR: +0.185 | Beta: 1.00**

### Generation 13: Exploiting the Run 217 Ridge (Runs 218–226)
- **Macro Distributional Shift:** Group Mean Sharpe **0.758 ± 0.103**; Group Mean Return **69.12% ± 14.21%**. Confirmed multi-seed stability (Seed 101 Sharpe 0.850, Return 79.37%).
- **Baseline Control Parameters:** `Scale=25`, `Gamma=0.88`, `LossAv=0.35`, `Up=1.18`, `Tilt=0.10`.

### Generation 15: Risk-Parity Intra-Sleeve Weighting (Runs 236–244)
- **Result:** **REFUTED AND PURGED.** Group Mean Sharpe collapsed to **0.687**; Excess Return fell to **-21.45%** (0 out of 9 models achieved positive excess).
- **Causal Mechanism Autopsy:** Downweighting volatile stocks via $1/\text{ATRP}$ starved momentum compounders during equity expansion and duplicated the $H=5$ staggered sleeve diversification.
- **Action Taken:** Permanently purged. Code reverted strictly to equal-weighted active sleeves ($1/K_t$).

---

## 6. Generation 16 Scientific Contract: Factor Registry Hygiene & De-biasing

### Mechanism & Economic Rationale
The pre-Gen 16 registry contained 4 overlapping momentum horizons (`Mom_21`, `Mom_63`, `Mom_126`, `Mom_252_21`) and 3 overlapping volatility metrics (`ATRP`, `SemiDev`, `IVol`). This high collinearity ($\kappa \approx 45$, VIF $> 10$) allowed PPO to learn synthetic curve-flattening arbitrage (shorting longer momentum vs. shorter momentum) to artificially smooth short-term episodic loss penalties, which degraded secular outperformance. Replacing the 4 dead/collinear factors with orthogonal, beta-decontaminated factors (`Residual Momentum`, `Range Position`, `Downside Beta`, `Efficiency Ratio`) decouples the action space, lowers the condition number ($\kappa < 10$), and forces policy gradients to load onto genuine alpha compounders.

### Baseline Control
- Primary Control Model: Gen 13 Benchmark Run 217 / 221 (`OOS Return 87.76%`, `Sharpe 0.882`, `Excess +6.90%`, `Max DD -20.03%`, `Beta 1.00`).
- Population Baseline: Gen 13 Group Distribution (Sharpe $0.758 \pm 0.103$, Return $69.12\% \pm 14.21\%$).

### Falsifiable Decision Gates (OOS 2022–2026)
1. **Group Mean Sharpe Ratio:** Must exceed **$0.758$** across the 9-run grid sweep.
2. **Distributional Dispersion:** Multi-seed Sharpe standard deviation must contract to **$\le 0.080$** (eliminating seed fragility).
3. **Alpha Generation Capacity:** At least **3 of 9 models** must generate positive OOS Excess Return vs. SPY ($> 0.0\%$).
4. **Beta Preservation:** OOS Market Beta must remain strictly anchored ($\beta \ge 0.98$).