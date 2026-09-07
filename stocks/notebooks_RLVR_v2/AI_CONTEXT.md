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
- `[0:12]` Cross-Sectional Mean (12 dims): Normalized feature means across universe.
- `[12:24]` Cross-Sectional Std (12 dims): Normalized feature standard deviations.
- `[24:36]` Benchmark Vector (12 dims): Trend, volatility, momentum of `config.benchmark`.
- `[36:46]` Stationary Macro Vector (10 dims): Yield curve slope, credit spreads, macro indicators.

### Action Space (16 Dimensions, Tanh bounded [-1, 1])
- `[0:12]` Alpha Weights: L2-normalized feature scoring weights.
- `[12]` Offset: Cross-sectional rank start percentile `[0, rank_max_offset]`.
- `[13]` Width: Basket size mapped to `[min_basket_width, rank_max_width]` (default: 5 to 10 stocks).
- `[14]` Equity Exposure: Mapped to `[1.0 - max_cash_pct, 1.0]` (locked to 1.0 under Zero Cash mandate).
- `[15]` Active Tilt: Blending ratio between active stock basket and benchmark beta shelter `[min_active_tilt, 1.0]` (floor: 0.20 to 0.35).

---

## 4. Mathematical & Accounting Conventions
- **Pure Math Separation:** NumPy vector reductions and compounding logic reside strictly in `core/quant.py` (`QuantUtils`). DataFrame slicing and FIFO sleeve tracking reside in `core/accounting.py` (`MTMPortfolioEngine`).
- **Stateless Accounting Boundaries:**
  - `MTMPortfolioEngine.extract_sleeve_return_array` is `@staticmethod`.
  - `MTMPortfolioEngine.calculate_sleeve_simple_ret` is `@staticmethod`. Can be called directly on class or instance without state side-effects.
- **Compounding Wealth Curves (Base 1.0):**
  $$V_p(t) = V_p(t-1) \cdot (1 + r_{\text{net},t})$$
  $$V_{\text{bm}}(t) = V_{\text{bm}}(t-1) \cdot (1 + r_{\text{bm},t})$$
- **Alpha Multiplier (Base 1.0): Strictly Geometric Ratio:**
  $$\text{Alpha Multiplier}_t = \frac{V_p(t)}{V_{\text{bm}}(t)}$$
- **Banned Math:** Compounding $(1 + \text{reward})$ or $(1 + r_p - r_{\text{bm}})$ is strictly prohibited. Additive compounding causes geometric divergence and artificial equity distortion. All legacy additive fallbacks must be rejected.
- **RL Step Reward:** Linear spread $r_t = r_{p,t} - r_{\text{bm},t}$ when `upside_alpha_mult = 1.0` and `loss_aversion_penalty = 0.0`.

---

## 5. Artifacts, Serialization & Checkpoint Contracts
- **Checkpoint Path:** `output/model_checkpoints/model_{ID}_s{seed}_ep_{epoch}.pt`
- **Catalog Registry:** Single source of truth catalog at `output/model_catalog.parquet`.
- **Leaderboard Log:** `Grid_sweep.csv` records all OOS evaluated checkpoints with strict deduplication on model stem.
- **Loading Convention:** Strictly query `payload["grid_params"]` via `torch.load(path, map_location="cpu", weights_only=False)`. Filename regex parsing is strictly a fallback.
- **Required Artifact Outputs per Model:**
  1. `output/model_checkpoints/model_{STEM}.pt`
  2. `output/blotter_df_{STEM}.parquet` (Strict geometric alpha_equity)
  3. `output/results_{STEM}.pkl` (Full telemetry dictionary: equity curves, metrics, blotter records)
- **Known Issue Under Active Remediation:**  
  `03a_RLVR_Auto_Analysis_v21.ipynb` currently processes only 5 of 9 available checkpoints and omits saving `results_*.pkl`. Auto-discovery filter and serialization pipeline require alignment.

---

## 6. Verification & Tripwire Invariants
- **Gate:** `pytest tests/test_mtm_invariants.py tests/test_accounting_provenance.py tests/test_rl_metrics.py`
- **Dynamic Config:** Never hardcode `'SPY'` or fixed holding periods. Always consume `config.benchmark` and `config.holding_period`.
- **Purge Gap:** An $H$-day purge gap must be enforced before train/val and val/test split boundaries.