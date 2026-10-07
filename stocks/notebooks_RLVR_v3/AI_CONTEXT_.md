# AI_CONTEXT.md: Production System State & Quantitative Ledger

## 1. Quantitative Architecture & Domain Model

- **Model Framing:** Overlapping Mark-to-Market (MTM) Multi-Sleeve Portfolio via Proximal Policy Optimization (PPO).
- **Staggered Sleeves:** Capital is divided into $H$ equal portions (Weight per sleeve = $1/H$, where $H = \text{config.holding\_period}$).
- **Intra-Sleeve Weighting:** Assets within each active sleeve are strictly equally weighted ($1/K_t$). List ordering is irrelevant; evaluate baskets via set equality or Jaccard similarity.
- **Execution Convention (T+1 Close):** Eliminates lookahead bias.
  - Day $T$ (Close): Feature observation up to $T$. Policy outputs action $A_T$.
  - Day $T \to T+1$: Pending order state. Zero PnL exposure.
  - Day $T+1$ (Close): Sleeve executes at $P_{T+1}$ (`buy_date`). Sleeve becomes active.
  - Day $T+1 \to T+H+1$: MTM active holding for $H$ consecutive market sessions.
  - Day $T+H+1$ (Close): Sleeve liquidates at $P_{T+H+1}$ (`sell_date`). Drops out of sleeve queue.
- **Vectorized Return Indexing (from row $T$ perspective):**
  - Holding Day 1 ($T+1 \to T+2$): `ret_1d.shift(-2)`
  - Holding Day $k$ ($T+k \to T+k+1$): `ret_1d.shift(-(k+1))`
  - Holding Day $H$ ($T+H \to T+H+1$): `ret_1d.shift(-(H+1))`
- **Exposure Mandates:** Gross Exposure = 1.0 (Zero cash hoarding, `max_cash_pct = 0.0`). Downside risk management is governed via dynamic benchmark beta hedging (`min_active_tilt = 0.20`), not cash un-investment.
- **Dynamic Config Invariant:** NEVER hardcode `'SPY'` or holding periods in code, tests, or notebooks. Reference dynamic fields `config.benchmark` and `config.holding_period`.

---

## 2. Determinism & Bit-Level Reproducibility Contract

- **Training Invariant:** Training the identical seed $S$ on identical feature caches must produce bit-for-bit identical network weights across separate process runs: $\text{SHA-256}(W_{\text{run1}}) \equiv \text{SHA-256}(W_{\text{run2}})$.
- **Entropy Leak Defense (4 Systemic Gates):**
  1. **Environment Seeding (`rl_discovery/environment.py`):** `DiscoveryEnv.__init__` must accept `seed: Optional[int] = None` and instantiate `self.rng = random.Random(seed)` before calling `self.reset()`. `reset()` must never fall back to unseeded system entropy.
  2. **Rank-Isolated Vector Envs (`rl_discovery/adapter.py`):** `make_stratified_train_envs` must accept `seed: Optional[int] = None` and bind each worker rank with isolated deterministic seeds:
     $$\text{worker\_seed}_i = \text{seed} + i \times 1000$$
  3. **Isolated Batch Shuffling (`rl_discovery/trainer.py`):** `PPOTrainer` must instantiate its own isolated `self.rng = np.random.default_rng(seed)`. Global `np.random.permutation` or `np.random.shuffle` is strictly forbidden.
  4. **CUDA Determinism (`run_walk_forward.py`):** `set_seed(seed)` must enforce:
     - `os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"`
     - `os.environ["PYTHONHASHSEED"] = str(seed)`
     - `torch.use_deterministic_algorithms(True, warn_only=True)`
     - `torch.backends.cudnn.deterministic = True`
     - `torch.backends.cudnn.benchmark = False`

---

## 3. Artifact Contracts & Functional Roles

1. **`output/canonical_anchors/` (Ground-Truth Vault):**
   - **Retention Contract:** Immutable vault, read-only. Never overwritten by in-flight training loops.
   - **Constituent Continuous Blotters (`blotter_continuous_s{seed}_gen{G}.parquet`, ~767 KB):** Full 31-column execution telemetry:
     `['date', 'decision_date', 'buy_date', 'sell_date', 'universe_size', 'max_score', 'min_score', 'observation', 'raw_actions', 'offset', 'width', 'top_3', 'tickers', 'weight_active', 'weight_benchmark', 'weight_cash', 'equity_exposure', 'active_tilt', 'predicted_reward', 'rl_reward_received', 'gross_stock_daily_simple_ret', 'bm_daily_simple_ret', 'cash_daily_simple_ret', 'slippage_daily_simple_loss', 'gross_daily_simple_ret', 'net_daily_simple_ret', 'alpha_daily_simple_ret', 'penalized_alpha_daily_simple_ret', 'agent_equity', 'bm_equity', 'alpha_equity']`.
     - `raw_actions`: NumPy `ndarray` of shape `(16,)` (12 factor weights + 4 portfolio controls).
     - `observation`: NumPy `ndarray` of shape `(46,)` (12 factor means + 12 factor stds + 12 benchmark features + 10 macro features).
     - `tickers`, `top_3`: NumPy `ndarray` of string ticker symbols.
   - **`run_metadata_gen{G}.json` (Artifact Lineage Contract):**
     - Mandatory manifest mirroring training configuration, data lineage, and environment flags.
     - **Lineage Binding Mandate:** MUST serialize `"feature_cache"` (e.g. `"alpha_cache_41d_2015.parquet"`).
   - **`canonical_scoring_slice.parquet` (Hermetic Provenance Anchor):**
     - Lightweight cross-sectional slice of audit sessions (`["2022-04-07", "2026-07-09"]`) extracting features from the Gen 18 baseline cache.

2. **Feature Cache & Market Data Lineage (`data/`):**
   - **`data/df_ohlcv.parquet` (8,576,706 rows):** Institutional market database with `MultiIndex(['Ticker', 'Date'])`. Single-ticker extraction MUST use zero-copy cross-sectional slicing (`df.xs(key, level='Ticker')`).
   - **`data/alpha_cache_41d_2015.parquet`:** Authoritative feature cache for Gen 25 ($N=1024$ on `2026-09-23`). Anchored to 2015 origin date.
   - **`data/alpha_cache_41d_1998.parquet`:** Restated live cache anchored to 1998 origin date ($N=981$ on `2026-09-23`).

3. **`output/model_checkpoints/walk_forward_gen{G}/`:**
   - **Multi-Chunk Walk-Forward Topology:** 5 discrete chunk models per seed (`model_chunk{0..4}_s{seed}_champion.pt`), mapping to chronological walk-forward out-of-sample folds $[0..4]$.
   - Payload: `model_state_dict`, `scaler_state`, `val_fitness`, `val_sharpe`, `val_excess`, `val_ir`, `epoch`, `seed`, `chunk_id`, `grid_params`, `holding_period`, `benchmark`.

---

## 4. Two-Tier Verification Gate Contract

- **Tier 1 (Bit-Level Identity):**
  - Requires matching baseline feature cache lineage (`alpha_cache_41d_2015.parquet`) or `canonical_scoring_slice.parquet`.
  - $\Delta w_{\text{active}} < 10^{-6}$.
  - 0 basket session mismatches; exact top-3 identity; integer offset identity ($N_{\text{live}} \equiv N_{\text{blotter}}$).
  - Scores match bare-metal dot products ($Z \cdot w_{\text{norm}}$) to $\text{atol} = 10^{-8}$.
  - Observation scaler verification achieves IEEE 754 float32 machine epsilon: $\text{atol} \le 1.192 \times 10^{-7}$.

- **Tier 2 (Corporate Action & Restatement Invariants):**
  - Triggered when evaluating against a restated live cache where $N_{\text{current}} \ne N_{\text{blotter}}$.
  - Maximum universe drift: $|\Delta N| / N \le 5.0\%$.
  - Basket width invariance: `width == row["width"]` (strictly equal).
  - Financial accounting returns, slippage, and equity curves maintain $\text{atol} \le 10^{-6}$.

---

## 5. Generation History & Empirical Ledger

- **Gen 25 Audit (Authoritative Production Baseline Control):**
  - Lineage: `alpha_cache_41d_2015.parquet` (1,120 sessions: `2022-03-25` to `2026-09-25`).
  - Seeds: `[42, 101, 777]`.
  - Ex-post blend: **+124.71%** return (**+49.34%** excess vs SPY), Sharpe **0.8301**, Max DD **-26.00%**, Beta **1.3040**, Diversification Ratio **1.0552**.
  - **Divergence Finding (Gen 25 vs 26 vs 27 vs 28):** In-depth audit proved runs diverged due to unseeded environment start dates in Chunk 0 (`DiscoveryEnv.reset()` using OS clock entropy). All downstream generation results prior to the determinism patch are non-deterministic paths of Chunk 0.