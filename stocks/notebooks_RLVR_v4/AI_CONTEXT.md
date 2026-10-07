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

## 2. Artifact Contracts & Functional Roles

1. **`output/canonical_anchors/` (Ground-Truth Vault):**
   - **Retention Contract:** Immutable vault, read-only. Never overwritten by in-flight training loops.
   - **Constituent Continuous Blotters (`blotter_continuous_s{seed}_gen{G}.parquet`, ~767 KB):** Full 31-column execution telemetry:
     `['date', 'decision_date', 'buy_date', 'sell_date', 'universe_size', 'max_score', 'min_score', 'observation', 'raw_actions', 'offset', 'width', 'top_3', 'tickers', 'weight_active', 'weight_benchmark', 'weight_cash', 'equity_exposure', 'active_tilt', 'predicted_reward', 'rl_reward_received', 'gross_stock_daily_simple_ret', 'bm_daily_simple_ret', 'cash_daily_simple_ret', 'slippage_daily_simple_loss', 'gross_daily_simple_ret', 'net_daily_simple_ret', 'alpha_daily_simple_ret', 'penalized_alpha_daily_simple_ret', 'agent_equity', 'bm_equity', 'alpha_equity']`.
     - `raw_actions`: NumPy `ndarray` of shape `(16,)` (12 factor weights + 4 portfolio controls).
     - `observation`: NumPy `ndarray` of shape `(46,)` (12 factor means + 12 factor stds + 12 benchmark features + 10 macro features).
     - `tickers`, `top_3`: NumPy `ndarray` of string ticker symbols.
   - **`run_metadata_gen{G}.json` (Artifact Lineage Contract):**
     - Mandatory manifest mirroring training configuration, data lineage, and environment flags.
     - **Lineage Binding Mandate:** MUST serialize `"feature_cache"` (e.g. `"alpha_cache_41d_2015.parquet"`). Downstream blotter verification engines dynamically resolve the exact cache lineage via this field rather than global environment defaults.
   - **`canonical_scoring_slice.parquet` (Hermetic Provenance Anchor):**
     - Lightweight (~190 KB, 1,984 rows) cross-sectional slice of audit sessions (`["2022-04-07", "2026-07-09"]`) extracted from the Gen 18 baseline cache. Decouples CI/CD unit testing from living cache mutations in `data/`.

2. **Feature Cache & Market Data Lineage (`data/`):**
   - **`data/df_ohlcv.parquet` (8,576,706 rows):** Institutional market database with `MultiIndex(['Ticker', 'Date'])`. Single-ticker extraction MUST use zero-copy cross-sectional slicing (`df.xs(key, level='Ticker')`).
   - **`data/alpha_cache_41d_2015.parquet`:** Authoritative feature cache for Gen 25 ($N=1024$ on `2026-09-23`). Lookback features and expanding percentiles are anchored to a 2015 origin date.
   - **`data/alpha_cache_41d_1998.parquet` (453.20 MB):** Restated live cache anchored to 1998 origin date. Pruned 43 delisted equities ($N=981$ on `2026-09-23`).

3. **`output/model_checkpoints/walk_forward_gen{G}/`:**
   - **Multi-Chunk Walk-Forward Topology:** 
     - **Gen 18:** 4 discrete chunk models per seed (`model_chunk{0..3}_s{seed}_champion.pt`).
     - **Gen 25+:** 5 discrete chunk models per seed (`model_chunk{0..4}_s{seed}_champion.pt`), mapping to chronological walk-forward out-of-sample folds $[0..4]$.
   - Monolithic dictionary payload:
     `model_state_dict`, `scaler_state`, `val_fitness`, `val_sharpe`, `val_excess`, `val_ir`, `epoch`, `seed`, `chunk_id`, `grid_params`, `holding_period`, `benchmark`.
   - **Chunk-Observation Binding Invariant:** Because each chunk trains on a rolling historical window, `scaler_state` (`mean`, `var`, `count`) evolves across chunks. Terminal blotter rows ($T_{\text{last}}$, `sample_idx = -1`, index $1119/1120$) are generated by the final fold: **Chunk 4** for Gen 25+, not Chunk 0 or Chunk 3. Observation verification dynamically binds to the active fold model.

---

## 3. Two-Tier Verification Gate Contract

- **Tier 1 (Bit-Level Identity):**
  - Requires matching baseline feature cache lineage (`run_metadata.json` $\to$ `alpha_cache_41d_2015.parquet`) or `canonical_scoring_slice.parquet`.
  - $\Delta w_{\text{active}} < 10^{-6}$.
  - 0 basket session mismatches; exact top-3 identity; integer offset identity ($N_{\text{live}} \equiv N_{\text{blotter}}$).
  - Scores match bare-metal dot products ($Z \cdot w_{\text{norm}}$) to $\text{atol} = 10^{-8}$.
  - Observation scaler verification achieves IEEE 754 float32 machine epsilon: $\text{atol} \le 1.192 \times 10^{-7}$.
  - **Empirical Ground-Truth Proof (Gen 25 Terminal Audit on 2015 Cache):**
    - Date: `2026-09-23`, $T = 1119 / 1120$.
    - Checkpoint: `model_chunk4_s777_champion.pt`.
    - Cache: `alpha_cache_41d_2015.parquet`.
    - Universe: $1024 \equiv 1024$; Offset: $332 \equiv 332$; Width: $6.0 \equiv 6.0$.
    - Max Observation Vector Error across all 46 dimensions: **$1.19209290 \times 10^{-7}$ (Exact Tier 1 Bit-Level Identity)**.

- **Tier 2 (Corporate Action & Restatement Invariants):**
  - Triggered when evaluating against a restated live cache where $N_{\text{current}} \ne N_{\text{blotter}}$ (e.g., replaying Gen 25 on the pruned 1998 cache).
  - Maximum universe drift: $|\Delta N| / N \le 5.0\%$.
  - Proportional rank offset invariance:
    $$\left|\frac{\text{offset}_{\text{replayed}}}{N_{\text{current}}} - \frac{\text{offset}_{\text{blotter}}}{N_{\text{blotter}}}\right| \le \frac{1.0}{\min(N_{\text{current}}, N_{\text{blotter}})}$$
  - Basket width invariance: `width == row["width"]` (strictly equal, independent of universe restatement).
  - Financial accounting returns, slippage, and equity curves maintain $\text{atol} \le 10^{-6}$.

--- 

## 4. Test Suite Map & Ground-Truth Alignment

- **`tests/test_accounting_provenance.py`:** PASSING. Validates daily compounding, slippage deduction, and net equity math against `output/canonical_anchors/blotter_continuous_*.parquet`.
- **`tests/test_scoring_provenance.py`:** PASSING. Validates bare-metal NumPy factor equalization ($Z \in [-4, 4]$) and `SelectionLogic.apply_action` against `output/canonical_anchors/canonical_scoring_slice.parquet`.
- **`tests/test_verify_oos_returns.py`:** PASSING (Zero Skips). 
  - Resolves dynamic `config.benchmark` against `data/df_ohlcv.parquet` using high-performance $O(1)$ MultiIndex `.xs()` slices.
  - Reconciles Next-Day Close ($T+1$) execution session conventions: maps decision date $T$ to execution session (`buy_date`), eliminating false lag divergences down to $\text{atol} < 10^{-6}$.
  - Enforces institutional discrete geometric compounding ($E_t / E_{t-1} - 1 = r_{\text{net}, t}$) and geometric wealth ratios ($E_{\alpha, t} = E_{\text{agent}, t} / E_{\text{bm}, t}$).

---

## 5. Generation History & Empirical Ledger

- **Gen 25 Audit (Authoritative Production Baseline Control - Promoted):**
  - Data Lineage: `alpha_cache_41d_2015.parquet` (1,120 deployment sessions).
  - Walk-Forward Topology: 5 discrete chunks (`chunk0` through `chunk4`). Active terminal fold: `chunk4`.
  - Constituent seeds: `[42, 101, 777]`.
  - Ex-post blend: **+149.34%** agent cumulative return (**+49.34%** excess return vs benchmark), Sharpe **0.8301**, Max DD **-26.00%**, Beta **1.3040**, Diversification Ratio **1.0552**.
  - Provenance Audit: Verified bit-level identity ($\text{atol} \le 1.192 \times 10^{-7}$) on terminal session (`2026-09-23`) via `model_chunk4_s777_champion.pt`.
  - **Verdict: PERMANENT PRODUCTION BENCHMARK (Retires Gen 24).**
