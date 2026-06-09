User: "RLVR_Part1_AlphaCache.ipynb" explain what this file is doing step-by-step as i am new to RL
Kimi: Here is a step-by-step explanation of what `RLVR_Part1_AlphaCache.ipynb` does, written for someone new to Reinforcement Learning (RL).

---

### **The Big Picture (Why this notebook exists)**

In RL, an **agent** learns to make decisions by repeatedly observing the **environment**, taking an **action**, and receiving a **reward**. In this trading project:

*   **Agent:** The AI that decides whether to buy, sell, or hold.
*   **Environment:** The stock market.
*   **Observation (State):** The information the agent is allowed to see before making a decision (e.g., recent prices, volatility, volume).

This notebook is **Part 1: The Data Preparation Phase**. Before the AI can learn, you must convert messy raw market data into a clean, structured "observation cube" (called the **AlphaCache**). Think of it as creating the "flashcards" that the agent will study from in Part 2.

---

### **Step-by-Step Breakdown**

#### **Cell 1: Introduction**
This is just documentation. It tells you that this notebook is CPU-optimized (no GPU needed yet) and will build the **Feature Cube**, saving it to disk so the training notebook (Part 2) can load it later.

#### **Cell 2: Environment & Path Setup**
```python
RUNNING_IN_COLAB = True/False
```
*   **What it does:** Sets up file paths. It detects if you are running in Google Colab (cloud) or a local Windows PC, mounts Google Drive if needed, and tells Python where to find the project folders (`core`, `rl_discovery`, etc.).
*   **Why it matters:** The code needs to import custom modules written by the project author. This step ensures Python can "see" those folders.

#### **Cell 3: Load Raw Data into Memory**
```python
df_ohlcv = pd.read_parquet(...)
features_df = pd.read_parquet(...)
macro_df = pd.read_parquet(...)
```
*   **What it does:** Loads three datasets from disk into RAM:
    1.  **OHLCV Data:** Historical stock prices (Open, High, Low, Close, Volume) for many tickers.
    2.  **Features Data:** Pre-calculated technical indicators (e.g., moving averages, RSI).
    3.  **Macro Data:** Broad economic data (e.g., VIX, interest rates, yield spreads).
*   **Why it matters:** RL agents don't learn from one number at a time. They need a rich history of market conditions. This is the raw "ingredient list."

#### **Cell 4: Import Project Modules**
```python
from core.settings import TradingConfig, CacheConfig
from data_pipeline.screener import UniverseScreener
from data_pipeline.cache import AlphaCache
```
*   **What it does:** Imports the project's custom Python classes.
*   **Key concepts for beginners:**
    *   `TradingConfig`: A central "settings file" (how many stocks to consider, dates, etc.).
    *   `UniverseScreener`: A **filter** that decides which stocks are "tradeable" on any given day (e.g., removing stocks with missing data or delisted tickers).
    *   `AlphaCache`: The builder that creates the multi-day lookback feature cube.

#### **Cell 5: Building the AlphaCache (The Core Work)**
This is the most important cell. Let's break it into sub-steps:

**A. Reshape the Price Data**
```python
df_close = df_ohlcv["Adj Close"].unstack(level=0)
```
*   Raw OHLCV data is usually "long format" (one row per stock per day). This converts it to **"wide format"**: a matrix where **rows = dates** and **columns = stock tickers** (e.g., AAPL, TSLA).
*   *Analogy:* It's like converting a vertical list into a spreadsheet where each column is a different stock.

**B. Create a Clean Trading Calendar**
```python
trading_calendar = pd.DatetimeIndex(df_close.dropna(how="all").index.sort_values())
```
*   Creates a sorted list of all valid trading days, ignoring dates where the entire market was closed.

**C. Initialize the Universe Screener**
```python
screener = UniverseScreener(...)
```
*   This acts as the **"gatekeeper."** On any given day, it decides which stocks are eligible to be traded based on data quality, liquidity, or other rules defined in `TradingConfig`.
*   *RL Concept:* The agent cannot trade every stock in existence. The screener defines the valid **action space** (which stocks the agent is allowed to pick from each day).

**D. Instantiate and Build the AlphaCache**
```python
cache = AlphaCache(screener=screener, config=config, lookbacks=[CacheConfig.LOOKBACK])
cache.build(start_date=train_start)
```
*   **What is a Lookback?** In RL, the agent shouldn't only see "today's price." It needs context: "What happened over the last N days?" The `lookback` (e.g., 189 days) defines how much history is included in each observation.
*   **What is the Feature Cube?** It is a 3D data structure:
    *   **Dimension 1:** Time (trading days)
    *   **Dimension 2:** Stocks (tickers in the universe)
    *   **Dimension 3:** Features (price, volume, technical indicators, macro data)
*   *RL Concept:* This cube represents the **State Space**. Every "slice" of this cube for a specific `(date, stock)` pair becomes the **observation vector** that the neural network will look at before deciding to buy or sell.

**E. Save to Disk**
```python
cache.feature_cube.to_parquet(cache_file_path)
```
*   The final structured cube is saved as a `.parquet` file (a fast, compressed format).
*   *Why?* Building this cube is computationally expensive. By saving it now, the training notebook (Part 2) can simply **load** it instantly instead of recalculating it every time you tweak the AI model.

---

### **Summary: What did we just create?**

| Raw Input | Process | Output |
|-----------|---------|--------|
| Messy price tables, indicators, macro data | Filtered, aligned, and reshaped by the Screener and Cache builder | A clean 3D **AlphaCache** (Feature Cube) |

**For your RL learning:** Understand that this notebook does **not** do any machine learning yet. It builds the **environment's observation space**. In Part 2, the RL agent will read from this cube, treating each slice as its "eyes" on the market, and learn to trade based on what it sees.
User: do the same for "RLVR_Part2_Training.ipynb"
Kimi: Here is a step-by-step explanation of `RLVR_Part2_Training.ipynb`, written for someone new to Reinforcement Learning.

---

### **The Big Picture (Why this notebook exists)**

In Part 1, you built the **AlphaCache**—a clean "observation cube" of market data. This notebook is **Part 2: The Learning Phase**.

If you think of RL like teaching a dog a trick:
*   **Part 1** built the training ground and defined what the dog is allowed to see.
*   **Part 2** is where the dog actually tries the trick, gets a treat (or a scolding), and slowly learns what works.

Specifically, this notebook:
1.  **Reloads** the AlphaCache.
2.  **Creates a "Reward Oracle"** that tells the agent if its decisions were good or bad.
3.  **Builds a virtual trading environment** (the "gym").
4.  **Trains a neural network** using an algorithm called **PPO** (Proximal Policy Optimization).
5.  **Tests** the trained agent on unseen data to see if it actually learned or just memorized.

---

### **Step-by-Step Breakdown**

#### **Cell 1: Introduction**
This explains that this notebook is GPU-accelerated (uses a graphics card for heavy math) and handles the actual training and validation of the RL agent.

#### **Cell 2: Environment & Path Setup**
Identical to Part 1. It mounts Google Drive (if in Colab) and ensures Python can find the project files. No RL concepts here—just logistics.

#### **Cell 3: Load Raw Data**
Loads the same three raw datasets (OHLCV, features, macro data) as Part 1. These are needed because the training environment needs to map predictions back to real prices and dates, even though the heavy feature engineering was already done in Part 1.

#### **Cell 4: Imports**
This cell brings in the RL machinery. Here are the key pieces in plain English:

| Import | What it is | Analogy |
|--------|-----------|---------|
| `RLOracle` | The "judge" that calculates profit/loss | A teacher grading your homework |
| `DiscoveryEnv` | The raw trading logic (dates, positions, returns) | The actual stock market simulator |
| `RLVRGymEnv` | A wrapper that formats the simulator for standard RL libraries | A translator making the simulator speak "gym language" |
| `AbsoluteZeroAgent` | The neural network (the AI "brain") | The student trying to learn |
| `RolloutBuffer` | Temporary memory of recent experiences | A notepad holding the last 64 actions taken |
| `PPOTrainer` | The learning algorithm | The study method / backpropagation logic |
| `AgentEvaluator` | The final exam grader | A test that checks performance on new data |

#### **Cell 5: Reload AlphaCache & Bootstrap the Environment**

This is the "setup before the game starts" cell. It has several sub-steps:

**A. Rebuild the Screener**
Recreates the `UniverseScreener` from Part 1. The training notebook needs to know which stocks were valid on which days.

**B. Load the Feature Cube**
```python
feature_cube = pd.read_parquet(cache_file_path)
```
Loads the 3D AlphaCache you built in Part 1. This is the agent's "eyes."

**C. The AlphaCacheStub**
Because Part 1 saved only the raw cube (not the full Python object), this creates a lightweight "fake" object that behaves like the original `AlphaCache` so downstream code can use it without complaining.

**D. Create the Reward Oracle**
```python
oracle = RLOracle(screener, config)
oracle.precompute_reward_matrix(holding_period=5)
```
*   **What it does:** Pre-calculates the **reward** for every possible decision the agent could have made on every day. It assumes a holding period of 5 days.
*   **RL Concept:** In RL, the agent doesn't know if a trade was good until later. The Oracle looks at the future (from the historical data's perspective) and says: *"If you bought Stock X on Monday and held for 5 days, your return would be +2.1%."*
*   **Analogy:** The Oracle is an answer key. It peeks at the future of the historical data so it can grade the agent immediately during training.

**E. Train / Test Split**
```python
cal_train = trading_calendar[...CacheConfig.START_DATE...CacheConfig.END_DATE]
cal_test = trading_calendar[...CacheConfig.END_DATE...]
```
*   Splits time into two eras: a **training period** (the past) and a **test period** (the future).
*   **RL/ML Concept:** You **must** test on data the agent has never seen. If you test on training data, the agent might have just memorized patterns (overfitting) rather than learning true strategy.

**F. Initialize Environments**
```python
env_train = DiscoveryEnv(...)
gym_train = RLVRGymEnv(env_train, macro_df)

env_test = DiscoveryEnv(...)
gym_test = RLVRGymEnv(env_test, macro_df)
```
*   `DiscoveryEnv`: The core market simulator. It tracks dates, portfolio value, and uses the Oracle's reward matrix to hand out rewards.
*   `RLVRGymEnv`: A wrapper that converts the custom simulator into the standard "Gym" format (`reset()`, `step()`, `observation`, `reward`, `done`). This standard format is what the PPO algorithm expects.

---

#### **Cell 6: PPO Agent Training (The Core Learning Loop)**

This is where the magic happens. Let's break it down carefully.

**A. Hardware Setup**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
Chooses the GPU if available; otherwise uses the CPU.

**B. Initialize the Agent and Trainer**
```python
agent = AbsoluteZeroAgent(obs_dim=33, action_dim=13).to(device)
trainer = PPOTrainer(agent, lr=3e-4, clip_coef=0.2, ent_coef=0.01)
```
*   **Agent:** A neural network with two "heads":
    1.  **Actor:** Decides *which action to take* (which stock to trade).
    2.  **Critic:** Estimates *how good the current situation is* (predicts future value).
*   **Observation Dim (33):** The agent sees 33 numbers at once (features from the AlphaCache).
*   **Action Dim (13):** The agent can choose among 13 possible actions (e.g., buy different stocks, hold cash, etc.).
*   **PPOTrainer:** Implements the PPO algorithm. It updates the neural network weights to favor actions that led to high rewards, but it uses a "clip" mechanism to prevent the network from changing too radically in one step (which causes instability).

**C. The Training Loop (50 Epochs)**
This loop repeats 50 times. Each epoch is one round of practice.

**Phase 1: Rollout (Experience Collection)**
```python
for step in range(num_steps):  # num_steps = 64
    action, logprob, _, value = agent.get_action_and_value(obs)
    next_obs, reward, done, _, info = gym_train.step(action)
    buffer.add(obs, action, logprob, reward, value, done)
```
*   The agent looks at the current market state (`obs`).
*   The **Actor** picks an action probabilistically (it might pick good actions more often, but still explores).
*   The environment executes the trade and returns:
    *   `next_obs`: The new market state.
    *   `reward`: The profit/loss from the Oracle.
    *   `done`: Whether the episode ended (e.g., last trading day).
*   All of this is stored in the `RolloutBuffer` (a memory of 64 recent steps).
*   **NaN Guard:** If the observation contains invalid numbers, the code crashes immediately to protect the math.

**Phase 2: Advantage Calculation (GAE)**
```python
buffer.compute_advantages(next_value, next_done=done)
```
*   **What is Advantage?** It answers: *"Was this action better or worse than expected?"*
    *   If the Critic predicted a +1% return, but the action actually yielded +3%, the **advantage is +2%**. The action was better than expected.
    *   If it yielded -1%, the advantage is negative. The action was worse than expected.
*   **GAE (Generalized Advantage Estimation):** A smart way of calculating this that balances immediate rewards against long-term future rewards.

**Phase 3: Update (Learning)**
```python
trainer.update(buffer, update_epochs=4, mini_batch_size=16)
```
*   The PPO algorithm looks at the 64 stored experiences.
*   It increases the probability of actions with positive advantage (good moves).
*   It decreases the probability of actions with negative advantage (bad moves).
*   The `clip_coef=0.2` prevents it from over-optimizing one specific action too aggressively.
*   `ent_coef=0.01` adds an "entropy bonus" to encourage exploration—so the agent doesn't get stuck doing the exact same thing every time.

**Phase 4: Logging & Checkpoints**
Every 10 epochs, it prints the average reward and saves the neural network weights to disk. This is like saving your video game progress so you don't lose it if Colab disconnects.

**Final Saves**
After all 50 epochs:
*   Saves a CSV of training metrics (reward over time).
*   Saves the final trained model weights (`absolute_zero_ppo_agent_v1.pt`).

---

#### **Cell 7: Out-of-Sample (OOS) Walk-Forward Evaluation**

Now for the final exam. The agent is tested on the **test period** (data it never saw during training).

**A. Deterministic Evaluation**
```python
results = AgentEvaluator.evaluate(agent, gym_test, device)
```
*   The agent acts **greedily** (no randomness). It picks what it truly thinks is the best action.
*   It walks through the unseen test dates one by one, trading and accumulating returns.

**B. Performance Metrics**
```python
Total Cumulative Return: 31.29%
Sharpe Ratio (Annualized): 0.583
Total Trading Steps: 120
```
*   **Cumulative Return:** Total profit/loss over the test period.
*   **Sharpe Ratio:** A risk-adjusted score. Higher is better (above 1.0 is generally considered good; 0.583 is modest but positive).
*   **Trading Steps:** How many days the agent actively traded.

**C. Visualization**
It builds an interactive Plotly chart showing the agent's equity curve (portfolio value over time) during the test period.

**D. File Outputs**
*   `oos_evaluation_results.pkl`: Raw results dictionary.
*   `oos_equity_curve.csv`: The profit curve as a spreadsheet.
*   `oos_performance_chart.html`: An interactive chart you can open in a browser.

---

### **Summary: What did we just do?**

| Step | RL Concept | Plain English |
|------|-----------|---------------|
| 1 | **State Space** | Loaded the AlphaCache (the agent's "eyes") |
| 2 | **Reward Function** | Built the Oracle (the "grader" using future historical returns) |
| 3 | **Environment** | Created train/test simulators (the "virtual stock market") |
| 4 | **Agent** | Initialized the Actor-Critic neural network (the "brain") |
| 5 | **Policy** | Trained with PPO for 50 epochs (practice → grade → improve) |
| 6 | **Evaluation** | Tested on unseen data (the "final exam") |
| 7 | **Metrics** | Measured return, Sharpe ratio, and plotted the equity curve |

**Key Takeaway for RL Beginners:** The agent does not "understand" finance. It is a pattern-matching machine that learned, through trial and error, which combinations of the 33 input features tend to lead to positive rewards over a 5-day holding period. The test set tells you whether that pattern was real or just a fluke of the training data.
User: where is this path ===cache_file_path===
Kimi: The path is constructed in the code and its final resolved value is shown in the execution output.

### How the path is built in code

In both notebooks, `cache_file_path` is assembled using Python's `pathlib`:

```python
cache_file_path = project_path / CacheConfig.get_filename()
```

This means:

| Component | Value (from the uploaded notebook) |
|-----------|-----------------------------------|
| `project_path` | `/content/drive/MyDrive` (because `RUNNING_IN_COLAB = True`) |
| `CacheConfig.get_filename()` | A filename string generated by the `CacheConfig` class in `core/settings.py` |

### The actual resolved path

From the **execution output** in Part 2, you can see exactly what filename `CacheConfig.get_filename()` produced:

```
[RELOADED] AlphaCache features loaded successfully from /content/drive/MyDrive/alpha_cache_189d_2021_2023.parquet
```

So the full path is:

**`/content/drive/MyDrive/alpha_cache_189d_2021_2023.parquet`**

### Decoding the filename

Based on the naming convention (`alpha_cache_189d_2021_2023.parquet`), the `CacheConfig` class likely auto-generates the filename from constants like:
*   `189d` → the lookback window (189 days)
*   `2021_2023` → the training date range
*   `.parquet` → the compressed columnar storage format

If you were running this locally on Windows (`RUNNING_IN_COLAB = False`), the same code would resolve to something like:

`C:/Users/ping/Files_win10/python/py311/stocks/alpha_cache_189d_2021_2023.parquet`
User: pinpoint code changes to put all outputs to ``output_path = codebase_path / "output" (i.e. put all outputs in codbase_path's output dir)
Kimi: Here are the exact code changes needed in both notebooks. In every case, replace `project_path` with `output_path = codebase_path / "output"` (and ensure the directory exists).

---

### **Part 1: `RLVR_Part1_AlphaCache.ipynb`**

#### **Cell 5 — AlphaCache Save Location**

**Before:**
```python
# 3. Save to the centrally managed dynamic filename
cache_file_path = project_path / CacheConfig.get_filename()
cache.feature_cube.to_parquet(cache_file_path)
print(f"\n[SAVED] AlphaCache successfully written to {cache_file_path}")
```

**After:**
```python
# 3. Save to the centrally managed dynamic filename
output_path = codebase_path / "output"
output_path.mkdir(parents=True, exist_ok=True)
cache_file_path = output_path / CacheConfig.get_filename()
cache.feature_cube.to_parquet(cache_file_path)
print(f"\n[SAVED] AlphaCache successfully written to {cache_file_path}")
```

---

### **Part 2: `RLVR_Part2_Training.ipynb`**

#### **Cell 5 — AlphaCache Reload Location**

**Before:**
```python
# 1. Resolve path using identical centralized filename logic
cache_file_path = project_path / CacheConfig.get_filename()
feature_cube = pd.read_parquet(cache_file_path)
```

**After:**
```python
# 1. Resolve path using identical centralized filename logic
output_path = codebase_path / "output"
cache_file_path = output_path / CacheConfig.get_filename()
feature_cube = pd.read_parquet(cache_file_path)
```

---

#### **Cell 6 — PPO Training Checkpoints, History, and Final Model**

**Before:**
```python
# Ensure a directory exists for intermediate model weight checkpoints
checkpoint_dir = project_path / "model_checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
```
... (inside loop) ...
```python
        chk_path = checkpoint_dir / f"absolute_zero_ppo_epoch_{epoch+1:03d}.pt"
```
... (after loop) ...
```python
history_save_path = project_path / "ppo_training_history.csv"
df_history.to_csv(history_save_path, index=False)
```
... and ...
```python
model_file_path = project_path / "absolute_zero_ppo_agent_v1.pt"
torch.save(agent.state_dict(), str(model_file_path))
```

**After:**
```python
# Ensure a directory exists for intermediate model weight checkpoints
output_path = codebase_path / "output"
checkpoint_dir = output_path / "model_checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
```
... (inside loop — no change needed, path auto-resolves) ...
```python
        chk_path = checkpoint_dir / f"absolute_zero_ppo_epoch_{epoch+1:03d}.pt"
```
... (after loop) ...
```python
history_save_path = output_path / "ppo_training_history.csv"
df_history.to_csv(history_save_path, index=False)
```
... and ...
```python
model_file_path = output_path / "absolute_zero_ppo_agent_v1.pt"
torch.save(agent.state_dict(), str(model_file_path))
```

---

#### **Cell 7 — OOS Evaluation Outputs (Pickle, CSV, HTML)**

**Before:**
```python
results_save_path = project_path / "oos_evaluation_results.pkl"
```
... and ...
```python
csv_save_path = project_path / "oos_equity_curve.csv"
```
... and ...
```python
html_save_path = project_path / "oos_performance_chart.html"
```

**After:**
```python
output_path = codebase_path / "output"
results_save_path = output_path / "oos_evaluation_results.pkl"
```
... and ...
```python
csv_save_path = output_path / "oos_equity_curve.csv"
```
... and ...
```python
html_save_path = output_path / "oos_performance_chart.html"
```

---

### **Summary of What Changes**

| Notebook | Cell | Variable/Path Changed | New Target |
|----------|------|----------------------|------------|
| Part 1 | 5 | `cache_file_path` | `codebase_path / "output" / alpha_cache_...parquet` |
| Part 2 | 5 | `cache_file_path` (reload) | `codebase_path / "output" / alpha_cache_...parquet` |
| Part 2 | 6 | `checkpoint_dir` | `codebase_path / "output" / model_checkpoints` |
| Part 2 | 6 | `history_save_path` | `codebase_path / "output" / ppo_training_history.csv` |
| Part 2 | 6 | `model_file_path` | `codebase_path / "output" / absolute_zero_ppo_agent_v1.pt` |
| Part 2 | 7 | `results_save_path` | `codebase_path / "output" / oos_evaluation_results.pkl` |
| Part 2 | 7 | `csv_save_path` | `codebase_path / "output" / oos_equity_curve.csv` |
| Part 2 | 7 | `html_save_path` | `codebase_path / "output" / oos_performance_chart.html` |

**Note:** The raw data **loads** (`df_ohlcv`, `features_df`, `macro_df`) already point to `codebase_path / "output"` in Cell 2 of both notebooks, so those require **no changes**.
User: the current lookback=189, discuss how to add another lookback (i.e. [21, 189]) to the current code. don't code, discuss. does the current framework allow this kind of change?
Kimi: Here is a discussion of how to add a second lookback (e.g., `[21, 189]`) to the current framework, and whether the existing code supports this change.

---

## **Current Architecture: Single Lookback Design**

The framework is currently hardcoded around **one** lookback window. Evidence from the notebooks:

| Location | What it shows |
|----------|-------------|
| `CacheConfig.LOOKBACK` | A single constant (189) |
| `AlphaCache(..., lookbacks=[CacheConfig.LOOKBACK])` | A list with one element, but the API name `lookbacks` (plural) hints at future intent |
| `obs_dim=33` | The agent's neural network input size is fixed. This implies the feature cube has a fixed width |
| `alpha_cache_189d_2021_2023.parquet` | Filename embeds a single lookback value |

**Verdict:** The framework was **designed with multi-lookback in mind** (note the plural `lookbacks` parameter), but the current training pipeline is **not wired to consume multiple lookbacks simultaneously**.

---

## **What Would Need to Change? (No Code)**

### **1. AlphaCache Generation (Part 1)**

**Current behavior:** Builds one feature cube where each observation contains 189 days of history.

**To support `[21, 189]`:** The `AlphaCache.build()` method would need to:

*   **Generate two separate feature sets** for each `(date, ticker)` pair:
    *   A "short-term" feature slice (21-day lookback)
    *   A "long-term" feature slice (189-day lookback)
*   **Concatenate them** into a wider observation vector, OR
*   **Keep them as separate tensors** and let the model fuse them

**Decision point:** Do the 21-day features mean the *same* indicators computed over 21 days, or *different* indicators? For example:
*   Same indicators, shorter window → 21-day RSI, 21-day volatility
*   Different indicators → short-term might use intraday patterns, long-term uses macro trends

This affects whether `AlphaCache` simply loops the same calculation with different window sizes, or needs a whole new feature definition pipeline.

---

### **2. The Feature Cube Dimensions**

**Current:** `(dates × tickers × 33 features)` — all 33 features derived from 189-day history.

**With dual lookback:** Two possible shapes:

| Approach | Shape | Pros/Cons |
|----------|-------|-----------|
| **Concatenated** | `(dates × tickers × 66 features)` | Simplest. Agent sees everything at once. But loses structural separation between time scales. |
| **Stacked** | `(dates × tickers × 2 lookbacks × 33 features)` | Preserves structure. Requires model architecture change (e.g., separate processing branches). |
| **Separate caches** | Two files: `alpha_cache_21d_...` and `alpha_cache_189d_...` | Most flexible. Training notebook decides how to combine. But doubles storage and load complexity. |

The `CacheConfig.get_filename()` method would need to handle naming: `alpha_cache_21d_189d_2021_2023.parquet` or similar.

---

### **3. The Agent's Neural Network (Part 2)**

**Current:** `AbsoluteZeroAgent(obs_dim=33, action_dim=13)`

**Problem:** If you concatenate 21-day and 189-day features into 66 inputs, `obs_dim` must become `66`. This is a **hard architectural change** — the first layer's weight matrix shape changes.

**Options:**

| Model Architecture | How it handles dual lookback | Complexity |
|--------------------|------------------------------|------------|
| **Flat concatenation** | Single input layer of size 66. Let the network learn which features matter at which scale. | Low. Just change `obs_dim`. Risk: network may ignore one scale entirely. |
| **Two-tower / Siamese** | Two separate input branches (21-branch and 189-branch), each with their own layers, fused before the action head. | Medium. Requires rewriting `AbsoluteZeroAgent`. Benefit: forces the network to learn scale-specific representations. |
| **Attention / Transformer** | Treat the two lookbacks as a sequence of two "tokens" and let attention learn cross-scale relationships. | High. Major architectural overhaul. |

**The `RolloutBuffer` also hardcodes `obs_dim=33`** in its initialization, so it would need to match whatever the agent expects.

---

### **4. The Reward Oracle**

**Current:** `oracle.precompute_reward_matrix(holding_period=5)`

**Question:** Does the reward (5-day forward return) change based on lookback?

*   **No, keep reward the same:** The agent's *action* (what to buy) and *outcome* (5-day profit) are independent of how far back it looked. The lookback only affects the *state observation*. This is the cleanest interpretation.
*   **Yes, scale reward horizon with lookback:** Perhaps the 21-day lookback agent should be rewarded on 1-day returns, and 189-day on 20-day returns. This would require **multiple oracles** or a dynamic reward function.

The current `RLOracle` design suggests a single reward matrix, so the "reward stays the same, only state changes" approach is more compatible.

---

### **5. Training Loop & Validation**

**Current:** One environment, one agent, one buffer.

**With dual lookback:** The environment must serve **combined observations**:

```python
# Conceptually, inside DiscoveryEnv or RLVRGymEnv:
obs_short = cache_21d.get_obs(date, ticker)   # shape (33,)
obs_long  = cache_189d.get_obs(date, ticker)  # shape (33,)
obs = np.concatenate([obs_short, obs_long])    # shape (66,)
```

Or, if using the two-tower model, the environment returns a tuple or dict:
```python
obs = {"short": obs_short, "long": obs_long}
```

This requires changes to:
*   `DiscoveryEnv` — how it queries the cache
*   `RLVRGymEnv` — how it formats observations for the agent
*   `RolloutBuffer` — how it stores observations (must match agent input format)

---

## **Does the Current Framework Allow This?**

| Aspect | Assessment |
|--------|------------|
| **Data pipeline (`AlphaCache`)** | **Partially.** The `lookbacks` parameter is already a list, suggesting the author anticipated this. But the internal logic for multi-lookback feature generation, concatenation, and filename handling is not shown in the notebooks. |
| **Environment (`DiscoveryEnv`, `RLVRGymEnv`)** | **No.** The environment takes a single `feature_cube`. It has no concept of multiple caches or fused observations. |
| **Agent (`AbsoluteZeroAgent`)** | **No.** Fixed `obs_dim=33`. Changing this requires redefining the network architecture. |
| **Trainer (`PPOTrainer`, `RolloutBuffer`)** | **Partially.** The buffer takes `obs_dim` as an argument, so it is configurable. But it assumes a flat vector observation. Dict/tuple observations (for two-tower models) would break it. |
| **Config system (`CacheConfig`, `TradingConfig`)** | **Partially.** `CacheConfig.get_filename()` would need to encode multiple lookbacks. The `START_DATE`/`END_DATE` logic is unaffected. |

---

## **Recommended Migration Path (Discussion Only)**

If you wanted to implement this without breaking the existing single-lookback workflow:

1. **Phase 1 — Data:** Modify `AlphaCache` to accept multiple lookbacks and emit either:
    *   A single wide cube (concatenated), OR
    *   Multiple cube files with a naming convention

2. **Phase 2 — Config:** Update `CacheConfig` to store a list of lookbacks and generate filenames like `alpha_cache_21d_189d_...parquet`

3. **Phase 3 — Agent (optional):** Start with the **flat concatenation** approach (easiest). Change `obs_dim` dynamically based on the cache metadata. Only move to two-tower if performance is poor.

4. **Phase 4 — Environment:** Update `DiscoveryEnv` to accept either multiple cache objects or a wide feature cube, and slice the correct columns for the observation.

5. **Phase 5 — Backward compatibility:** Keep `lookbacks=[189]` as the default so existing single-lookback experiments still run unchanged.

---

## **Bottom Line**

> **Is the framework "ready" for this?**  
> The *data pipeline API* (`lookbacks` as a list) suggests the author thought about it. But the *training pipeline* (fixed `obs_dim`, single cache file, flat environment) is not ready. You would need to touch **Part 1 (cache builder)**, **Part 2 (agent architecture)**, and **the glue between them (environment)**. It is a moderate refactor, not a one-line change.
