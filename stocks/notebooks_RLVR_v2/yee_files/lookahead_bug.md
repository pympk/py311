Here is a step-by-step breakdown of how the RL environment step aligns with real-world time, and exactly where the **lookahead ("time travel") bug** occurred.

---

### 1. The Real-World Trading Timeline vs. The Code

Imagine a concrete calendar week with a single stock (e.g., **NVDA**):

| Day | Time | Real-World Event | What Happened in Your Buggy Code (Notebook 02) |
| :--- | :--- | :--- | :--- |
| **Monday (T)** | 09:30 - 16:00 | NVDA surges **+10%** from Friday's close ($100 \to $110). | Market trades throughout the day. |
| **Monday (T)** | 16:00 (Close) | **Decision Date:** Markets close. We calculate features (NVDA momentum, RSI, 5-day slope). | `Observation` is built and given to the RL Agent. |
| **Monday (T)** | 16:00:01 | **Agent Action:** The agent sees NVDA's strong features and outputs action: *"Buy NVDA"*. | `env.step(action)` is called. |
| **Inside `step()`** | — | **The Bug Occurs Here!** | `reward_matrix.loc["Monday"]` was set to `Ret_1d` ($+10\%$).<br>`self.active_sleeves.append(["NVDA"])` immediately added NVDA to Monday's portfolio.<br>The environment immediately returned `reward = +10%` to the agent! |

---

### 2. Why is this Lookahead / Hindsight Bias?

In real life:
- If you make a decision on **Monday at 4:00 PM Close**, you **cannot** capture the $+10\%$ that happened between Friday Close and Monday Close. That return is in the past.
- Your profit must come from price changes **after** you enter the trade (e.g., Monday Close $\to$ Tuesday Close, or Tuesday Close $\to$ Wednesday Close).

In Notebook 02:
- The agent was allowed to observe Monday's $+10\%$ surge in the state vector, decide to buy NVDA at Monday 4:00 PM, and **get paid Monday's $+10\%$ surge as its reward!**
- The RL agent quickly learned a trivial rule: *"Whenever a stock had a huge positive return today, choose it, and I will be rewarded with today's positive return."*
- This is why the **Validation Sharpe was 7.69**—the agent was essentially picking winning lottery numbers after the numbers had already been broadcast.

---

### 3. Why Did Out-of-Sample (OOS) Sharpe Collapse to 0.245 in Notebook 03?

In Notebook 03:
```python
# Forward 1-Day Return: (P_{t+1} - P_t) / P_t
reward_matrix = df_close.pct_change(1).shift(-1)
```
- In Notebook 03, `reward_matrix` was defined using `.shift(-1)` (Monday's row contained Tuesday's return: $P_{\text{Tue}} - P_{\text{Mon}}$).
- The agent was never trained to predict **tomorrow's** price movement ($T \to T+1$); it was only trained to identify **today's** past price movement ($T-1 \to T$).
- When tested on genuine forward returns in Notebook 03, the "hindsight cheat" was gone, and the strategy collapsed to a near-zero Sharpe ratio ($0.245$).

---

### 4. The Proper T+1 Execution Timeline (Next-Day Close)

Under your domain model's **Next-Day Close (T+1 Close)** convention to eliminate all execution slippage and lookahead bias:

```text
Day T (Close)           : Decision Date. Agent computes action A_T.
                          Order is placed with broker ("Market-on-Close for Day T+1").
                          Status: PENDING. (0% PnL exposure to new stock).

Day T -> Day T+1        : Pending interval. Portfolio return = returns of older active sleeves.

Day T+1 (Close)         : Order executes at P_{T+1}. Sleeve becomes ACTIVE.

Day T+1 -> Day T+2      : Holding Day 1. First PnL exposure: (P_{T+2} - P_{T+1}) / P_{T+1}.
                          (From Day T perspective, this is shift(-2)).

Day T+k -> Day T+k+1    : Holding Day k. PnL exposure: shift(-(k+1)).

Day T+H+1 (Close)       : Sleeve reaches holding period H and liquidates at P_{T+H+1}.
```

---

### 5. Summary of What Needs to be Changed in Code

1. **In Notebook 02 (Training loop environment setup):**
   - Ensure `reward_matrix` is **always** forward returns (`df_close.pct_change(1).shift(-1)`), never `features_df["Ret_1d"]`.
2. **In `MTMPortfolioEngine` (`core/accounting.py`):**
   - Keep a 1-day **`pending_sleeve`** buffer so that a trade chosen at step $T$ does not earn returns during step $T$, but instead becomes active and earns returns starting at step $T+1$.