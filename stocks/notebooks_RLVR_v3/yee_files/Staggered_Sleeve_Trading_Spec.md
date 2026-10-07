
---

# Specification: Overlapping Multi-Sleeve Mark-to-Market (MTM) Portfolio Engine

## 1. System Overview & Core Parameters

The execution model implements an **overlapping, multi-sleeve, mark-to-market (MTM)** portfolio strategy. Total portfolio capital is partitioned across time into $H$ discrete sub-portfolios called **Sleeves**, where $H = \text{holding\_period}$.

* **$H$ (`holding_period`)**: Number of trading days a sleeve is held ($H \ge 1$).
* **Sleeve Capital Fraction**: $\text{Weight}_{\text{sleeve}} = \frac{1}{H}$
* **$K_T$ (`selected_tickers`)**: Number of assets chosen at day $T$.
* **Asset Weight in Sleeve**: $\omega_{i, T} = \frac{1}{K_T}$ for all $i \in K_T$.
* **Global Asset Weight**: $W_{i, T}^{\text{global}} = \frac{1}{H \times K_T}$

---

## 2. Sleeve Lifecycle & Execution Timeline

Execution operates on a **Next-Day Close (T+1 Close)** convention to eliminate lookahead bias.

```text
Day T (Close)          Day T+1 (Close)        Day T+2 (Close)                 Day T+H+1 (Close)
──────●──────────────────────●──────────────────────●─────────────────...────────────●─────────>
[Decision A_T]         [Trade Executes]       [Holding Day 1]                 [Holding Day H]
 Order Pending          Entry Price            MTM Step 1                      Exit Price &
                        P_{T+1}                P_{T+2}                         Liquidation P_{T+H+1}
```

### Chronological Step-by-Step

| Step | Time / Trigger | State / Event | Mathematical Definition / Action |
| :--- | :--- | :--- | :--- |
| **0. Decision** | **Day $T$ Close** | Agent generates action $A_T$ | $A_T \to \text{Select assets } K_T \text{ using info up to Close}_T$. |
| **1. Pending** | **$T \to T+1$** | Order is in transit | **No PnL exposure.** Portfolio has zero risk on action $A_T$. |
| **2. Entry** | **Day $T+1$ Close** | Sleeve $T$ is initialized | Buy all $i \in K_T$ at price $P_{i, T+1}$. Sleeve becomes **ACTIVE**. |
| **3. Holding** | **Day $T+1 \to T+H+1$** | Active MTM holding | Sleeve accrues daily 1-day MTM returns for $H$ consecutive days. |
| **4. Exit** | **Day $T+H+1$ Close** | Sleeve $T$ is liquidated | Sell all $i \in K_T$ at price $P_{i, T+H+1}$. Sleeve is **TERMINATED**. |

---

## 3. Daily Mark-to-Market (MTM) Return Mechanics

### 3.1. Single Sleeve Daily Return
For a sleeve initialized at Day $T+1$, its mark-to-market return on any holding day $k \in \{1, 2, \dots, H\}$ (from day $T+k$ to day $T+k+1$) is:

$$R_{\text{sleeve } T}(T+k+1) = \sum_{i \in K_T} \omega_{i, T} \cdot \left( \frac{P_{i, T+k+1} - P_{i, T+k}}{P_{i, T+k}} \right)$$

### 3.2. Total Portfolio Daily Reward (Mark-to-Market)
On any calendar day $t$, the total portfolio daily return (agent reward $r_t$) is the sum of contributions from all **currently active sleeves**:

$$r_t = R_{\text{portfolio}}(t) = \frac{1}{H} \sum_{s \in \mathcal{S}_t^{\text{active}}} R_{\text{sleeve } s}(t)$$

---

## 4. Pipeline Progression (Warm-Up to Steady-State)

```text
Day 1:   [Sleeve 1]                                                      => Total Active = 1/H
Day 2:   [Sleeve 1] + [Sleeve 2]                                         => Total Active = 2/H
...
Day H:   [Sleeve 1] + [Sleeve 2] + ... + [Sleeve H]                      => Total Active = 1.0 (Full)
Day H+1: [Sleeve 2] + [Sleeve 3] + ... + [Sleeve H+1]   (Sleeve 1 drops) => Total Active = 1.0 (Steady State)
```

### Invariant Rules for AI Agents:
1. **Steady-State Invariant**: For all $t > H$, exactly $H$ sleeves are active simultaneously. Total active capital is always $100\%$ ($H \times \frac{1}{H} = 1.0$).
2. **Warm-Up Phase**: For $t \le H$, portfolio capital utilized is $\frac{t}{H}$ (unallocated cash return = $0\%$).

---

## 5. Vectorized Indexing / Shift Reference (Row $T$ Perspective)

When computing target forward returns in a 2D tabular feature store indexed at **Decision Row $T$**:

| Holding Day ($k$) | Price Interval Held | Target Return Formula | Offset from Row $T$ |
| :--- | :--- | :--- | :--- |
| **Day 1** | Close $T+1 \to$ Close $T+2$ | $(P_{T+2} - P_{T+1}) / P_{T+1}$ | `ret_1d.shift(-2)` |
| **Day 2** | Close $T+2 \to$ Close $T+3$ | $(P_{T+3} - P_{T+2}) / P_{T+2}$ | `ret_1d.shift(-3)` |
| ... | ... | ... | ... |
| **Day $H$** | Close $T+H \to$ Close $T+H+1$ | $(P_{T+H+1} - P_{T+H}) / P_{T+H}$ | `ret_1d.shift(-(H+1))` |