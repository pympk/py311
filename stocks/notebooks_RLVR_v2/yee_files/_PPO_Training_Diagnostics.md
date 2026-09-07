
---

# 🧠 PPO Training Diagnostics Guide

This reference explains the formulas, target ranges, and actionable interpretations for each legend trace in the **Agent Training Diagnostics** chart.

---

### 1. Total Loss & Policy Loss

* **Policy Loss ($L_{policy}$)**
  $$\mathcal{L}_{CLIP}(\theta) = -\hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right) \right]$$
  *Where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio and $\hat{A}_t$ is the estimated advantage.*
  * **Interpretation**: Measures policy update direction.
  * **Good Trend**: Fluctuates near small negative or zero values.
  * **Bad Trend**: Drops drastically to large negative numbers or explodes upward (indicates policy destabilization).

* **Total Loss ($L_{total}$)**
  $$L_{total} = L_{policy} + c_1 \cdot L_{value} - c_2 \cdot H_{entropy}$$
  * **Interpretation**: Weighted combination of all sub-losses.
  * **Good Trend**: Smooths out and converges over epochs.
  * **Bad Trend**: Continuous upward trend or sharp spikes (indicates hyperparameter mismatch).

---

### 2. Value Loss & Explained Variance (Critic Performance)

* **Value Loss ($L_{value}$)**
  $$\mathcal{L}_{VF}(\phi) = \frac{1}{2} \hat{\mathbb{E}}_t \left[ \left(V_\phi(s_t) - V_t^{target}\right)^2 \right]$$
  * **Interpretation**: Mean Squared Error (MSE) of Critic’s expected return predictions vs actual discounted rewards.
  * **Good Trend**: Decreases steadily and plateaus at a low baseline.
  * **Bad Trend**: Increases continuously or oscillates wildly (Critic failing to evaluate market states).

* **Explained Variance ($e_{var}$)**
  $$y_{var} = \text{Var}(V^{target}), \quad e_{var} = 1 - \frac{\text{Var}(V^{target} - V_\phi(s))}{\text{Var}(V^{target})}$$
  * **Interpretation**: Percentage of return variance explained by Critic ($1.0 = 100\%$ accuracy).
  * **Good Trend**: Rises towards **$0.50 \to 1.00$**.
  * **Bad Trend**: **$< 0.0$** (Critic is performing worse than predicting a simple static mean return).

---

### 3. Entropy (Exploration / Curiosity)

* **Entropy ($H$)**
  $$H(\pi_\theta(\cdot|s_t)) = \hat{\mathbb{E}}_t \left[ -\ln \pi_\theta(a_t|s_t) \right]$$
  *For Gaussian policy with covariance matrix $\Sigma$: $H = \frac{k}{2}(1 + \ln(2\pi)) + \frac{1}{2}\ln|\Sigma|$*
  * **Interpretation**: Measures policy randomness and continuous exploration.
  * **Good Trend**: Smooth, gradual decay from high initial state down to a steady plateau (transitioning from exploration to exploitation).
  * **Bad Trend**: Drops abruptly to $\approx 0$ in early epochs (**Premature Convergence** / Agent gets stuck in sub-optimal strategy) or never drops (Agent fails to learn).

---

### 4. PPO Stability Metrics (Approx KL & Clip Fraction)

* **Approx KL ($D_{KL}$)**
  $$D_{KL} \approx \frac{1}{2} \hat{\mathbb{E}}_t \left[ \left(\ln \pi_{\theta_{old}}(a_t|s_t) - \ln \pi_\theta(a_t|s_t)\right)^2 \right]$$
  * **Interpretation**: Quantifies how far policy distribution moves in a single update step.
  * **Good Range**: **$0.005 \le D_{KL} \le 0.030$**.
  * **Bad Trend**: **$> 0.05$** (Updates too large; risk of policy collapse) or **$\approx 0.000$** (Agent frozen; non-learning).

* **Clip Fraction ($\text{Frac}_{clip}$)**
  $$\text{Frac}_{clip} = \hat{\mathbb{E}}_t \left[ \mathbb{I}\left(|r_t(\theta) - 1| > \epsilon\right) \right]$$
  * **Interpretation**: Percentage of mini-batch updates hitting clipping threshold $\epsilon$ (typically $0.2$).
  * **Good Range**: **$0.05 \le \text{Frac}_{clip} \le 0.20$** ($5\% - 20\%$).
  * **Bad Trend**: **$> 0.30$** (Learning rate or update epochs too high) or **$< 0.01$** (Learning rate too small).

---

### 5. Average Step Reward

* **Average Step Reward ($\bar{R}$)**
  $$\bar{R} = \frac{1}{N} \sum_{t=1}^{N} R_t$$
  *Where $R_t$ is the raw or penalized alpha reward received per environment step.*
  * **Interpretation**: Agent’s out-of-sample/in-sample performance signal during training.
  * **Good Trend**: Consistent upward trajectory across training epochs.s

---

### Quick Diagnostic Reference Summary

| Diagnostic Metric | Healthy Range / Trend | Primary Remedy if Unhealthy |
| :--- | :--- | :--- |
| **Value Loss** | Low & Steady Decay | Lower `LR` or increase Critic network capacity |
| **Explained Var** | $0.50 \to 1.00$ | Increase Value Network depth / Decrease reward noise |
| **Entropy** | Gradual Monotonic Decay | Increase/decrease `ENT_COEF` |
| **Approx KL** | $0.005 \to 0.030$ | Lower `LR` or reduce `UPDATE_EPOCHS` |
| **Clip Fraction** | $0.05 \to 0.20$ | Adjust PPO clip parameter $\epsilon$ or batch size |
| **Avg Reward** | Upward Trend | Tweak advantage normalization or reward shaping penalty |