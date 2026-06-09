import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import cast

# --- Domain Imports ---
from core.settings import TradingConfig
from data_pipeline.screener import UniverseScreener
from data_pipeline.cache import AlphaCache
from rl_discovery.oracle import RLOracle
from rl_discovery.environment import DiscoveryEnv
from rl_discovery.adapter import RLVRGymEnv
from rl_discovery.agent import AbsoluteZeroAgent
from rl_discovery.trainer import RolloutBuffer, PPOTrainer
from rl_discovery.validator import AgentEvaluator

# ============================================================================
# TASK 1: DATA PIPELINE PREP & ALPHA CACHE
# ============================================================================
print("\n--- [TASK 1] Initializing Universe and AlphaCache ---")

# Load the file you placed in the folder (adjust extension as needed)
df_ohlcv = pd.read_parquet("df_ohlcv.parquet")
features_df = pd.read_parquet("features_df.parquet")
macro_df = pd.read_parquet("macro_df.parquet")

# Setup Config & Extract Wide Close Prices
config = TradingConfig()
# Convert MultiIndex OHLCV to Wide Format (Dates x Tickers)
df_close = df_ohlcv["Adj Close"].unstack(level=0)
# Ensure clean calendar
trading_calendar = pd.DatetimeIndex(df_close.dropna(how="all").index.sort_values())

# 1. Initialize Screener (The Gatekeeper)
screener = UniverseScreener(
    df_close=df_close,
    features_df=features_df,
    macro_df=macro_df,
    trading_calendar=trading_calendar,
    config=config,
)

# 2. Pre-compute State Space (Alpha Matrix)
# We use a 189-day lookback as our standard strategy evaluation window
cache = AlphaCache(screener=screener, config=config, lookbacks=[189])

# MENTOR NOTE: Building the cache is compute-heavy but only happens once.
# In a real DeepMind workflow, this cache would be serialized to a Parquet blob.
train_start = "2015-01-01"
cache.build(start_date=train_start)

# MENTOR NOTE: Save the heavy CPU computation to your Google Drive
cache_file_path = "alpha_cache_189d_2015_2023.parquet"

# Save it to disk
cache.feature_cube.to_parquet(cache_file_path)
print(f"[SAVED] AlphaCache written to {cache_file_path}")

# In future sessions, you can skip cache.build() and just do:
# cache.feature_cube = pd.read_parquet(cache_file_path)

# ============================================================================
# TASK 2: RL ENVIRONMENT & ORACLE INITIALIZATION
# ============================================================================
print("\n--- [TASK 2] Bootstrapping RL Oracle and Environments ---")

# 1. Oracle (The Veritable Reward Truth)
holding_period = 5
oracle = RLOracle(screener, config)
oracle.precompute_reward_matrix(holding_period=holding_period)

# 2. Time-Split Calendars (Vertical Slicing Temporal Data)
cal_train = trading_calendar[
    (trading_calendar >= pd.Timestamp("2015-01-01"))
    & (trading_calendar < pd.Timestamp("2023-01-01"))
]
cal_test = trading_calendar[(trading_calendar >= pd.Timestamp("2023-01-01"))]

# 3. Environments
env_train = DiscoveryEnv(
    cache.feature_cube, oracle.reward_matrix, cal_train, holding_period
)
gym_train = RLVRGymEnv(env_train, macro_df)

env_test = DiscoveryEnv(
    cache.feature_cube, oracle.reward_matrix, cal_test, holding_period
)
gym_test = RLVRGymEnv(env_test, macro_df)

# ============================================================================
# TASK 3: PPO TRAINING LOOP
# ============================================================================
print("\n--- [TASK 3] Executing PPO Actor-Critic Training Loop ---")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware Accelerator: {device}")

# Neural Nets and Trainer
agent = AbsoluteZeroAgent(obs_dim=33, action_dim=13).to(device)
trainer = PPOTrainer(agent, lr=3e-4, clip_coef=0.2, ent_coef=0.01)

num_epochs = 50
num_steps = 64  # Rollout batch size

# MENTOR NOTE: We pre-allocate memory to avoid memory fragmentation overhead
# during the highly iterative PyTorch backward passes.
buffer = RolloutBuffer(num_steps=num_steps, obs_dim=33, action_dim=13, device=device)

for epoch in range(num_epochs):
    obs, _ = gym_train.reset()
    epoch_rewards = []

    # 1. Rollout Phase (Data Collection)
    for step in range(num_steps):
        # [GUARD] Trap NaNs before they hit PyTorch
        if np.isnan(obs).any():
            raise ValueError(
                "NaN detected in observation. Stopping to prevent exploding gradients."
            )

        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)

        # MENTOR NOTE: torch.no_grad() is essential here. We are *acting* in the environment,
        # not learning. We don't want PyTorch wasting memory building a computational graph.
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(
                obs_tensor.unsqueeze(0)
            )

        # Step Environment
        next_obs, reward, done, _, info = gym_train.step(action.cpu().numpy()[0])

        # Store experience
        buffer.add(obs, action[0], logprob[0], reward, value[0], done)
        epoch_rewards.append(reward)

        obs = next_obs
        if done:
            obs, _ = gym_train.reset()

    # 2. Advantage Calculation (GAE)
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        next_value = agent.get_value(obs_tensor).flatten()

    buffer.compute_advantages(next_value, next_done=done)

    # 3. Surrogate Objective Optimization Phase
    trainer.update(buffer, update_epochs=4, mini_batch_size=16)

    # Buffer reset implicitly handled by tracking pointers in RolloutBuffer (step = 0)
    buffer.step = 0

    if (epoch + 1) % 10 == 0:
        mean_rew = np.mean(epoch_rewards)
        # Assuming Veritable Reward is log return, let's translate to roughly % gain per step
        pct_gain = (np.exp(mean_rew) - 1.0) * 100
        print(
            f"Epoch {epoch+1:03d}/{num_epochs} | Mean Step Reward: {mean_rew:.6f} ({pct_gain:.2f}%)"
        )

# MENTOR NOTE: Save the trained Neural Network weights
model_file_path = "absolute_zero_ppo_agent_v1.pt"

# torch.save serializes the dictionary of weights
torch.save(agent.state_dict(), model_file_path)
print(f"[SAVED] Trained Agent weights written to {model_file_path}")

# ============================================================================
# TASK 4: OUT-OF-SAMPLE VALIDATION
# ============================================================================
print("\n--- [TASK 4] Walk-Forward Deterministic Evaluation (OOS) ---")

# Switch to Deterministic execution. In trading, exploration is for the sandbox;
# exploitation is for capital allocation.
results = AgentEvaluator.evaluate(agent, gym_test, device)

print(f"\n[OOS METRICS]")
print(f"Total Cumulative Return: {results['total_return']*100:.2f}%")
print(f"Sharpe Ratio (Annualized): {results['sharpe_ratio']:.3f}")
print(f"Total Trading Steps: {results['steps']}")

# Plotting the Equity Curve using DDD Ubiquitous Charting Style
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=results["dates"],
        y=results["equity_curve"],
        mode="lines",
        line=dict(color="green", width=2),
        name="RL Agent Portfolio (OOS)",
    )
)

fig.update_layout(
    title=f"Absolute Zero: Out-of-Sample Agent Performance<br><sup>OOS Sharpe: {results['sharpe_ratio']:.2f} | Returns: {results['total_return']*100:.2f}%</sup>",
    yaxis_title="Cumulative Return (Base 1.0)",
    xaxis_title="Date",
    template="plotly_white",
    hovermode="x unified",
    height=500,
)

fig.show()

# Save the results dictionary as a CSV for quantitative debugging
df_results = pd.DataFrame(
    {"Date": results["dates"], "Equity_Curve": results["equity_curve"]}
)
df_results.to_csv("oos_equity_curve.csv", index=False)
print("[SAVED] Equity curve raw data saved to oos_equity_curve.csv")

# Save the interactive Plotly chart as an HTML file
fig.write_html("oos_performance_chart.html")
print("[SAVED] Interactive chart saved to oos_performance_chart.html")
#
