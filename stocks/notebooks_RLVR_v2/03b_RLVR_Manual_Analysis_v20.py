# %% [markdown]
# # 03b_RLVR_Manual_Analysis_v20.ipynb
#
# ### 🔬 Granular Out-of-Sample Performance & Factor Allocation Inspection
# - **Observation Space:** 46 Dimensions (12 CS Mean, 12 CS Std, 12 Benchmark Vector, 10 Stationary Macro).
# - **Action Space:** 16 Dimensions (12 Feature Scoring Weights, Offset, Width, Equity Exposure, Active Tilt).
# - **Zero-Regex Checkpoint Loading:** Queries `model_catalog.parquet` or checkpoint payloads directly.
# - **Dynamic Benchmark Integration:** Dynamic `config.benchmark` throughout.

# %% [code]
# CELL 1: Environment Setup & Automatic Champion Discovery
import gc
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from IPython.display import display
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

from core.paths import LOCAL_DATA_DIR, OUTPUT_DIR
from core.settings import CacheConfig, TradingConfig
from data_pipeline.loader import load_processed_data
from data_pipeline.utils import get_master_trading_calendar
from rl_discovery.adapter import RLVRGymEnv
from rl_discovery.agent import AbsoluteZeroAgent
from rl_discovery.environment import DiscoveryEnv
from rl_discovery.validator import AgentEvaluator
from strategy.registry import get_strategy_registry

RUNNING_IN_COLAB = "google.colab" in sys.modules
if RUNNING_IN_COLAB:
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")
    rl_root = Path(
        "/content/drive/Othercomputers/My Computer/Files_win10/python/py311/stocks/notebooks_RLVR_v2"
    )
else:
    rl_root_default = Path(
        r"C:\Users\ping\Files_win10\python\py311\stocks\notebooks_RLVR_v2"
    )
    rl_root = rl_root_default if rl_root_default.exists() else Path.cwd()

if str(rl_root) not in sys.path:
    sys.path.append(str(rl_root))
os.chdir(rl_root)

checkpoint_dir = OUTPUT_DIR / "model_checkpoints"
if not checkpoint_dir.exists():
    raise FileNotFoundError(f"❌ Checkpoint directory does not exist: {checkpoint_dir}")

# Discover champion using model_catalog.parquet or direct payload inspection (NO REGEX)
catalog_path = OUTPUT_DIR / "model_catalog.parquet"
target_file: Optional[str] = None

if catalog_path.exists():
    try:
        catalog_df = pd.read_parquet(catalog_path)
        # Filter for models that physically exist in the checkpoint directory
        catalog_df["exists"] = catalog_df["checkpoint_name"].apply(
            lambda name: (checkpoint_dir / str(name)).exists()
        )
        valid_catalog = catalog_df[catalog_df["exists"]].copy()

        if not valid_catalog.empty:
            sorted_cat = valid_catalog.sort_values(
                by=["rolling_val_sharpe", "val_sharpe"], ascending=[False, False]
            )
            target_file = str(sorted_cat.iloc[0]["checkpoint_name"])
            print(f"📊 Discovered Champion from Catalog: {target_file}")
            print(
                f"   Rolling Sharpe: {sorted_cat.iloc[0].get('rolling_val_sharpe', 0.0):.3f} | Val Sharpe: {sorted_cat.iloc[0].get('val_sharpe', 0.0):.3f}"
            )
    except Exception as e:
        print(f"⚠️ Could not load catalog: {e}. Falling back to payload scan...")

if target_file is None:
    # Direct payload inspection fallback
    scored_models = []
    for pt_file in checkpoint_dir.glob("*.pt"):
        try:
            ckpt_meta = torch.load(pt_file, map_location="cpu", weights_only=False)
            r_sh = float(ckpt_meta.get("rolling_val_sharpe", -999.0))
            v_sh = float(ckpt_meta.get("val_sharpe", -999.0))
            scored_models.append((pt_file.name, r_sh, v_sh))
        except Exception:
            continue

    if not scored_models:
        raise RuntimeError(
            f"❌ No loadable `.pt` checkpoints found in: {checkpoint_dir}"
        )

    scored_models.sort(key=lambda x: (x[1], x[2]), reverse=True)
    target_file = scored_models[0][0]
    print(f"🔍 Discovered Champion via Payload Scan: {target_file}")

print(f"\n👉 Active target_file initialized: {target_file}")

# %% [code]
# CELL 2: Manual Target File Override (Leave as is to use auto-champion)
# target_file = "model_P2_Pivot_Tilt20_Width5_Gamma88_s42_ep_52.pt"

# %% [code]
# CELL 3: Checkpoint Loading & Deterministic OOS Evaluation
target_file = str(target_file).strip().replace('"', "")
target_path = checkpoint_dir / target_file
if not target_path.exists():
    target_path = OUTPUT_DIR / target_file
if not target_path.exists():
    raise FileNotFoundError(f"❌ Checkpoint file not found: {target_file}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🤖 Loading PyTorch Checkpoint: {target_path.name}")
checkpoint = torch.load(target_path, map_location=device, weights_only=False)

grid_params = checkpoint.get("grid_params", {}) if isinstance(checkpoint, dict) else {}
model_state = (
    checkpoint.get("model_state_dict", checkpoint)
    if isinstance(checkpoint, dict)
    else checkpoint
)
scaler_state = (
    checkpoint.get("scaler_state", None) if isinstance(checkpoint, dict) else None
)

# Reconstruct TradingConfig directly from internal dictionary
config = TradingConfig()

if "holding_period" in checkpoint:
    config.holding_period = int(checkpoint["holding_period"])
elif "holding_period" in grid_params:
    config.holding_period = int(grid_params["holding_period"])

if "benchmark" in checkpoint:
    config.benchmark_ticker = str(checkpoint["benchmark"])
elif "benchmark" in grid_params:
    config.benchmark_ticker = str(grid_params["benchmark"])

if "min_active_tilt" in grid_params:
    config.min_active_tilt = float(grid_params["min_active_tilt"])
if "min_width" in grid_params:
    config.min_basket_width = int(grid_params["min_width"])
elif "min_basket_width" in grid_params:
    config.min_basket_width = int(grid_params["min_basket_width"])

if "max_cash_pct" in grid_params:
    config.max_cash_pct = float(grid_params["max_cash_pct"])
if "loss_aversion_penalty" in grid_params:
    config.loss_aversion_penalty = float(grid_params["loss_aversion_penalty"])
if "UPSIDE_ALPHA_MULT" in grid_params:
    config.upside_alpha_mult = float(grid_params["UPSIDE_ALPHA_MULT"])
if "GAMMA" in grid_params:
    config.gamma = float(grid_params["GAMMA"])

print(
    f"🔧 Reconstructed TradingConfig: benchmark = {config.benchmark}, "
    f"holding_period = {config.holding_period}, min_active_tilt = {config.min_active_tilt}, "
    f"min_basket_width = {config.min_basket_width}, gamma = {config.gamma}"
)

# Load data and prepare OOS test bench
df_ohlcv, macro_df, features_df = load_processed_data()
df_close = df_ohlcv["Adj Close"].unstack(level=0).sort_index()
trading_calendar = get_master_trading_calendar(df_ohlcv, config.calendar_ticker)

cache_file_path = LOCAL_DATA_DIR / CacheConfig.get_filename()
feature_cube = pd.read_parquet(cache_file_path)

reward_matrix = df_close.pct_change(1, fill_method=None).shift(-1)
reward_matrix["CASH"] = 0.0

valid_dates = feature_cube.index.get_level_values("Date").unique()
valid_calendar = trading_calendar[trading_calendar.isin(valid_dates)]
TEST_START = pd.Timestamp("2022-04-01")
cal_test = valid_calendar[valid_calendar >= TEST_START]

target_stem = Path(target_file).stem
blotter_path = OUTPUT_DIR / f"blotter_df_{target_stem}.parquet"
results_save_path = OUTPUT_DIR / f"results_{target_stem}.pkl"

# Fast-path: Reuse existing verified blotter if present
if blotter_path.exists() and results_save_path.exists():
    print(f"⚡ Found pre-computed blotter and metrics on disk: {blotter_path.name}")
    blotter_df = pd.read_parquet(blotter_path)
    with open(results_save_path, "rb") as f:
        results = pickle.load(f)
else:
    print("   -> Executing Deterministic OOS Evaluation...")
    env_test = DiscoveryEnv(
        feature_cube=feature_cube,
        simple_ret_matrix=reward_matrix,
        calendar=cal_test,
        macro_df=macro_df,
        config=config,
    )
    gym_test = RLVRGymEnv(env_test, macro_df)
    gym_test.is_training = False

    if scaler_state is not None:
        gym_test.scaler.load_state(scaler_state)

    obs_dim = gym_test.observation_space.shape[0]
    action_dim = gym_test.action_space.shape[0]
    assert obs_dim == 46, f"Expected 46 observation dimensions, got {obs_dim}"
    assert action_dim == 16, f"Expected 16 action dimensions, got {action_dim}"

    agent = AbsoluteZeroAgent(obs_dim=obs_dim, action_dim=action_dim).to(device)
    agent.load_state_dict(model_state)

    results = AgentEvaluator.evaluate(
        agent=agent, env=gym_test, device=device, detailed_log=True
    )
    blotter_df = pd.DataFrame(results.get("blotter", []))
    blotter_df.to_parquet(blotter_path)
    with open(results_save_path, "wb") as f:
        pickle.dump(results, f)

date_col = (
    blotter_df["date"] if "date" in blotter_df.columns else blotter_df["decision_date"]
)
blotter_df["date"] = pd.to_datetime(date_col)
blotter_df = blotter_df.sort_values("date").reset_index(drop=True)
print(f"✅ Total OOS days evaluated: {len(blotter_df)}")

# %% [code]
# CELL 4: Training Diagnostics (2x3 Subplot Grid)
training_history = checkpoint.get(
    "training_history", results.get("training_history", [])
)

if training_history:
    history_df = pd.DataFrame(training_history)
    epochs = history_df["epoch"]

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Total & Policy Loss",
            "Value Loss (Critic)",
            "Entropy (Exploration)",
            "PPO Stability (Approx KL & Clip Frac)",
            "Explained Variance (Value Accuracy)",
            "Average Step Reward",
        ),
        shared_xaxes=True,
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    if "total_loss" in history_df.columns:
        fig.add_trace(
            go.Scatter(
                x=epochs, y=history_df["total_loss"], mode="lines", name="Total Loss"
            ),
            row=1,
            col=1,
        )
    if "policy_loss" in history_df.columns:
        fig.add_trace(
            go.Scatter(
                x=epochs, y=history_df["policy_loss"], mode="lines", name="Policy Loss"
            ),
            row=1,
            col=1,
        )
    if "value_loss" in history_df.columns:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history_df["value_loss"],
                mode="lines",
                name="Value Loss",
                line=dict(color="orange"),
            ),
            row=1,
            col=2,
        )
    if "entropy" in history_df.columns:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history_df["entropy"],
                mode="lines",
                name="Entropy",
                line=dict(color="green"),
            ),
            row=1,
            col=3,
        )
    if "approx_kl" in history_df.columns:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history_df["approx_kl"],
                mode="lines",
                name="Approx KL",
                line=dict(color="teal"),
            ),
            row=2,
            col=1,
        )
    if "clip_fraction" in history_df.columns:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history_df["clip_fraction"],
                mode="lines",
                name="Clip Fraction",
                line=dict(color="crimson", dash="dot"),
            ),
            row=2,
            col=1,
        )
    if "explained_variance" in history_df.columns:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history_df["explained_variance"],
                mode="lines",
                name="Explained Var",
                line=dict(color="brown"),
            ),
            row=2,
            col=2,
        )
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=2, col=2)
        fig.add_hline(y=0.0, line_dash="dot", line_color="red", row=2, col=2)
    if "avg_reward" in history_df.columns:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history_df["avg_reward"],
                mode="lines",
                name="Avg Reward",
                line=dict(color="purple"),
            ),
            row=2,
            col=3,
        )

    fig.update_layout(
        height=750,
        width=1200,
        title_text=f"🧠 Agent Training Diagnostics ({target_file})",
    )
    fig.show()
else:
    print("⚠️ No training history available for this checkpoint.")

# %% [code]
# CELL 5: Action Space Mapping (16 Dimensions)
registry = get_strategy_registry(config)
action_names = list(registry.keys()) + [
    "Rank Offset",
    "Rank Width",
    "Equity Exposure",
    "Active Tilt",
]

actions = np.vstack(blotter_df["raw_actions"].values)
assert actions.shape[1] == 16, f"Expected 16 actions, got {actions.shape[1]}"

for i, name in enumerate(action_names):
    blotter_df[name] = actions[:, i]

print("=== Action Space Mapping (16 Dimensions) ===")
for i, name in enumerate(action_names):
    print(f"  [{i:02d}] {name}")

# %% [code]
# CELL 6: Observation Space Mapping (46 Dimensions)
cross_sectional_features = [
    f"{name} ({'Z' if bp.scaling_type == 'Z-Score' else 'S'})"
    for name, bp in registry.items()
]
macro_features = [
    "Mkt_Ret",
    "Mkt_Ret_Z",
    "Macro_Trend",
    "Macro_Trend_Z",
    "Yield_Curve_10Y2Y_Z",
    "Macro_Trend_Vel_Z",
    "Macro_Trend_Mom",
    "Macro_Vix_Z",
    "Macro_Vix_Ratio",
    "Mkt_Vol_63d_Z",
]
benchmark_name = config.benchmark

obs_names = (
    [f"Mean: {f}" for f in cross_sectional_features]
    + [f"Std: {f}" for f in cross_sectional_features]
    + [f"{benchmark_name}: {f}" for f in cross_sectional_features]
    + macro_features
)
assert len(obs_names) == 46, f"Expected 46 observation dimensions, got {len(obs_names)}"
print(
    f"✅ All 46 observation dimensions mapped dynamically for benchmark {benchmark_name}."
)

# %% [code]
# CELL 7: High-Level OOS Performance Summary
print("=" * 60)
print(f"🏆 CHAMPION MODEL: {target_file}")
print("=" * 60)
print(f"Total Cumulative Return : {results['total_return']*100:.2f}%")
print(f"Sharpe Ratio (Ann)      : {results['sharpe_ratio']:.3f}")
print(f"Sortino Ratio (Ann)     : {results['sortino_ratio']:.3f}")
print(f"Max Drawdown            : {results['max_drawdown']*100:.2f}%")
print(f"Information Ratio       : {results['information_ratio']:.3f}")
print(f"Beta to {benchmark_name}       : {results['beta']:.2f}")
print(f"Total Trading Days      : {results['steps']} days")
print("=" * 60)

# %% [code]
# CELL 8: Strategy Personality & Allocation Dial Distribution
mean_actions = blotter_df[action_names].mean().reset_index()
mean_actions.columns = ["Action Dimension", "Average Value"]
mean_actions = mean_actions.sort_values("Average Value", ascending=True)

fig = px.bar(
    mean_actions,
    x="Average Value",
    y="Action Dimension",
    orientation="h",
    title=f"🤖 Agent Personality & Factor Tilts (OOS Average) — {target_file}",
    color="Average Value",
    color_continuous_scale="Viridis",
)
fig.update_layout(height=650, width=950)
fig.show()

# %% [code]
# CELL 9: 4-Row Performance & Allocation Timeline
blotter_df["Benchmark_Equity"] = (1.0 + blotter_df["bm_daily_simple_ret"]).cumprod()
blotter_df["Peak"] = blotter_df["agent_equity"].cummax()
blotter_df["Drawdown"] = (blotter_df["agent_equity"] / blotter_df["Peak"]) - 1.0

t0_start_date = blotter_df["date"].iloc[0]
t0_row = pd.DataFrame(
    [
        {
            "plot_date": t0_start_date,
            "agent_equity": 1.0,
            "Benchmark_Equity": 1.0,
            "alpha_equity": 1.0,
            "Drawdown": 0.0,
            "weight_active": blotter_df["weight_active"].iloc[0],
            "weight_benchmark": blotter_df["weight_benchmark"].iloc[0],
            "weight_cash": blotter_df["weight_cash"].iloc[0],
        }
    ]
)

realized_series = pd.DataFrame(
    {
        "plot_date": blotter_df["date"],
        "agent_equity": blotter_df["agent_equity"],
        "Benchmark_Equity": blotter_df["Benchmark_Equity"],
        "alpha_equity": blotter_df["alpha_equity"],
        "Drawdown": blotter_df["Drawdown"],
        "weight_active": blotter_df["weight_active"],
        "weight_benchmark": blotter_df["weight_benchmark"],
        "weight_cash": blotter_df["weight_cash"],
    }
)

plot_df = pd.concat([t0_row, realized_series], ignore_index=True)

fig = make_subplots(
    rows=4,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    row_heights=[0.38, 0.22, 0.20, 0.20],
    subplot_titles=(
        f"📈 OOS Realized Equity Curve (Agent vs {benchmark_name})",
        "🔥 Cumulative Alpha Multiplier (Strictly V_p / V_bm)",
        f"🏛️ Dynamic Tri-Asset Allocation (Active Basket / {benchmark_name} / Cash)",
        "📉 Drawdown Profile",
    ),
)

fig.add_trace(
    go.Scatter(
        x=plot_df["plot_date"],
        y=plot_df["agent_equity"],
        mode="lines",
        name="Agent Strategy",
        line=dict(color="blue", width=2),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=plot_df["plot_date"],
        y=plot_df["Benchmark_Equity"],
        mode="lines",
        name=f"Benchmark ({benchmark_name})",
        line=dict(color="gray", width=2, dash="dash"),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=plot_df["plot_date"],
        y=plot_df["alpha_equity"],
        mode="lines",
        name="Alpha Equity",
        line=dict(color="purple", width=2),
    ),
    row=2,
    col=1,
)
fig.add_hline(y=1.0, line_dash="dot", line_color="black", row=2, col=1)

fig.add_trace(
    go.Scatter(
        x=plot_df["plot_date"],
        y=plot_df["weight_active"],
        mode="lines",
        stackgroup="alloc",
        name="Active Stocks",
        line=dict(color="forestgreen", width=0.5),
    ),
    row=3,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=plot_df["plot_date"],
        y=plot_df["weight_benchmark"],
        mode="lines",
        stackgroup="alloc",
        name=f"Benchmark ({benchmark_name})",
        line=dict(color="royalblue", width=0.5),
    ),
    row=3,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=plot_df["plot_date"],
        y=plot_df["weight_cash"],
        mode="lines",
        stackgroup="alloc",
        name="Cash",
        line=dict(color="gold", width=0.5),
    ),
    row=3,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=plot_df["plot_date"],
        y=plot_df["Drawdown"],
        mode="lines",
        fill="tozeroy",
        name="Drawdown",
        line=dict(color="red", width=1),
    ),
    row=4,
    col=1,
)

fig.update_layout(
    height=1100,
    width=1100,
    hovermode="x unified",
    title_text=f"Performance & Allocation Profile ({target_file})",
)
fig.update_yaxes(title_text="Multiplier", row=1, col=1)
fig.update_yaxes(title_text="Alpha Multiplier", row=2, col=1)
fig.update_yaxes(
    title_text="Exposure %", tickformat=".0%", range=[0, 1.0], row=3, col=1
)
fig.update_yaxes(title_text="Drawdown %", tickformat=".1%", row=4, col=1)
fig.show()

# %% [code]
# CELL 10: Complete Universe Selection & Trade Blotter Debug Inspector
sample_idx = -1
row = blotter_df.iloc[sample_idx]

u_size = row.get("universe_size", "N/A")
r_offset = row.get("offset", 0)
r_width = row.get("width", 0)
selected_tickers = row.get("tickers", [])
top_3 = row.get("top_3", [])

w_act = row.get("weight_active", 1.0)
w_bm = row.get("weight_benchmark", 0.0)
w_csh = row.get("weight_cash", 0.0)
eq_exp = row.get("equity_exposure", 1.0)
act_tilt = row.get("active_tilt", 1.0)


def fmt_date(d):
    return (
        pd.to_datetime(d).strftime("%Y-%m-%d")
        if pd.notna(d) and d is not None
        else "N/A"
    )


date_str = fmt_date(row.get("date"))
buy_str = fmt_date(row.get("buy_date"))
sell_str = fmt_date(row.get("sell_date"))

print("=" * 75)
print(f"📋 UNIVERSE SELECTION & DEBUG INSPECTOR — Decision Date: {date_str}")
print("=" * 75)

print("🗓️  1. TRADE TIMELINE")
print(f"  • Decision Date (T)     : {date_str}")
print(f"  • Execution Buy Date    : {buy_str}")
print(f"  • Realized Sell Date    : {sell_str}")
print(f"  • Holding Period Buffer : {config.holding_period} trading days")
print("-" * 75)

print("🏛️  2. CAPITAL ALLOCATION & EXECUTION DIALS")
print(f"  • Total Equity Exposure : {eq_exp*100:.2f}% (Full Market Exposure)")
print(f"  • Active Stock Tilt     : {act_tilt*100:.2f}%")
print(f"  • Effective w_active    : {w_act*100:.2f}% (Single-Stock Active Basket)")
print(f"  • Effective w_benchmark : {w_bm*100:.2f}% (Benchmark {config.benchmark})")
print(f"  • Effective w_cash      : {w_csh*100:.2f}% (Risk-Free Cash)")
print("-" * 75)

print("🧠  3. ACTIVE FACTOR WEIGHTS (Cross-Sectional Scoring)")
strat_weights = []
for strat_name in action_names:
    if strat_name not in [
        "Rank Offset",
        "Rank Width",
        "Equity Exposure",
        "Active Tilt",
    ]:
        if strat_name in row:
            strat_weights.append((strat_name, row[strat_name]))

if strat_weights:
    for name, wt in sorted(strat_weights, key=lambda x: abs(x[1]), reverse=True):
        bars_count = int(abs(wt) * 25)
        bar_char = "+" if wt >= 0 else "-"
        bar_str = bar_char * bars_count
        print(f"  • {name:<26} : {wt:+.4f} | [{bar_str:<25}]")
print("-" * 75)

print("🎯  4. UNIVERSE SELECTION & SCORES")
print(f"  • Eligible Universe Size : {u_size} tickers")
print(f"  • Cross-Section Max Score: {row.get('max_score', 0.0):+.4f}")
print(f"  • Cross-Section Min Score: {row.get('min_score', 0.0):+.4f}")
offset_val = (
    int(r_offset)
    if pd.notna(r_offset) and isinstance(r_offset, (int, float, np.integer))
    else 0
)
print(f"  • Decoded Rank Offset    : {offset_val} (Starts at Rank {offset_val + 1})")
print(f"  • Decoded Rank Width     : {r_width} tickers")
print(f"  • Top 3 Overall Universe : {top_3}")
print("-" * 75)

print(f"📦  5. ALL {len(selected_tickers)} SELECTED TICKERS:")
for rank_i, ticker in enumerate(selected_tickers, start=1):
    print(
        f"    [{rank_i:02d}] Ticker: {ticker:<6} | Universe Rank: {offset_val + rank_i} of {u_size}"
    )
print("-" * 75)

print("💰  6. DAILY MTM RETURN & ALPHA ACCOUNTING")
r_stock = row.get("gross_stock_daily_simple_ret", 0.0)
r_mkt = row.get("bm_daily_simple_ret", 0.0)
r_net = row.get("net_daily_simple_ret", 0.0)
r_alpha = row.get("alpha_daily_simple_ret", 0.0)
slippage = row.get("slippage_daily_simple_loss", 0.0)

print(f"  • Daily Active Stock Return: {r_stock*100:+.3f}%")
print(f"  • Daily Benchmark Return ({config.benchmark}): {r_mkt*100:+.3f}%")
print(f"  • Daily Amortized Slippage : -{slippage*100:.4f}% ({slippage*10000:.1f} bps)")
print(f"  • Daily Net Portfolio Return: {r_net*100:+.3f}%")
print(f"  • Daily Pure Alpha Spread  : {r_alpha*100:+.3f}% (Net of Slippage & Beta)")
print(f"  • Daily Agent Equity       : {row.get('agent_equity', 1.0):.4f}")
print(f"  • Cumulative Alpha Equity  : {row.get('alpha_equity', 1.0):.4f}")
print("=" * 75)

# %% [code]
# CELL 11: Top 5 Outperformance Days & Crash Day Radar
print("🌟 TOP 5 HIGHEST RETURN DAYS (Net of Slippage) 🌟")
top_5 = blotter_df.nlargest(5, "net_daily_simple_ret")[
    [
        "date",
        "net_daily_simple_ret",
        "bm_daily_simple_ret",
        "alpha_daily_simple_ret",
        "agent_equity",
    ]
    + action_names[:3]
].copy()
top_5["date"] = top_5["date"].dt.strftime("%Y-%m-%d")
for col in ["net_daily_simple_ret", "bm_daily_simple_ret", "alpha_daily_simple_ret"]:
    top_5[col] = top_5[col].apply(lambda x: f"{x:.4%}")
top_5["agent_equity"] = top_5["agent_equity"].apply(lambda x: f"{x:.3f}")
for col in action_names[:3]:
    top_5[col] = top_5[col].apply(lambda x: f"{float(x):.4f}")

print(top_5.to_string(index=False))

print("\n" + "=" * 50)
print("🚨 CRASH DAY ANALYSIS (Worst Single-Day Return) 🚨")
worst_idx = blotter_df["net_daily_simple_ret"].idxmin()
worst_day = blotter_df.iloc[worst_idx]
day_before = blotter_df.iloc[max(0, worst_idx - 1)]

print(
    f"Worst Day: {worst_day['date'].strftime('%Y-%m-%d')} | "
    f"Net Return: {worst_day['net_daily_simple_ret']*100:.2f}% | "
    f"Benchmark: {worst_day['bm_daily_simple_ret']*100:.2f}% | "
    f"Alpha: {worst_day['alpha_daily_simple_ret']*100:.2f}% | "
    f"Portfolio Multiplier: {worst_day['agent_equity']:.3f}"
)
print("-" * 50)

crash_weights = pd.DataFrame(
    {
        "Dimension": action_names,
        "Weight on Worst Day": worst_day[action_names].values,
        "Weight Day Prior": day_before[action_names].values,
    }
)

fig = px.line_polar(
    crash_weights,
    r="Weight on Worst Day",
    theta="Dimension",
    line_close=True,
    title=f"Radar: Agent Factor Allocation on {worst_day['date'].strftime('%Y-%m-%d')} (Crash Day)",
)
fig.update_traces(fill="toself", line_color="red")
fig.show()

# %% [code]
# CELL 12: Observation Space Sanity & Outlier Bounds (46 Dimensions)
obs_raw = blotter_df["observation"].values
obs_array = np.vstack([np.array(x).flatten() for x in obs_raw])
assert (
    obs_array.shape[1] == 46
), f"Expected 46 observation dimensions, got {obs_array.shape[1]}"

obs_df = pd.DataFrame(obs_array, columns=obs_names, index=blotter_df["date"])
desc_stats = obs_df.describe().T[["mean", "std", "min", "max"]]


def format_with_clip(val: float) -> str:
    if pd.isna(val):
        return ""
    formatted = f"{val:.4f}"
    if abs(abs(val) - 4.0) < 1e-3:
        return f"{formatted} [CLIP]"
    return formatted


mapping_table = pd.DataFrame(
    {
        "Dim Index": [f"Dim_{i:02d}" for i in range(46)],
        "Feature Name": obs_names,
        "Mean": desc_stats["mean"].map(lambda x: f"{x:.4f}").values,
        "Std Dev": desc_stats["std"].map(lambda x: f"{x:.4f}").values,
        "Min": desc_stats["min"].map(format_with_clip).values,
        "Max": desc_stats["max"].map(format_with_clip).values,
    }
)

print("\n--- Mapped Observation Space Statistics (46 Dimensions) ---")
print(mapping_table.to_string(index=False))

fig = go.Figure()
for col in obs_df.columns:
    fig.add_trace(
        go.Box(
            y=obs_df[col],
            name=col,
            boxpoints="outliers",
            jitter=0.5,
            marker=dict(size=2),
        )
    )

fig.update_layout(
    title=f"🤖 Agent Observation Space Distribution (46 Dimensions) — {target_file}",
    yaxis_title="Normalized Value",
    xaxis_title="Observation Dimension",
    height=650,
    width=1250,
    showlegend=False,
    shapes=[
        dict(
            type="line",
            y0=3.0,
            y1=3.0,
            x0=-0.5,
            x1=45.5,
            line=dict(color="orange", width=1, dash="dash"),
        ),
        dict(
            type="line",
            y0=-3.0,
            y1=-3.0,
            x0=-0.5,
            x1=45.5,
            line=dict(color="orange", width=1, dash="dash"),
        ),
        dict(
            type="line",
            y0=4.0,
            y1=4.0,
            x0=-0.5,
            x1=45.5,
            line=dict(color="red", width=1, dash="dash"),
        ),
        dict(
            type="line",
            y0=-4.0,
            y1=-4.0,
            x0=-0.5,
            x1=45.5,
            line=dict(color="red", width=1, dash="dash"),
        ),
    ],
)
fig.show()
