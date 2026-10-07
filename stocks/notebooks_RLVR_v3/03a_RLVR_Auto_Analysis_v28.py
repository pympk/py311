#!/usr/bin/env python
# coding: utf-8

#  # 03a_RLVR_Auto_Analysis_v27.ipynb
# 
# 
# 
#  ### 🤖 Automated Walk-Forward Blotter Ingestion & Diversification Audit (Gen 25 Baseline)
# 
# 
# 
#  - **Track 1: Continuous Walk-Forward Leaderboard**: Scans and parses authoritative Parquet blotters.
# 
#  - **Dynamic Lineage Resolution**: Ingests `run_metadata.json` (`output/model_checkpoints/walk_forward_gen{G}/run_metadata.json`) to bind seeds and multi-chunk boundaries dynamically.
# 
#  - **Ex-Post Capital Ensembling Verification**: Enforces linear return combination $r_{\text{blend}, t} = \frac{1}{M}\sum r_{m, t}$.
# 
#  - **Covariance & Diversification Ratio ($DR$) Decomposition**: Mathematically proves portfolio variance reduction.
# 
#  - **3-Row Synchronized Performance Architecture**: Cumulative Equity, Strict Geometric Alpha Ratio ($V_p / V_{\text{bm}}$), and Underwater Drawdown with dynamic chunk demarcations.

# In[1]:


# CELL 1: Setup & Environment Initialization
import gc
import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

from core.paths import LOCAL_DATA_DIR, OUTPUT_DIR
from core.quant import QuantUtils
from core.settings import CacheConfig, TradingConfig
from data_pipeline.loader import load_processed_data
from data_pipeline.utils import get_master_trading_calendar
from rl_discovery.adapter import RLVRGymEnv, make_eval_env
from rl_discovery.agent import AbsoluteZeroAgent
from rl_discovery.environment import DiscoveryEnv
from rl_discovery.validator import AgentEvaluator

# Universal Path Resolution
RUNNING_IN_COLAB = "google.colab" in sys.modules
if RUNNING_IN_COLAB:
    from google.colab import drive  # type: ignore[import-untyped]

    drive.mount("/content/drive")
    rl_root = Path(
        "/content/drive/Othercomputers/My Computer/Files_win10/python/py311/stocks/notebooks_RLVR_v3"
    )
else:
    rl_root_default = Path(
        r"C:\Users\ping\Files_win10\python\py311\stocks\notebooks_RLVR_v3"
    )
    rl_root = rl_root_default if rl_root_default.exists() else Path.cwd()

if str(rl_root) not in sys.path:
    sys.path.append(str(rl_root))
os.chdir(rl_root)

checkpoint_dir = OUTPUT_DIR / "model_checkpoints"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
checkpoint_dir.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    f"🚀 Evaluation Engine Initialized | Platform: {'Google Colab' if RUNNING_IN_COLAB else 'Local Workstation'} | Device: {device}"
)
print(f"📁 Checkpoint Root: {checkpoint_dir}")
print(f"📁 Output Destination: {OUTPUT_DIR}")


# In[2]:


# CELL 2: Pre-load Market Data & Precompute Base Returns
print("\n" + "=" * 65)
print("📦 PRE-LOADING MARKET DATA & ALPHACACHE...")
print("=" * 65)

base_config = TradingConfig()
benchmark_sym = base_config.benchmark

data = load_processed_data()
df_ohlcv = data.df_ohlcv
macro_df = data.macro_df
features_df = data.features_df

df_close = df_ohlcv["Adj Close"].unstack(level=0).sort_index()
trading_calendar = get_master_trading_calendar(df_ohlcv, base_config.calendar_ticker)

cache_file_path = LOCAL_DATA_DIR / CacheConfig.get_filename()
feature_cube = pd.read_parquet(cache_file_path)

reward_matrix = df_close.pct_change(1, fill_method=None).shift(-1)
reward_matrix["CASH"] = 0.0

valid_dates = set(feature_cube.index.get_level_values("Date").unique())
valid_calendar = trading_calendar[trading_calendar.isin(valid_dates)]

print(
    f"📅 Valid Master Calendar: {valid_calendar.min().strftime('%Y-%m-%d')} to {valid_calendar.max().strftime('%Y-%m-%d')} ({len(valid_calendar)} sessions)"
)
print(
    f"🏛️ Dynamic Benchmark: {benchmark_sym} | Default Holding Period: {base_config.holding_period} sessions"
)
print("=" * 65)

del df_ohlcv
gc.collect()


# In[3]:


# CELL 3: Track 1 — Ingest & Benchmark Continuous Walk-Forward Blotters


def calculate_blotter_metrics(df_blotter: pd.DataFrame) -> Dict[str, Any]:
    """Computes standardized portfolio performance metrics from an authoritative blotter."""
    b_df = df_blotter.copy()
    date_col = "date" if "date" in b_df.columns else "decision_date"
    b_df["date"] = pd.to_datetime(b_df[date_col])
    b_df = b_df.sort_values("date").reset_index(drop=True)

    daily_net = b_df["net_daily_simple_ret"].to_numpy(dtype=float)
    daily_bm = b_df["bm_daily_simple_ret"].to_numpy(dtype=float)
    daily_spread = daily_net - daily_bm

    # Strict Geometric Compounding Invariant
    cum_agent = float(np.prod(1.0 + daily_net) - 1.0)
    cum_bm = float(np.prod(1.0 + daily_bm) - 1.0)
    excess_ret = float(cum_agent - cum_bm)

    mean_ret = float(np.mean(daily_net))
    std_ret = float(np.std(daily_net, ddof=1)) if len(daily_net) > 1 else 1e-8
    sharpe = float((mean_ret / std_ret) * np.sqrt(252)) if std_ret > 1e-8 else 0.0

    mean_sp = float(np.mean(daily_spread))
    std_sp = float(np.std(daily_spread, ddof=1)) if len(daily_spread) > 1 else 1e-8
    ir = float((mean_sp / std_sp) * np.sqrt(252)) if std_sp > 1e-8 else 0.0

    downside = daily_spread[daily_spread < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 1e-8
    sortino = (
        float((mean_sp / downside_std) * np.sqrt(252)) if downside_std > 1e-8 else 0.0
    )

    eq = b_df["agent_equity"].to_numpy(dtype=float)
    peaks = np.maximum.accumulate(eq)
    dds = (eq - peaks) / np.maximum(peaks, 1e-8)
    max_dd = float(np.min(dds))

    bm_var = float(np.var(daily_bm, ddof=1)) if len(daily_bm) > 1 else 1e-8
    beta = float(np.cov(daily_net, daily_bm)[0, 1] / bm_var) if bm_var > 1e-8 else 1.0

    return {
        "Total Return": cum_agent,
        "BM Return": cum_bm,
        "Excess Return": excess_ret,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "IR": ir,
        "Max DD": max_dd,
        "Beta": beta,
        "Sessions": len(daily_net),
        "Start Date": pd.to_datetime(b_df["date"].iloc[0]).strftime("%Y-%m-%d"),
        "End Date": pd.to_datetime(b_df["date"].iloc[-1]).strftime("%Y-%m-%d"),
    }


def parse_blotter_sort_key(filename: str) -> Tuple[int, int, int, str]:
    """
    Deterministic hierarchical sort key:
      1. Generation number (ascending integer, e.g. gen25 -> 25)
      2. Tier rank:
         - 0: ex_post_blend (always lead constituent of the generation)
         - 1: constituent seeds (sorted numerically by seed ID)
         - 2: committee or other combination blotters
         - 3: unclassified / fallback
      3. Seed ID (numerical: 42, 101, 777, 1337, 2026)
      4. Canonical filename (tie-breaker)
    """
    gen_match = re.search(r"gen(\d+)", filename, re.IGNORECASE)
    gen_num = int(gen_match.group(1)) if gen_match else 9999

    fname_lower = filename.lower()
    if "ex_post_blend" in fname_lower:
        tier_rank = 0
        seed_num = 0
    else:
        seed_match = re.search(r"_s(\d+)(?:_|\.|$)", fname_lower)
        if seed_match:
            tier_rank = 1
            seed_num = int(seed_match.group(1))
        elif "committee" in fname_lower:
            tier_rank = 2
            seed_num = 0
        else:
            tier_rank = 3
            seed_num = 0

    return (gen_num, tier_rank, seed_num, filename)


# Ingest all walk-forward continuous blotters
continuous_blotters = sorted(list(OUTPUT_DIR.glob("blotter_continuous_*.parquet")))
print(f"🔍 Found {len(continuous_blotters)} Continuous Walk-Forward Blotters:")
wf_records = []
for bp in continuous_blotters:
    df_b = pd.read_parquet(bp)
    date_col = "date" if "date" in df_b.columns else "decision_date"
    df_b["date"] = pd.to_datetime(df_b[date_col])
    metrics = calculate_blotter_metrics(df_b)
    metrics["Blotter Name"] = bp.name
    wf_records.append(metrics)

df_wf = pd.DataFrame()
if wf_records:
    df_wf = pd.DataFrame(wf_records)
    print("\n" + "=" * 105)
    print("🏆 CONTINUOUS WALK-FORWARD LEADERBOARD")
    print("=" * 105)

    # Attach hierarchical sorting metadata
    sort_metadata = [parse_blotter_sort_key(name) for name in df_wf["Blotter Name"]]
    df_wf["_gen"] = [m[0] for m in sort_metadata]
    df_wf["_tier"] = [m[1] for m in sort_metadata]
    df_wf["_seed"] = [m[2] for m in sort_metadata]

    # Hierarchical Sort: Gen -> ex_post_blend first -> Seed ID numerically
    df_wf = df_wf.sort_values(
        by=["_gen", "_tier", "_seed", "Blotter Name"]
    ).reset_index(drop=True)

    summary_cols = [
        "Blotter Name",
        "Total Return",
        "BM Return",
        "Excess Return",
        "Sharpe",
        "Sortino",
        "IR",
        "Max DD",
        "Beta",
        "Sessions",
    ]
    display_df = df_wf[summary_cols].copy()
    display_df["Total Return"] = display_df["Total Return"].map(
        lambda x: f"{x*100:+.2f}%"
    )
    display_df["BM Return"] = display_df["BM Return"].map(lambda x: f"{x*100:+.2f}%")
    display_df["Excess Return"] = display_df["Excess Return"].map(
        lambda x: f"{x*100:+.2f}%"
    )
    display_df["Max DD"] = display_df["Max DD"].map(lambda x: f"{x*100:.2f}%")
    display_df["Sharpe"] = display_df["Sharpe"].map(lambda x: f"{x:.3f}")
    display_df["Sortino"] = display_df["Sortino"].map(lambda x: f"{x:.3f}")
    display_df["IR"] = display_df["IR"].map(lambda x: f"{x:.3f}")
    display_df["Beta"] = display_df["Beta"].map(lambda x: f"{x:.2f}")

    # Format across all rows to guarantee bit-level column alignment, then inject group spacers
    lines = display_df.to_string(index=False).splitlines()
    header = lines[0]
    data_lines = lines[1:]

    output_lines = [header]
    prev_gen = None
    for gen, line in zip(df_wf["_gen"], data_lines):
        if prev_gen is not None and gen != prev_gen:
            output_lines.append("")  # Visual demarcation between generation families
        output_lines.append(line)
        prev_gen = gen

    print("\n".join(output_lines))
else:
    print("⚠️ No continuous blotters detected in output directory.")


# In[4]:


# CELL 4: Dynamic Metadata Ingestion, Constituent & Ex-Post Ensemble Synthesizer

# ==============================================================================
# 🎛️ USER SELECTION: Specify Target Generation or Target Blotter Name
# - If TARGET_GEN is specified (e.g. 25), loads metadata and constituents for that gen.
# - If TARGET_BLOTTER is specified, parses target generation directly from filename.
# - If both are None, auto-selects the latest generation discovered in CELL 3.
# ==============================================================================
TARGET_GEN: Optional[int] = None
TARGET_BLOTTER: Optional[str] = None


def load_run_metadata(
    gen: int, output_path: Path, ckpt_path: Path
) -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
    """
    Dynamically searches and loads run_metadata.json across standardized candidate locations.
    """
    candidates = [
        ckpt_path / f"walk_forward_gen{gen}" / "run_metadata.json",
        output_path
        / "model_checkpoints"
        / f"walk_forward_gen{gen}"
        / "run_metadata.json",
        output_path / "canonical_anchors" / f"run_metadata_gen{gen}.json",
        output_path / f"run_metadata_gen{gen}.json",
        output_path / "run_metadata.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f), p
            except Exception as e:
                print(f"⚠️ Warning: Could not parse metadata at {p}: {e}")
    return None, None


# 1. Resolve Active Generation (GEN)
if TARGET_BLOTTER is not None:
    match = re.search(r"gen(\d+)", TARGET_BLOTTER, re.IGNORECASE)
    if match:
        GEN = int(match.group(1))
    else:
        raise ValueError(
            f"Could not parse generation integer from TARGET_BLOTTER: {TARGET_BLOTTER}"
        )
elif TARGET_GEN is not None:
    GEN = TARGET_GEN
else:
    if not df_wf.empty and "_gen" in df_wf.columns:
        valid_gens = [g for g in df_wf["_gen"].unique() if g != 9999]
        GEN = max(valid_gens) if valid_gens else 25
    else:
        GEN = 25

print("\n" + "=" * 80)
print(f"🔬 GENERATION {GEN} CONSTITUENT AUDIT & DIVERSIFICATION DECOMPOSITION")
print("=" * 80)

# 2. Ingest Metadata for Selected Generation
metadata, meta_path = load_run_metadata(GEN, OUTPUT_DIR, checkpoint_dir)

if metadata is not None:
    print(f"📄 Authoritative Metadata Ingested: {meta_path}")
    print(f"  • Timestamp UTC : {metadata.get('timestamp_utc', 'N/A')}")
    print(f"  • Audit Verdict : {metadata.get('verdict', 'N/A')}")

    # Extract Seeds
    if "seeds" in metadata and isinstance(metadata["seeds"], list):
        SEEDS = [int(s) for s in metadata["seeds"]]
    else:
        SEEDS = [42, 101, 777]

    # Extract Dynamic Multi-Chunk Windows
    raw_specs = metadata.get("chunk_specs", [])
    if raw_specs:
        CHUNK_WINDOWS = [
            {
                "chunk_id": int(spec.get("chunk_id", idx)),
                "deploy_start": str(spec["deploy_start"]),
                "deploy_end": (
                    str(spec["deploy_end"]) if spec.get("deploy_end") else None
                ),
                "name": spec.get("name", f"Chunk_{idx}"),
            }
            for idx, spec in enumerate(raw_specs)
            if spec.get("deploy_start")
        ]
    else:
        CHUNK_WINDOWS = []

    # Display Data Lineage
    lineage = metadata.get("data_lineage", {})
    if lineage:
        print(
            f"  • Cache Lineage : {lineage.get('cache_file')} ({lineage.get('cache_file_size_bytes', 0) / 1e6:.1f} MB)"
        )
        print(
            f"  • Universe / Dim: Universe={lineage.get('universe_size')} | Obs={lineage.get('obs_dim')} | Actions={lineage.get('action_dim')}"
        )
else:
    print(
        f"⚠️ run_metadata.json not found for Gen {GEN}. Deriving configuration from output directory..."
    )
    # Infer seeds from continuous blotters present in output
    seed_files = list(OUTPUT_DIR.glob(f"blotter_continuous_s*_gen{GEN}.parquet"))
    inferred_seeds = sorted(
        list(
            {
                int(m.group(1))
                for f in seed_files
                if (m := re.search(r"_s(\d+)(?:_|\.|$)", f.name))
            }
        )
    )
    SEEDS = inferred_seeds if inferred_seeds else [42, 101, 777]
    CHUNK_WINDOWS = [
        {"chunk_id": 0, "deploy_start": "2022-03-25", "deploy_end": "2023-03-24"},
        {"chunk_id": 1, "deploy_start": "2023-03-25", "deploy_end": "2024-03-24"},
        {"chunk_id": 2, "deploy_start": "2024-03-25", "deploy_end": "2025-03-24"},
        {"chunk_id": 3, "deploy_start": "2025-03-25", "deploy_end": "2026-03-24"},
        {"chunk_id": 4, "deploy_start": "2026-03-25", "deploy_end": None},
    ]

print(f"  • Active Constituent Seeds : {SEEDS}")
print(f"  • Total Walk-Forward Folds : {len(CHUNK_WINDOWS)}")
for cw in CHUNK_WINDOWS:
    end_str = cw["deploy_end"] if cw["deploy_end"] else "Terminal"
    print(f"    - Fold {cw['chunk_id']}: {cw['deploy_start']} -> {end_str}")

# 3. Load Constituent Seed Blotters
constituent_blotters: Dict[int, pd.DataFrame] = {}
print("\nConstituent Seed Performance:")
for s in SEEDS:
    seed_blotter_path = OUTPUT_DIR / f"blotter_continuous_s{s}_gen{GEN}.parquet"
    if not seed_blotter_path.exists():
        anchor_path = (
            OUTPUT_DIR
            / "canonical_anchors"
            / f"blotter_continuous_s{s}_gen{GEN}.parquet"
        )
        if anchor_path.exists():
            seed_blotter_path = anchor_path
        else:
            raise FileNotFoundError(
                f"❌ Missing constituent continuous blotter for Seed {s} (Gen {GEN}) at {seed_blotter_path}"
            )

    df_seed = pd.read_parquet(seed_blotter_path)
    date_col = "date" if "date" in df_seed.columns else "decision_date"
    df_seed["date"] = pd.to_datetime(df_seed[date_col])
    df_seed = df_seed.sort_values("date").reset_index(drop=True)
    constituent_blotters[s] = df_seed
    m = calculate_blotter_metrics(df_seed)
    print(
        f"  • Seed {s:>4}: Total Ret = {m['Total Return']*100:>+7.2f}% | Sharpe = {m['Sharpe']:>6.3f} | Beta = {m['Beta']:>4.2f} | Max DD = {m['Max DD']*100:>6.2f}% | Sessions = {m['Sessions']}"
    )

# 4. Synthesize or Load Ex-Post Blend Blotter
blend_path = OUTPUT_DIR / f"blotter_continuous_ex_post_blend_gen{GEN}.parquet"
ref_seed = SEEDS[0]
dates = constituent_blotters[ref_seed]["date"].to_numpy()
bm_rets = constituent_blotters[ref_seed]["bm_daily_simple_ret"].to_numpy(dtype=float)

ret_matrix = np.column_stack(
    [
        constituent_blotters[s]["net_daily_simple_ret"].to_numpy(dtype=float)
        for s in SEEDS
    ]
)
ex_post_rets = np.mean(ret_matrix, axis=1)

ex_post_equity = np.cumprod(1.0 + ex_post_rets)
bm_equity = np.cumprod(1.0 + bm_rets)
alpha_equity = ex_post_equity / np.maximum(bm_equity, 1e-8)

df_ex_post = pd.DataFrame(
    {
        "date": dates,
        "net_daily_simple_ret": ex_post_rets,
        "bm_daily_simple_ret": bm_rets,
        "alpha_daily_simple_ret": ex_post_rets - bm_rets,
        "agent_equity": ex_post_equity,
        "bm_equity": bm_equity,
        "alpha_equity": alpha_equity,
    }
)
df_ex_post.to_parquet(blend_path, index=False)

# 5. Mathematical Proof: Diversification Ratio & Covariance Decomposition
weights = np.ones(len(SEEDS)) / len(SEEDS)
individual_stds = np.std(ret_matrix, axis=0, ddof=1) * np.sqrt(252)
ensemble_std = float(np.std(ex_post_rets, ddof=1) * np.sqrt(252))
weighted_vol = float(np.dot(weights, individual_stds))
div_ratio = weighted_vol / (ensemble_std + 1e-8)

corr_matrix = pd.DataFrame(
    np.corrcoef(ret_matrix, rowvar=False),
    index=[f"Seed {s}" for s in SEEDS],
    columns=[f"Seed {s}" for s in SEEDS],
)

ep_metrics = calculate_blotter_metrics(df_ex_post)
constituent_sharpes = [
    calculate_blotter_metrics(constituent_blotters[s])["Sharpe"] for s in SEEDS
]
avg_constituent_sharpe = float(np.mean(constituent_sharpes))

print("\n" + "-" * 80)
print("📊 EX-POST ENSEMBLE COVARIANCE & RISK REDUCTION DECOMPOSITION:")
print("  • Pairwise Return Correlation Matrix:")
print(corr_matrix.to_string())
print(f"\n  • Constituent Weighted Annualized Volatility : {weighted_vol*100:.2f}%")
print(f"  • Realized Ex-Post Ensemble Annual Volatility: {ensemble_std*100:.2f}%")
print(
    f"  • Absolute Volatility Reduction              : -{(weighted_vol - ensemble_std)*100:.2f}%"
)
print(
    f"  • Diversification Ratio (DR)                 : {div_ratio:.4f} (Values > 1.0 prove structural diversification)"
)
print(f"  • Constituent Mean Sharpe                    : {avg_constituent_sharpe:.3f}")
print(
    f"  • Ex-Post Blended Sharpe                     : {ep_metrics['Sharpe']:.3f} (Gain: +{ep_metrics['Sharpe'] - avg_constituent_sharpe:+.3f})"
)
print(
    f"  • Blend Total Return                         : {ep_metrics['Total Return']*100:+.2f}% (Excess: {ep_metrics['Excess Return']*100:+.2f}%)"
)
print(
    f"  • Blend Max Drawdown                         : {ep_metrics['Max DD']*100:.2f}%"
)
print(f"  • Blend Realized Beta vs Benchmark           : {ep_metrics['Beta']:.4f}")
print("-" * 80)

# 6. Audit against Institutional Scorecard if present in metadata
if metadata and "institutional_scorecard" in metadata:
    card = metadata["institutional_scorecard"]
    print("\n🏛️ INSTITUTIONAL SCORECARD GATES (from run_metadata.json):")
    for gate_name, gate_info in card.items():
        status = "✅ PASS" if gate_info.get("pass") else "❌ FAIL"
        details = ", ".join(
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in gate_info.items()
            if k != "pass"
        )
        print(f"  • {gate_name:<25}: {status} ({details})")
    print("-" * 80)


#  ### CELL 5: Interactive Visual Master Chart (Dual-Generation Walk-Forward Comparison)

# In[5]:


# ==============================================================================
# 🎛️ USER CONFIGURATION: Select generations to compare (e.g. [24, 25] or [25])
# Default: None -> Auto-detects [GEN - 1, GEN] if both exist, otherwise [GEN]
# ==============================================================================
COMPARE_GENS: Optional[List[Union[int, str]]] = None
MAX_SEEDS_PER_GEN = 5

# Discover all generations present across continuous blotters
all_blotter_files = list(OUTPUT_DIR.glob("blotter_continuous_*.parquet"))
discovered_gens = sorted(
    list(
        {
            int(m.group(1))
            for f in all_blotter_files
            if (m := re.search(r"gen(\d+)", f.name, re.IGNORECASE))
        }
    )
)

if not discovered_gens:
    raise FileNotFoundError(f"❌ No continuous blotter files found in {OUTPUT_DIR}")

# Resolve target comparison generations
current_active_gen = globals().get("GEN", discovered_gens[-1])
if COMPARE_GENS is None or len(COMPARE_GENS) == 0:
    if (current_active_gen - 1) in discovered_gens:
        target_gens = [current_active_gen - 1, current_active_gen]
    else:
        target_gens = (
            discovered_gens[-2:] if len(discovered_gens) >= 2 else [current_active_gen]
        )
else:
    parsed_gens = []
    for item in COMPARE_GENS:
        val = (
            int(re.search(r"(\d+)", str(item)).group(1))
            if re.search(r"(\d+)", str(item))
            else None
        )
        if val is not None and val in discovered_gens and val not in parsed_gens:
            parsed_gens.append(val)
    target_gens = parsed_gens if parsed_gens else [current_active_gen]

print(
    f"🎯 Target Comparison Generations: {target_gens} (Discovered: {discovered_gens})"
)

# Visual Palettes: Gen 0 = Baseline (Cool Blue/Purple), Gen 1 = Challenger (Emerald/Amber)
GEN_THEMES = [
    {
        "blend_color": "#2563EB",  # Royal Blue
        "blend_width": 2.8,
        "blend_dash": "solid",
        "seed_dash": "dot",
        "seed_width": 1.2,
        "palette": ["#93C5FD", "#60A5FA", "#818CF8", "#A78BFA", "#C084FC"],
        "fill_color": "rgba(37, 99, 235, 0.08)",
    },
    {
        "blend_color": "#00CC96",  # Emerald Green
        "blend_width": 3.2,
        "blend_dash": "solid",
        "seed_dash": "dash",
        "seed_width": 1.3,
        "palette": ["#34D399", "#10B981", "#F59E0B", "#FB923C", "#F43F5E"],
        "fill_color": "rgba(0, 204, 150, 0.14)",
    },
]

# Initialize 3-Row Canvas
benchmark_sym = TradingConfig().benchmark
title_str = (
    f"🏆 Walk-Forward Performance & Risk Decomposition: Gen {target_gens[0]} vs Gen {target_gens[1]}"
    if len(target_gens) > 1
    else f"🏆 Walk-Forward Performance & Risk Decomposition: Gen {target_gens[0]}"
)

fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.45, 0.27, 0.28],
    subplot_titles=(
        f"📈 Cumulative Realized Performance vs Benchmark ({benchmark_sym})",
        "🔥 Strict Geometric Alpha Ratio (V_p / V_bm)",
        "📉 Underwater Drawdown Dynamics",
    ),
)

benchmark_plotted = False

# Iterate over selected generations
for gen_idx, gen in enumerate(target_gens):
    theme = GEN_THEMES[gen_idx % len(GEN_THEMES)]

    # 1. Discover Constituent Seeds
    seed_files = sorted(
        list(OUTPUT_DIR.glob(f"blotter_continuous_s*_gen{gen}.parquet"))
    )
    seed_map: Dict[int, pd.DataFrame] = {}
    for sf in seed_files:
        m = re.search(r"_s(\d+)(?:_|\.|$)", sf.name)
        if m:
            s_id = int(m.group(1))
            df_s = pd.read_parquet(sf)
            date_col = "date" if "date" in df_s.columns else "decision_date"
            df_s["date"] = pd.to_datetime(df_s[date_col])
            seed_map[s_id] = df_s.sort_values("date").reset_index(drop=True)

    active_seeds = sorted(seed_map.keys())[:MAX_SEEDS_PER_GEN]

    # Plot Benchmark Once from earliest valid series
    if not benchmark_plotted and active_seeds:
        ref_df = seed_map[active_seeds[0]]
        dates = ref_df["date"]
        bm_curve = ref_df["bm_equity"].to_numpy(dtype=float)
        bm_peaks = np.maximum.accumulate(bm_curve)
        bm_dds = (bm_curve - bm_peaks) / np.maximum(bm_peaks, 1e-8)

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=bm_curve,
                mode="lines",
                name=f"Benchmark ({benchmark_sym})",
                line=dict(color="#111827", width=2.0, dash="dash"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=np.ones(len(dates)),
                mode="lines",
                name="Benchmark Parity (1.0)",
                line=dict(color="#111827", width=1.0, dash="dot"),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=bm_dds,
                mode="lines",
                name="Benchmark DD",
                line=dict(color="#6B7280", width=1.2, dash="dash"),
            ),
            row=3,
            col=1,
        )
        benchmark_plotted = True

    # 2. Plot Constituent Seeds
    for s_idx, s in enumerate(active_seeds):
        df_s = seed_map[s]
        m = calculate_blotter_metrics(df_s)
        s_eq = df_s["agent_equity"].to_numpy(dtype=float)
        s_alpha = df_s["alpha_equity"].to_numpy(dtype=float)
        s_peaks = np.maximum.accumulate(s_eq)
        s_dds = (s_eq - s_peaks) / np.maximum(s_peaks, 1e-8)
        color = theme["palette"][s_idx % len(theme["palette"])]

        fig.add_trace(
            go.Scatter(
                x=df_s["date"],
                y=s_eq,
                mode="lines",
                name=f"Gen {gen} s{s} (Sh: {m['Sharpe']:.2f})",
                line=dict(
                    color=color, width=theme["seed_width"], dash=theme["seed_dash"]
                ),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df_s["date"],
                y=s_alpha,
                mode="lines",
                line=dict(
                    color=color, width=theme["seed_width"], dash=theme["seed_dash"]
                ),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df_s["date"],
                y=s_dds,
                mode="lines",
                line=dict(color=color, width=0.8, dash=theme["seed_dash"]),
                showlegend=False,
            ),
            row=3,
            col=1,
        )

    # 3. Plot Ex-Post Blend Blotter
    blend_file = OUTPUT_DIR / f"blotter_continuous_ex_post_blend_gen{gen}.parquet"
    if blend_file.exists():
        df_b = pd.read_parquet(blend_file)
        date_col = "date" if "date" in df_b.columns else "decision_date"
        df_b["date"] = pd.to_datetime(df_b[date_col])
        df_b = df_b.sort_values("date").reset_index(drop=True)
        mb = calculate_blotter_metrics(df_b)

        b_eq = df_b["agent_equity"].to_numpy(dtype=float)
        b_alpha = df_b["alpha_equity"].to_numpy(dtype=float)
        b_peaks = np.maximum.accumulate(b_eq)
        b_dds = (b_eq - b_peaks) / np.maximum(b_peaks, 1e-8)

        is_challenger = gen_idx == (len(target_gens) - 1)
        fig.add_trace(
            go.Scatter(
                x=df_b["date"],
                y=b_eq,
                mode="lines",
                name=f"⭐ Gen {gen} Blend (Sh: {mb['Sharpe']:.2f} | Excess: {mb['Excess Return']*100:+.1f}%)",
                line=dict(
                    color=theme["blend_color"],
                    width=theme["blend_width"],
                    dash=theme["blend_dash"],
                ),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df_b["date"],
                y=b_alpha,
                mode="lines",
                line=dict(
                    color=theme["blend_color"],
                    width=theme["blend_width"] - 0.5,
                    dash=theme["blend_dash"],
                ),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df_b["date"],
                y=b_dds,
                mode="lines",
                fill="tozeroy" if is_challenger else None,
                fillcolor=theme["fill_color"] if is_challenger else None,
                name=f"Gen {gen} Blend DD",
                line=dict(
                    color=theme["blend_color"], width=1.8, dash=theme["blend_dash"]
                ),
            ),
            row=3,
            col=1,
        )

# 4. Draw Dynamic Walk-Forward Chunk Boundaries from Metadata
active_chunk_windows = globals().get("CHUNK_WINDOWS", [])
if not active_chunk_windows and target_gens:
    lead_gen = target_gens[-1]
    meta, _ = load_run_metadata(lead_gen, OUTPUT_DIR, checkpoint_dir)
    if meta and "chunk_specs" in meta:
        active_chunk_windows = [
            {
                "chunk_id": int(c.get("chunk_id", idx)),
                "deploy_start": str(c["deploy_start"]),
            }
            for idx, c in enumerate(meta["chunk_specs"])
            if c.get("deploy_start")
        ]

if active_chunk_windows:
    for win in active_chunk_windows[1:]:
        d_val = win["deploy_start"]
        fig.add_vline(x=d_val, line_dash="dash", line_color="#9CA3AF", opacity=0.7)
        fig.add_annotation(
            x=d_val,
            y=0.98,
            xref="x",
            yref="paper",
            text=f"Chunk {win['chunk_id']}",
            showarrow=False,
            xanchor="left",
            font=dict(size=10, color="#6B7280"),
        )

fig.update_layout(
    height=1100,
    width=1280,
    hovermode="x unified",
    margin=dict(t=150, b=50, l=70, r=50),
    title=dict(
        text=title_str,
        y=0.97,
        x=0.5,
        xanchor="center",
        yanchor="top",
        font=dict(size=16, color="#111827"),
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
        font=dict(size=10),
        bgcolor="rgba(255, 255, 255, 0.92)",
        bordercolor="rgba(0, 0, 0, 0.12)",
        borderwidth=1,
    ),
)
fig.update_yaxes(title_text="Wealth Multiplier", row=1, col=1)
fig.update_yaxes(title_text="Alpha (V_p / V_bm)", row=2, col=1)
fig.update_yaxes(title_text="Drawdown %", tickformat=".0%", row=3, col=1)
fig.show()


# In[6]:


# CELL 6: Track 2 — Batch Evaluator for Standalone Grid-Sweep Checkpoints
master_csv_path = rl_root / "Grid_sweep.csv"

all_pts = sorted(
    [p for p in checkpoint_dir.rglob("*.pt") if "walk_forward" not in p.parts]
)
print(f"\n🔍 Found {len(all_pts)} standalone checkpoints for Track 2 evaluation.")
active_gen = globals().get("GEN", 25)
print(
    f"👉 Master Champion Blotter Active: blotter_continuous_ex_post_blend_gen{active_gen}.parquet"
)
print("👉 Proceed to Notebook 03b for Deep Regime Attribution & Factor Diagnostics.")


# In[7]:


# Drop into a new cell after CELL 5 to monitor dynamic beta drift:
window = 63  # 1 quarter rolling
df_b = constituent_blotters[SEEDS[0]]  # or df_ex_post
ret_p = df_ex_post["net_daily_simple_ret"]
ret_bm = df_ex_post["bm_daily_simple_ret"]

cov = ret_p.rolling(window).cov(ret_bm)
var_bm = ret_bm.rolling(window).var()
rolling_beta = cov / var_bm
spread = ret_p - ret_bm
rolling_ir = (spread.rolling(window).mean() / spread.rolling(window).std()) * np.sqrt(
    252
)

fig_beta = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    subplot_titles=("Rolling 63d Beta vs Benchmark", "Rolling 63d Information Ratio"),
)
fig_beta.add_trace(
    go.Scatter(
        x=df_ex_post["date"],
        y=rolling_beta,
        name="Blend Beta",
        line=dict(color="#00CC96", width=2),
    ),
    row=1,
    col=1,
)
fig_beta.add_hline(y=1.25, line_dash="dash", line_color="red", row=1, col=1)
fig_beta.add_hline(y=0.90, line_dash="dash", line_color="red", row=1, col=1)
fig_beta.add_trace(
    go.Scatter(
        x=df_ex_post["date"],
        y=rolling_ir,
        name="Rolling IR",
        line=dict(color="#2563EB", width=2),
    ),
    row=2,
    col=1,
)
fig_beta.update_layout(
    height=600, width=1280, title=f"Gen {GEN} Dynamic Beta & IR Stability"
)
fig_beta.show()


# In[8]:


fig_corr = px.imshow(
    corr_matrix,
    text_auto=".3f",
    color_continuous_scale="Blues",
    title=f"Gen {GEN} Pairwise Constituent Return Correlation Matrix (DR = {div_ratio:.4f})",
    aspect="auto",
    width=650,
    height=500,
)
fig_corr.show()


# In[9]:


# Available if evaluating constituent continuous blotters with column 'weight_active':
fig_tilt = go.Figure()
for s in SEEDS:
    df_s = constituent_blotters[s]
    if "weight_active" in df_s.columns:
        fig_tilt.add_trace(
            go.Scatter(
                x=df_s["date"],
                y=df_s["weight_active"],
                mode="lines",
                name=f"Seed {s} w_active",
                line=dict(width=1),
            )
        )
fig_tilt.update_layout(
    height=400,
    width=1280,
    title=f"Gen {GEN} Dynamic Active Tilt (w_active) across Walk-Forward History",
)
fig_tilt.show()


# In[10]:


def get_oos_date_range(metadata: dict) -> tuple[str, str, int]:
    """Extracts (oos_start, oos_end, total_sessions) from run_metadata.json."""
    specs = metadata.get("chunk_specs", [])
    oos_start = specs[0]["deploy_start"] if specs else None

    # Terminal fold end: fallback to calendar_end if deploy_end is null/open-ended
    last_spec_end = specs[-1].get("deploy_end") if specs else None
    oos_end = last_spec_end or metadata.get("data_lineage", {}).get("calendar_end")

    total_sessions = metadata.get("blend_metrics", {}).get("sessions", 0)
    return oos_start, oos_end, total_sessions


oos_start, oos_end, n_sessions = get_oos_date_range(metadata)
print(f"📅 Authoritative OOS Range: {oos_start} to {oos_end} ({n_sessions} sessions)")
# -> 📅 Authoritative OOS Range: 2022-03-25 to 2026-09-25 (1120 sessions)


# In[11]:


# %%
# Visualizing Walk-Forward Training, Hist_Cutoff, and OOS Deploy Topology
import json
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from core.paths import OUTPUT_DIR

# 1. Ingest Chunk Metadata
metadata_path = (
    OUTPUT_DIR / "model_checkpoints" / "walk_forward_gen25" / "run_metadata.json"
)
if not metadata_path.exists():
    metadata_path = OUTPUT_DIR / "run_metadata.json"

if metadata_path.exists():
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    raw_specs = meta.get("chunk_specs", [])
    cal_end = meta.get("data_lineage", {}).get("calendar_end", "2026-09-25")
else:
    # Fallback to Gen 25 canonical specifications
    cal_end = "2026-09-25"
    raw_specs = [
        {
            "chunk_id": 0,
            "name": "Chunk_0_Base_Pretraining",
            "train_start": "2016-01-04",
            "train_end": "2022-03-24",
            "hist_cutoff_date": None,
            "deploy_start": "2022-03-25",
            "deploy_end": "2023-03-24",
            "is_base": True,
        },
        {
            "chunk_id": 1,
            "name": "Chunk_1_FineTune_Y1",
            "train_start": "2016-01-04",
            "train_end": "2023-03-24",
            "hist_cutoff_date": "2022-03-24",
            "deploy_start": "2023-03-25",
            "deploy_end": "2024-03-24",
            "is_base": False,
        },
        {
            "chunk_id": 2,
            "name": "Chunk_2_FineTune_Y2",
            "train_start": "2016-01-04",
            "train_end": "2024-03-24",
            "hist_cutoff_date": "2023-03-24",
            "deploy_start": "2024-03-25",
            "deploy_end": "2025-03-24",
            "is_base": False,
        },
        {
            "chunk_id": 3,
            "name": "Chunk_3_FineTune_Y3",
            "train_start": "2016-01-04",
            "train_end": "2025-03-24",
            "hist_cutoff_date": "2024-03-24",
            "deploy_start": "2025-03-25",
            "deploy_end": "2026-03-24",
            "is_base": False,
        },
        {
            "chunk_id": 4,
            "name": "Chunk_4_FineTune_Y4",
            "train_start": "2016-01-04",
            "train_end": "2026-03-24",
            "hist_cutoff_date": "2025-03-24",
            "deploy_start": "2026-03-25",
            "deploy_end": None,
            "is_base": False,
        },
    ]

# 2. Build Gantt Records
records = []
for spec in raw_specs:
    cid = spec["chunk_id"]
    name = f"Chunk {cid}"
    t_start = spec["train_start"]
    t_end = spec["train_end"]
    cutoff = spec.get("hist_cutoff_date")
    d_start = spec["deploy_start"]
    d_end = spec["deploy_end"] if spec.get("deploy_end") else cal_end

    if cutoff is None:
        # Pure Base Pretraining
        records.append(
            {
                "Chunk": name,
                "Type": "Base Training (Deep History)",
                "Start": t_start,
                "End": t_end,
                "Color": "#1E3A8A",  # Deep Navy
            }
        )
    else:
        # Pre-cutoff Distant Replay
        records.append(
            {
                "Chunk": name,
                "Type": "Distant History Replay (<= hist_cutoff)",
                "Start": t_start,
                "End": cutoff,
                "Color": "#3B82F6",  # Medium Blue
            }
        )
        # Post-cutoff Stratified Focus
        records.append(
            {
                "Chunk": name,
                "Type": "Recent Regime Training (> hist_cutoff)",
                "Start": cutoff,
                "End": t_end,
                "Color": "#93C5FD",  # Light Sky Blue
            }
        )

    # Out-of-Sample Deployment
    records.append(
        {
            "Chunk": name,
            "Type": "Out-of-Sample (OOS) Live Deployment",
            "Start": d_start,
            "End": d_end,
            "Color": "#10B981",  # Vivid Emerald
        }
    )

df_gantt = pd.DataFrame(records)

# 3. Construct Interactive Plotly Timeline
fig = go.Figure()
seen_types = set()

# Reverse display so Chunk 0 is at top, Chunk 4 is at bottom
chunk_order = [f"Chunk {i}" for i in reversed(range(len(raw_specs)))]

for _, row in df_gantt.iterrows():
    show_leg = row["Type"] not in seen_types
    seen_types.add(row["Type"])

    t_start = pd.to_datetime(row["Start"])
    t_end = pd.to_datetime(row["End"])
    duration_ms = (t_end - t_start).total_seconds() * 1000

    fig.add_trace(
        go.Bar(
            y=[row["Chunk"]],
            x=[duration_ms],
            base=[t_start],
            orientation="h",
            name=row["Type"],
            marker=dict(
                color=row["Color"],
                line=dict(color="#111827", width=0.8),
            ),
            showlegend=show_leg,
            hovertemplate=(
                f"<b>{row['Chunk']}</b><br>"
                f"Stage: {row['Type']}<br>"
                f"Window: {row['Start']} to {row['End']}<extra></extra>"
            ),
        )
    )

# 4. Highlight the Authoritative Continuous OOS Window
oos_master_start = pd.to_datetime(raw_specs[0]["deploy_start"])
oos_master_end = pd.to_datetime(cal_end)

fig.add_vrect(
    x0=oos_master_start,
    x1=oos_master_end,
    fillcolor="rgba(16, 185, 129, 0.08)",
    layer="below",
    line_width=1.5,
    line_dash="dash",
    line_color="#059669",
    annotation_text="Authoritative Continuous OOS Window (1,120 Sessions | 2022-03-25 -> 2026-09-25)",
    annotation_position="top left",
    annotation_font=dict(size=12, color="#065F46", family="monospace"),
)

fig.update_layout(
    title=dict(
        text="🧩 Walk-Forward Chronology: Training Regimes, hist_cutoff, & OOS Deployment (Gen 25)",
        x=0.5,
        xanchor="center",
        font=dict(size=16, color="#111827"),
    ),
    barmode="stack",
    xaxis=dict(
        type="date",
        title="Calendar Date",
        tickformat="%Y-%m",
        showgrid=True,
        gridcolor="#E5E7EB",
    ),
    yaxis=dict(
        title="Walk-Forward Folds",
        categoryorder="array",
        categoryarray=chunk_order,
    ),
    height=480,
    width=1280,
    margin=dict(t=80, b=50, l=100, r=40),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.03,
        xanchor="center",
        x=0.5,
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="#E5E7EB",
        borderwidth=1,
    ),
)
fig.show()
# %%

