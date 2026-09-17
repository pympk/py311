"""
Generation 17: Walk-Forward Multi-Seed Committee Engine Orchestrator.
Executes expanding walk-forward horizon fine-tuning (Chunks 0..3),
analytical behavioral anchoring, multi-seed committee ensembling,
and continuous geometric blotter stitching.
"""

import copy
import gc
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from core.paths import LOCAL_DATA_DIR, OUTPUT_DIR
from core.quant import QuantUtils
from core.settings import CacheConfig, TradingConfig
from data_pipeline.loader import load_processed_data
from data_pipeline.utils import get_master_trading_calendar
from rl_discovery.adapter import (
    ObservationScaler,
    RLVRGymEnv,
    make_eval_env,
    make_stratified_train_envs,
)
from rl_discovery.agent import AbsoluteZeroAgent, CommitteeAgent
from rl_discovery.trainer import PPOTrainer, RolloutBuffer
from rl_discovery.validator import AgentEvaluator

# Walk-Forward Horizon Definition Contract (Generation 17)
CHUNK_SPECS = [
    {
        "chunk_id": 0,
        "name": "Chunk_0_Base_Pretraining",
        "train_start": "2016-01-04",
        "train_end": "2022-03-24",
        "hist_cutoff_date": None,
        "deploy_start": "2022-03-25",
        "deploy_end": "2023-03-24",
        "is_base": True,
        "epochs": 90,
        "warmup_epochs": 25,
        "min_epochs": 70,
        "patience": 30,
        "lr": 2.0e-4,
        "critic_lr": 8.0e-4,
        "kl_anchor_coef": 0.0,
        "stratified_sampling": False,
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
        "epochs": 15,
        "warmup_epochs": 0,
        "min_epochs": 15,
        "patience": 15,
        "lr": 2.0e-5,
        "critic_lr": 8.0e-5,
        "kl_anchor_coef": 0.05,
        "stratified_sampling": True,
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
        "epochs": 15,
        "warmup_epochs": 0,
        "min_epochs": 15,
        "patience": 15,
        "lr": 2.0e-5,
        "critic_lr": 8.0e-5,
        "kl_anchor_coef": 0.05,
        "stratified_sampling": True,
    },
    {
        "chunk_id": 3,
        "name": "Chunk_3_FineTune_Y3",
        "train_start": "2016-01-04",
        "train_end": "2025-03-24",
        "hist_cutoff_date": "2024-03-24",
        "deploy_start": "2025-03-25",
        "deploy_end": "2026-09-09",
        "is_base": False,
        "epochs": 15,
        "warmup_epochs": 0,
        "min_epochs": 15,
        "patience": 15,
        "lr": 2.0e-5,
        "critic_lr": 8.0e-5,
        "kl_anchor_coef": 0.05,
        "stratified_sampling": True,
    },
]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_single_seed_chunk(
    seed: int,
    chunk_spec: Dict[str, Any],
    config: TradingConfig,
    feature_cube: pd.DataFrame,
    simple_ret_matrix: pd.DataFrame,
    macro_df: pd.DataFrame,
    cal_train: pd.DatetimeIndex,
    cal_val: pd.DatetimeIndex,
    hist_cutoff_idx: Optional[int],
    prior_checkpoint_path: Optional[Path],
    device: torch.device,
    checkpoint_dir: Path,
) -> Tuple[Path, Dict[str, Any]]:
    set_seed(seed)
    chunk_id = chunk_spec["chunk_id"]
    obs_dim = 3 * feature_cube.shape[1] + len(macro_df.columns)
    action_dim = feature_cube.shape[1] + 4

    agent = AbsoluteZeroAgent(obs_dim=obs_dim, action_dim=action_dim).to(device)
    anchor_agent: Optional[AbsoluteZeroAgent] = None
    initial_scaler_state: Optional[Dict[str, Any]] = None

    # Checkpoint Warm-Starting & Behavioral Anchor Setup
    if prior_checkpoint_path is not None and prior_checkpoint_path.exists():
        print(f"      [Warm-Start] Loading prior weights: {prior_checkpoint_path.name}")
        payload = torch.load(
            prior_checkpoint_path, map_location=device, weights_only=False
        )
        agent.load_state_dict(payload["model_state_dict"])
        initial_scaler_state = payload.get("scaler_state", None)

        if chunk_spec["kl_anchor_coef"] > 0.0:
            anchor_agent = AbsoluteZeroAgent(obs_dim=obs_dim, action_dim=action_dim).to(
                device
            )
            anchor_agent.load_state_dict(payload["model_state_dict"])
            anchor_agent.eval()
            for p in anchor_agent.parameters():
                p.requires_grad = False
            print(
                f"      [Behavioral Anchor] Initialized from prior chunk with beta={chunk_spec['kl_anchor_coef']}"
            )

    # Vectorized Stratified Training Environments
    num_envs = getattr(config, "num_envs", 8)
    num_steps = getattr(config, "num_steps", 512)
    mini_batch_size = getattr(config, "mini_batch_size", 256)

    envs = make_stratified_train_envs(
        feature_cube=feature_cube,
        simple_ret_matrix=simple_ret_matrix,
        calendar=cal_train,
        macro_df=macro_df,
        config=config,
        num_envs=num_envs,
        episode_steps=num_steps,
        hist_cutoff_idx=hist_cutoff_idx,
        initial_scaler_state=initial_scaler_state,
    )

    # Deterministic Validation Environment (Unbounded execution: episode_steps=0)
    gym_val = make_eval_env(
        feature_cube=feature_cube,
        simple_ret_matrix=simple_ret_matrix,
        calendar=cal_val,
        macro_df=macro_df,
        config=config,
        scaler_state=initial_scaler_state,
    )

    trainer = PPOTrainer(
        agent=agent,
        lr=chunk_spec["lr"],
        critic_lr=chunk_spec["critic_lr"],
        clip_coef=config.clip_coef,
        clip_vloss=True,
        ent_coef=config.entropy_coef_start,
        anchor_agent=anchor_agent,
        kl_anchor_coef=chunk_spec["kl_anchor_coef"],
    )

    buffer = RolloutBuffer(
        num_steps=num_steps,
        num_envs=num_envs,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )

    best_fitness = -np.inf
    best_checkpoint_path = checkpoint_dir / f"model_chunk{chunk_id}_s{seed}_champion.pt"
    best_scaler_state: Dict[str, Any] = {}
    patience_counter = 0

    total_epochs = chunk_spec["epochs"]
    warmup_epochs = chunk_spec["warmup_epochs"]
    min_epochs = chunk_spec["min_epochs"]
    patience = chunk_spec["patience"]

    for epoch in range(total_epochs):
        obs, _ = envs.reset()
        epoch_rewards = []

        for step in range(num_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs_tensor)

            next_obs, rewards, terminations, truncations, _ = envs.step(
                action.cpu().numpy()
            )
            scaled_rewards = rewards * 25.0

            buffer.add(
                obs=obs,
                action=action,
                logprob=logprob,
                reward=scaled_rewards,
                value=value,
                terminations=terminations,
                truncations=truncations,
            )
            epoch_rewards.append(float(np.mean(rewards)))
            obs = next_obs

        # Bootstrap Advantage at Horizon
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
        next_term_tensor = torch.tensor(
            terminations, dtype=torch.float32, device=device
        )
        with torch.no_grad():
            next_values = agent.get_value(obs_tensor)
        buffer.compute_advantages(next_values, next_termination=next_term_tensor)

        trainer.update_schedules(
            current_epoch=epoch + 1,
            total_epochs=total_epochs,
            ent_start=config.entropy_coef_start,
            ent_end=config.entropy_coef_end,
        )

        diag = trainer.update(
            buffer=buffer,
            update_epochs=config.ppo_epochs,
            mini_batch_size=mini_batch_size,
            stratified_sampling=chunk_spec["stratified_sampling"],
        )
        buffer.step = 0

        # Periodic Validation Checkpoint Gate
        train_scaler = envs.get_attr("scaler")[0]
        gym_val.scaler.load_state(train_scaler)
        val_res = AgentEvaluator.evaluate(agent, gym_val, device=device)

        fitness = QuantUtils.compute_composite_fitness(
            excess_return=val_res["excess_return"],
            information_ratio=val_res["information_ratio"],
        )

        is_warmup = (epoch + 1) <= warmup_epochs
        if not is_warmup and fitness > best_fitness:
            best_fitness = fitness
            patience_counter = 0
            best_scaler_state = {
                "mean": train_scaler.mean.copy(),
                "var": train_scaler.var.copy(),
                "count": train_scaler.count,
            }
            torch.save(
                {
                    "model_state_dict": agent.state_dict(),
                    "scaler_state": best_scaler_state,
                    "val_fitness": fitness,
                    "val_sharpe": val_res["sharpe_ratio"],
                    "val_excess": val_res["excess_return"],
                    "val_ir": val_res["information_ratio"],
                    "epoch": epoch + 1,
                    "seed": seed,
                    "chunk_id": chunk_id,
                    "grid_params": chunk_spec,
                    "holding_period": config.holding_period,
                    "benchmark": config.benchmark,
                },
                best_checkpoint_path,
            )
        else:
            patience_counter += 1

        if (epoch + 1) >= min_epochs and patience_counter >= patience:
            print(
                f"      [Early Stop] Seed {seed} stopped at Epoch {epoch+1} (Best Fitness: {best_fitness:.4f})"
            )
            break

    envs.close()
    del envs, gym_val, buffer, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_checkpoint_path, best_scaler_state


def stitch_continuous_blotters(
    chunk_blotters: List[List[Dict[str, Any]]],
) -> pd.DataFrame:
    """
    Stitches multiple sequential deployment blotters using strict geometric compounding:
        V_p(t) = V_p(T_chunk_end) * prod_{tau=T_start}^t (1 + r_{net, tau})
    """
    all_rows = []
    for blotter in chunk_blotters:
        all_rows.extend(blotter)

    df_stitched = (
        pd.DataFrame(all_rows)
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    net_rets = df_stitched["net_daily_simple_ret"].to_numpy(dtype=float)
    bm_rets = df_stitched["bm_daily_simple_ret"].to_numpy(dtype=float)

    p_curve = np.cumprod(1.0 + net_rets)
    bm_curve = np.cumprod(1.0 + bm_rets)
    alpha_curve = p_curve / np.maximum(bm_curve, 1e-8)

    df_stitched["agent_equity"] = p_curve
    df_stitched["bm_equity"] = bm_curve
    df_stitched["alpha_equity"] = alpha_curve

    return df_stitched


def compute_institutional_metrics(df_blotter: pd.DataFrame) -> Dict[str, float]:
    """Computes authoritative risk/return metrics from a continuous blotter DataFrame."""
    net_arr = df_blotter["net_daily_simple_ret"].to_numpy(dtype=float)
    bm_arr = df_blotter["bm_daily_simple_ret"].to_numpy(dtype=float)
    eq_arr = df_blotter["agent_equity"].to_numpy(dtype=float)
    bm_eq_arr = df_blotter["bm_equity"].to_numpy(dtype=float)

    total_return = float(eq_arr[-1] - 1.0) if len(eq_arr) > 0 else 0.0
    bm_total_return = float(bm_eq_arr[-1] - 1.0) if len(bm_eq_arr) > 0 else 0.0
    excess_return = float(total_return - bm_total_return)

    mean_ret = float(np.mean(net_arr)) if len(net_arr) > 0 else 0.0
    std_ret = float(np.std(net_arr, ddof=1)) if len(net_arr) > 1 else 0.0
    sharpe = float((mean_ret / std_ret) * np.sqrt(252.0)) if std_ret > 1e-8 else 0.0

    spread = net_arr - bm_arr
    mean_spread = float(np.mean(spread)) if len(spread) > 0 else 0.0
    std_spread = float(np.std(spread, ddof=1)) if len(spread) > 1 else 0.0
    info_ratio = (
        float((mean_spread / std_spread) * np.sqrt(252.0)) if std_spread > 1e-8 else 0.0
    )

    downside = spread[spread < 0.0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
    sortino = (
        float((mean_spread / downside_std) * np.sqrt(252.0))
        if downside_std > 1e-8
        else 0.0
    )

    running_max = np.maximum.accumulate(eq_arr)
    drawdowns = (eq_arr - running_max) / np.maximum(running_max, 1e-8)
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    bm_var = float(np.var(bm_arr, ddof=1)) if len(bm_arr) > 1 else 0.0
    if bm_var > 1e-8 and len(net_arr) > 1:
        cov_mat = np.cov(net_arr, bm_arr)
        beta = float(cov_mat[0, 1] / bm_var)
    else:
        beta = 1.0

    return {
        "total_return": total_return,
        "bm_total_return": bm_total_return,
        "excess_return": excess_return,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "information_ratio": info_ratio,
        "max_drawdown": max_dd,
        "beta": beta,
        "sessions": len(net_arr),
    }


def run_walk_forward_pipeline(seeds: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Main Entrypoint: Runs the 4-Chunk Expanding Walk-Forward Multi-Seed Committee Engine.
    """
    seeds = seeds or [42, 101, 777]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Walk-Forward Committee Engine (Seeds: {seeds}) on {device}")

    # 1. Load Preprocessed Data and AlphaCache
    data = load_processed_data()
    df_ohlcv = data.df_ohlcv
    macro_df = data.macro_df
    config = TradingConfig()
    df_close = df_ohlcv["Adj Close"].unstack(level=0).sort_index()
    master_cal = get_master_trading_calendar(df_ohlcv, config.calendar_ticker)

    cache_file = LOCAL_DATA_DIR / CacheConfig.get_filename()
    feature_cube = pd.read_parquet(cache_file)
    valid_dates = set(feature_cube.index.get_level_values("Date").unique())
    trading_calendar = master_cal[master_cal.isin(valid_dates)]

    simple_ret_matrix = df_close.pct_change(1, fill_method=None).shift(-1)
    simple_ret_matrix["CASH"] = 0.0

    checkpoint_dir = OUTPUT_DIR / "model_checkpoints" / "walk_forward_gen17"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    prior_checkpoints: Dict[int, Optional[Path]] = {s: None for s in seeds}
    prior_scalers: Dict[int, Optional[Dict[str, Any]]] = {s: None for s in seeds}

    committee_chunk_blotters: List[List[Dict[str, Any]]] = []
    seed_chunk_blotters: Dict[int, List[List[Dict[str, Any]]]] = {s: [] for s in seeds}

    obs_dim = 3 * feature_cube.shape[1] + len(macro_df.columns)
    action_dim = feature_cube.shape[1] + 4

    # 2. Iterate Across Expanding Walk-Forward Chunks
    for chunk in CHUNK_SPECS:
        cid = chunk["chunk_id"]
        cname = chunk["name"]
        t_start = pd.Timestamp(chunk["train_start"])
        t_end = pd.Timestamp(chunk["train_end"])
        d_start = pd.Timestamp(chunk["deploy_start"])
        d_end = pd.Timestamp(chunk["deploy_end"])

        # Partition Calendars
        cal_train_full = trading_calendar[
            (trading_calendar >= t_start) & (trading_calendar <= t_end)
        ]
        # Drop leakage buffer of holding_period days from training boundary
        hp = config.holding_period
        cal_train = cal_train_full[:-hp] if len(cal_train_full) > hp else cal_train_full

        cal_deploy = trading_calendar[
            (trading_calendar >= d_start) & (trading_calendar <= d_end)
        ]

        if len(cal_deploy) == 0:
            print(
                f"⚠️ [Chunk {cid}] Deployment window is empty in current data. Skipping."
            )
            continue

        # Determine Historical vs Recent Stratification Index
        hist_cutoff_idx: Optional[int] = None
        if chunk["hist_cutoff_date"] is not None:
            cutoff_ts = pd.Timestamp(chunk["hist_cutoff_date"])
            hist_matches = np.where(cal_train <= cutoff_ts)[0]
            if len(hist_matches) > 0:
                hist_cutoff_idx = int(hist_matches[-1])

        # Validation Window (Last 252 days of training window)
        val_start_idx = max(0, len(cal_train) - 252)
        cal_val = cal_train[val_start_idx:]

        print(f"\n{'='*75}")
        print(
            f"🔄 EXECUTING CHUNK {cid}: {cname} | Train: {len(cal_train)}d | Deploy: {len(cal_deploy)}d"
        )
        print(f"{'='*75}")

        # Train / Fine-tune Each Seed
        chunk_seed_agents: Dict[int, AbsoluteZeroAgent] = {}
        for seed in seeds:
            print(f"   ⚙️ Seed {seed}: Training Chunk {cid}...")
            ckpt_path, scaler_state = train_single_seed_chunk(
                seed=seed,
                chunk_spec=chunk,
                config=config,
                feature_cube=feature_cube,
                simple_ret_matrix=simple_ret_matrix,
                macro_df=macro_df,
                cal_train=cal_train,
                cal_val=cal_val,
                hist_cutoff_idx=hist_cutoff_idx,
                prior_checkpoint_path=prior_checkpoints[seed],
                device=device,
                checkpoint_dir=checkpoint_dir,
            )
            prior_checkpoints[seed] = ckpt_path
            prior_scalers[seed] = scaler_state

            # Instantiate Seed Agent for Evaluation
            agent_s = AbsoluteZeroAgent(obs_dim=obs_dim, action_dim=action_dim).to(
                device
            )
            payload = torch.load(ckpt_path, map_location=device, weights_only=False)
            agent_s.load_state_dict(payload["model_state_dict"])
            agent_s.eval()
            chunk_seed_agents[seed] = agent_s

            # Evaluate Seed Model Individually on Deployment Window
            gym_deploy_seed = make_eval_env(
                feature_cube=feature_cube,
                simple_ret_matrix=simple_ret_matrix,
                calendar=cal_deploy,
                macro_df=macro_df,
                config=config,
                scaler_state=scaler_state,
            )
            eval_seed_res = AgentEvaluator.evaluate(
                agent_s, gym_deploy_seed, device=device, detailed_log=True
            )
            seed_chunk_blotters[seed].append(eval_seed_res["blotter"])

        # Assemble Committee Agent for this Deployment Window
        committee_agent = CommitteeAgent(list(chunk_seed_agents.values()))
        # Combine scalers via arithmetic average of moving statistics
        avg_scaler = ObservationScaler(shape=(obs_dim,))
        avg_scaler.mean = np.mean(
            [s["mean"] for s in prior_scalers.values() if s is not None], axis=0
        )
        avg_scaler.var = np.mean(
            [s["var"] for s in prior_scalers.values() if s is not None], axis=0
        )
        avg_scaler.count = float(
            np.mean([s["count"] for s in prior_scalers.values() if s is not None])
        )

        gym_deploy_comm = make_eval_env(
            feature_cube=feature_cube,
            simple_ret_matrix=simple_ret_matrix,
            calendar=cal_deploy,
            macro_df=macro_df,
            config=config,
            scaler_state=avg_scaler,
        )
        eval_comm_res = AgentEvaluator.evaluate(
            committee_agent, gym_deploy_comm, device=device, detailed_log=True
        )
        committee_chunk_blotters.append(eval_comm_res["blotter"])

        print(
            f"   ✨ [Chunk {cid} OOS Results] Committee Sharpe: {eval_comm_res['sharpe_ratio']:.3f} | "
            f"Excess: {eval_comm_res['excess_return']*100:+.2f}% | IR: {eval_comm_res['information_ratio']:.3f}"
        )

    # 3. Continuous Blotter Geometric Stitching
    print("\n" + "=" * 75)
    print("🧵 STITCHING CONTINUOUS BLOTTERS (2022 - 2026 HORIZON)")
    print("=" * 75)

    df_comm_continuous = stitch_continuous_blotters(committee_chunk_blotters)
    comm_metrics = compute_institutional_metrics(df_comm_continuous)

    # Stitch Individual Seeds for Variance Reduction Gate
    seed_metrics = {}
    for s in seeds:
        df_s = stitch_continuous_blotters(seed_chunk_blotters[s])
        seed_metrics[s] = compute_institutional_metrics(df_s)

    mean_constituent_sharpe = float(
        np.mean([seed_metrics[s]["sharpe_ratio"] for s in seeds])
    )

    # 4. Falsifiable Gates Decision Scorecard
    gate1 = comm_metrics["sharpe_ratio"] >= 0.850
    gate2 = comm_metrics["excess_return"] >= 0.050
    gate3 = comm_metrics["sharpe_ratio"] > mean_constituent_sharpe
    gate4 = comm_metrics["beta"] >= 0.970

    scorecard = {
        "gate_1_sharpe_ge_0_85": {
            "pass": bool(gate1),
            "value": comm_metrics["sharpe_ratio"],
        },
        "gate_2_excess_ge_5pct": {
            "pass": bool(gate2),
            "value": comm_metrics["excess_return"],
        },
        "gate_3_variance_reduction": {
            "pass": bool(gate3),
            "committee_sharpe": comm_metrics["sharpe_ratio"],
            "constituent_avg_sharpe": mean_constituent_sharpe,
        },
        "gate_4_beta_ge_0_97": {"pass": bool(gate4), "value": comm_metrics["beta"]},
    }

    all_passed = all([gate1, gate2, gate3, gate4])
    verdict = (
        "CONFIRMED" if all_passed else ("PARTIAL" if gate1 or gate2 else "REFUTED")
    )

    print("\n" + "#" * 75)
    print(f"📊 GENERATION 17 INSTITUTIONAL VERDICT: {verdict}")
    print("#" * 75)
    print(
        f"  • Continuous Sharpe  : {comm_metrics['sharpe_ratio']:.3f}  [Target: >= 0.850] -> {'✅ PASS' if gate1 else '❌ FAIL'}"
    )
    print(
        f"  • Excess Return over SPY: {comm_metrics['excess_return']*100:+.2f}% [Target: >= +5.00%] -> {'✅ PASS' if gate2 else '❌ FAIL'}"
    )
    print(
        f"  • Committee vs Constituent: {comm_metrics['sharpe_ratio']:.3f} vs {mean_constituent_sharpe:.3f} -> {'✅ PASS' if gate3 else '❌ FAIL'}"
    )
    print(
        f"  • Beta Preservation  : {comm_metrics['beta']:.3f}  [Target: >= 0.970] -> {'✅ PASS' if gate4 else '❌ FAIL'}"
    )
    print(f"  • Total OOS Sessions : {comm_metrics['sessions']} days")
    print("#" * 75)

    # 5. Persist Continuous Blotter & Catalog Metrics
    out_blotter_path = OUTPUT_DIR / "blotter_continuous_committee_gen17.parquet"
    df_comm_continuous.to_parquet(out_blotter_path, index=False)
    print(f"💾 Saved authoritative continuous blotter: {out_blotter_path}")

    return {
        "verdict": verdict,
        "scorecard": scorecard,
        "committee_metrics": comm_metrics,
        "seed_metrics": seed_metrics,
        "blotter_path": str(out_blotter_path),
    }


if __name__ == "__main__":
    start_t = time.time()
    res = run_walk_forward_pipeline()
    print(
        f"\n⏱️ Completed in {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_t))}"
    )
