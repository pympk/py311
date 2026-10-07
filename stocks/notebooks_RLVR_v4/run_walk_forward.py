"""
Walk-Forward Multi-Seed Ex-Post Blend Engine Orchestrator (Chunks 0..4).
Executes expanding walk-forward fine-tuning, native constituent seed training,
and mathematically rigorous ex-post capital ensembling.
Strict Invariant: Zero inter-generation model migration. Each generation trains native weights.
"""

import argparse
import copy
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
from typing import Any, Dict, Optional, Tuple, List

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
from rl_discovery.agent import AbsoluteZeroAgent
from rl_discovery.trainer import PPOTrainer, RolloutBuffer
from rl_discovery.validator import AgentEvaluator

# =============================================================================
# WALK-FORWARD HORIZON DEFINITION CONTRACT (CHUNKS 0..4)
# =============================================================================
# Invariant: Fixed 1-year annual deployment strides for Chunks 0..3 (~250-252d).
# Terminal fold (Chunk 4) dynamically anchors deploy_end to trading_calendar.max().
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
        "deploy_end": "2026-03-24",
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
        "chunk_id": 4,
        "name": "Chunk_4_FineTune_Y4",
        "train_start": "2016-01-04",
        "train_end": "2026-03-24",
        "hist_cutoff_date": "2025-03-24",
        "deploy_start": "2026-03-25",
        "deploy_end": None,  # Dynamically binds to trading_calendar.max()
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
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


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
    generation: int = 21,
    force_retrain: bool = False,
) -> Tuple[Path, Dict[str, Any]]:
    chunk_id = chunk_spec["chunk_id"]
    best_checkpoint_path = checkpoint_dir / f"model_chunk{chunk_id}_s{seed}_champion.pt"

    # =========================================================================
    # RESUME CACHE (STRICTLY ISOLATED TO CURRENT GENERATION CHECKPOINTS)
    # =========================================================================
    if not force_retrain and best_checkpoint_path.exists():
        print(
            f"      ⏩ [CACHE HIT] Checkpoint exists: {best_checkpoint_path.name}. Skipping training."
        )
        payload = torch.load(
            best_checkpoint_path, map_location=device, weights_only=False
        )
        return best_checkpoint_path, payload.get("scaler_state", {})

    set_seed(seed)
    obs_dim = 3 * feature_cube.shape[1] + len(macro_df.columns)
    action_dim = feature_cube.shape[1] + 4

    agent = AbsoluteZeroAgent(obs_dim=obs_dim, action_dim=action_dim).to(device)
    anchor_agent: Optional[AbsoluteZeroAgent] = None
    initial_scaler_state: Optional[Dict[str, Any]] = None

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
        seed=seed,
    )

    gym_val = make_eval_env(
        feature_cube=feature_cube,
        simple_ret_matrix=simple_ret_matrix,
        calendar=cal_val,
        macro_df=macro_df,
        config=config,
        scaler_state=initial_scaler_state,
        seed=seed,
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
        seed=seed,
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
    best_scaler_state: Dict[str, Any] = initial_scaler_state or {}
    patience_counter = 0

    total_epochs = chunk_spec["epochs"]
    warmup_epochs = chunk_spec["warmup_epochs"]
    min_epochs = chunk_spec["min_epochs"]
    patience = chunk_spec["patience"]

    runtime_grid_params = {
        **config.to_dict(),
        **chunk_spec,
        "holding_period": config.holding_period,
        "benchmark": config.benchmark,
        "gamma": config.gamma,
    }

    # Embedded Convergence Telemetry Recorder
    history: Dict[str, List[Any]] = {
        "epoch": [],
        "total_loss": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "approx_kl": [],
        "clip_fraction": [],
        "explained_variance": [],
        "avg_reward": [],
        "step_reward": [],
        "val_fitness": [],
        "val_sharpe": [],
        "val_excess": [],
        "val_ir": [],
    }

    for epoch in range(total_epochs):
        t_epoch_start = time.time()
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

        train_scaler = envs.get_attr("scaler")[0]
        gym_val.scaler.load_state(train_scaler)
        val_res = AgentEvaluator.evaluate(agent, gym_val, device=device)

        raw_fitness = QuantUtils.compute_composite_fitness(
            excess_return=val_res["excess_return"],
            information_ratio=val_res["information_ratio"],
        )
        fitness = float(raw_fitness) if np.isfinite(raw_fitness) else -999.0

        ep_idx = epoch + 1
        mean_rew = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
        tot_l = float(diag.get("total_loss", diag.get("loss", 0.0)))
        pol_l = float(diag.get("policy_loss", diag.get("pg_loss", 0.0)))
        val_l = float(diag.get("value_loss", diag.get("v_loss", 0.0)))
        ent_v = float(diag.get("entropy", diag.get("ent_loss", 0.0)))
        kl_v = float(diag.get("approx_kl", diag.get("kl", 0.0)))
        clip_v = float(
            diag.get("clip_fraction", diag.get("clipfrac", diag.get("clip_frac", 0.0)))
        )
        ev_v = float(diag.get("explained_variance", 0.0))

        history["epoch"].append(ep_idx)
        history["total_loss"].append(tot_l)
        history["policy_loss"].append(pol_l)
        history["value_loss"].append(val_l)
        history["entropy"].append(ent_v)
        history["approx_kl"].append(kl_v)
        history["clip_fraction"].append(clip_v)
        history["explained_variance"].append(ev_v)
        history["avg_reward"].append(mean_rew)
        history["step_reward"].append(mean_rew)
        history["val_fitness"].append(fitness)
        history["val_sharpe"].append(float(val_res["sharpe_ratio"]))
        history["val_excess"].append(float(val_res["excess_return"]))
        history["val_ir"].append(float(val_res["information_ratio"]))

        is_warmup = (epoch + 1) <= warmup_epochs
        t_epoch_sec = time.time() - t_epoch_start
        new_best_flag = ""

        if not is_warmup and fitness > best_fitness and fitness > -900.0:
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
                    "generation": generation,
                    "grid_params": runtime_grid_params,
                    "holding_period": config.holding_period,
                    "benchmark": config.benchmark,
                    "history": {k: list(v) for k, v in history.items()},
                },
                best_checkpoint_path,
            )
            new_best_flag = f" ⭐ [NEW BEST! Fit: {fitness:.4f}]"
        else:
            if not is_warmup:
                patience_counter += 1

        warmup_tag = "🔥[WARMUP]" if is_warmup else f"P:{patience_counter}/{patience}"
        print(
            f"      [Chunk {chunk_id}|s{seed}] Ep {epoch+1:02d}/{total_epochs:02d} | "
            f"Rew: {mean_rew:+.4f} | "
            f"Loss: {tot_l:.4f} | "
            f"EV: {ev_v:.3f} | "
            f"Val Sh: {val_res['sharpe_ratio']:.3f} | "
            f"Excess: {val_res['excess_return']*100:+.2f}% | "
            f"{t_epoch_sec:.1f}s | {warmup_tag}{new_best_flag}"
        )

        if (epoch + 1) >= min_epochs and patience_counter >= patience:
            print(
                f"      🛑 [Early Stop] Seed {seed} stopped at Epoch {epoch+1} (Best Fitness: {best_fitness:.4f})"
            )
            break

    if best_checkpoint_path.exists():
        champion_payload = torch.load(
            best_checkpoint_path, map_location=device, weights_only=False
        )
        champion_payload["history"] = {k: list(v) for k, v in history.items()}
        champion_payload["grid_params"] = runtime_grid_params
        champion_payload["generation"] = generation
        torch.save(champion_payload, best_checkpoint_path)
    else:
        if prior_checkpoint_path is not None and prior_checkpoint_path.exists():
            print(
                f"      ⚠️ [Fallback] Retaining prior weights: {prior_checkpoint_path.name}"
            )
            shutil.copyfile(prior_checkpoint_path, best_checkpoint_path)
            prior_payload = torch.load(
                prior_checkpoint_path, map_location=device, weights_only=False
            )
            prior_payload["history"] = {k: list(v) for k, v in history.items()}
            prior_payload["grid_params"] = runtime_grid_params
            prior_payload["generation"] = generation
            torch.save(prior_payload, best_checkpoint_path)
            best_scaler_state = prior_payload.get("scaler_state", best_scaler_state)
        else:
            torch.save(
                {
                    "model_state_dict": agent.state_dict(),
                    "scaler_state": best_scaler_state,
                    "epoch": total_epochs,
                    "seed": seed,
                    "chunk_id": chunk_id,
                    "generation": generation,
                    "grid_params": runtime_grid_params,
                    "holding_period": config.holding_period,
                    "benchmark": config.benchmark,
                    "history": {k: list(v) for k, v in history.items()},
                },
                best_checkpoint_path,
            )

    envs.close()
    del envs, gym_val, buffer, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_checkpoint_path, best_scaler_state


def stitch_continuous_blotters(
    chunk_blotters: List[List[Dict[str, Any]]],
) -> pd.DataFrame:
    """Stitches sequential deployment blotters using strict geometric compounding."""
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


def synthesize_ex_post_blend(
    seed_blotters: Dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Synthesizes the ex-post blended portfolio from constituent seed blotters."""
    seeds = sorted(list(seed_blotters.keys()))
    if not seeds:
        raise ValueError("seed_blotters dictionary is empty")

    base_df = seed_blotters[seeds[0]].copy()
    num_sessions = len(base_df)

    for s in seeds[1:]:
        s_df = seed_blotters[s]
        if len(s_df) != num_sessions:
            raise ValueError(
                f"Seed {s} session count ({len(s_df)}) diverges from seed {seeds[0]} ({num_sessions})"
            )
        if not s_df["date"].equals(base_df["date"]):
            raise ValueError(
                f"Seed {s} date index diverges from seed {seeds[0]} reference dates"
            )

    rets_stack = np.stack(
        [seed_blotters[s]["net_daily_simple_ret"].to_numpy(dtype=float) for s in seeds],
        axis=0,
    )
    blend_net_rets = np.mean(rets_stack, axis=0)
    bm_rets = base_df["bm_daily_simple_ret"].to_numpy(dtype=float)

    p_curve = np.cumprod(1.0 + blend_net_rets)
    bm_curve = np.cumprod(1.0 + bm_rets)
    alpha_curve = p_curve / np.maximum(bm_curve, 1e-8)

    blend_df = base_df.copy()
    blend_df["net_daily_simple_ret"] = blend_net_rets
    blend_df["bm_daily_simple_ret"] = bm_rets
    blend_df["alpha_daily_simple_ret"] = blend_net_rets - bm_rets
    blend_df["agent_equity"] = p_curve
    blend_df["bm_equity"] = bm_curve
    blend_df["alpha_equity"] = alpha_curve

    numeric_cols = [
        "weight_active",
        "weight_benchmark",
        "weight_cash",
        "equity_exposure",
        "active_tilt",
        "gross_stock_daily_simple_ret",
        "gross_daily_simple_ret",
        "slippage_daily_simple_loss",
    ]
    for col in numeric_cols:
        if all(col in seed_blotters[s].columns for s in seeds):
            col_stack = np.stack(
                [seed_blotters[s][col].to_numpy(dtype=float) for s in seeds],
                axis=0,
            )
            blend_df[col] = np.mean(col_stack, axis=0)

    return blend_df


def compute_institutional_metrics(df_blotter: pd.DataFrame) -> Dict[str, float]:
    """Computes authoritative institutional risk/return metrics from continuous blotter."""
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


def save_run_manifest(
    generation: int,
    seeds: List[int],
    config: TradingConfig,
    chunk_specs: List[Dict[str, Any]],
    scorecard: Dict[str, Any],
    blend_metrics: Dict[str, Any],
    seed_metrics: Dict[int, Dict[str, float]],
    div_ratio: float,
    verdict: str,
    feature_cube: pd.DataFrame,
    macro_df: pd.DataFrame,
    trading_calendar: pd.DatetimeIndex,
    cache_file: Path,
    out_blend_blotter: Path,
    canonical_blend_blotter: Path,
    seed_blotter_paths: Dict[int, Path],
    checkpoint_paths: Dict[str, Path],
    checkpoint_dir: Path,
    canonical_dir: Path,
) -> Path:
    """
    Serializes a comprehensive generation metadata manifest (run_metadata.json)
    capturing environment determinism, data lineage, exact runtime config,
    walk-forward chunk schedules, and institutional scorecard.
    """
    manifest_data = {
        "generation": generation,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "seeds": seeds,
        "environment": {
            "python_version": sys.version,
            "platform": sys.platform,
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": "cuda:0" if torch.cuda.is_available() else "cpu",
            "device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            ),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "torch_num_threads": torch.get_num_threads(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "N/A"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", "N/A"),
        },
        "data_lineage": {
            "cache_file": cache_file.name,
            "cache_file_size_bytes": (
                cache_file.stat().st_size if cache_file.exists() else None
            ),
            "cache_file_mtime_utc": (
                datetime.fromtimestamp(
                    cache_file.stat().st_mtime, tz=timezone.utc
                ).isoformat()
                if cache_file.exists()
                else None
            ),
            "cache_config": {
                "lookback": CacheConfig.LOOKBACK,
                "start_date": CacheConfig.START_DATE,
                "end_date": CacheConfig.END_DATE,
            },
            "obs_dim": int(3 * feature_cube.shape[1] + len(macro_df.columns)),
            "action_dim": int(feature_cube.shape[1] + 4),
            "universe_size": int(feature_cube.shape[1]),
            "calendar_start": (
                str(trading_calendar.min().date())
                if len(trading_calendar) > 0
                else None
            ),
            "calendar_end": (
                str(trading_calendar.max().date())
                if len(trading_calendar) > 0
                else None
            ),
            "total_trading_sessions": int(len(trading_calendar)),
        },
        "chunk_specs": chunk_specs,
        "trading_config": config.to_dict(),
        "institutional_scorecard": scorecard,
        "blend_metrics": blend_metrics,
        "constituent_metrics": {str(k): v for k, v in seed_metrics.items()},
        "diversification_ratio": div_ratio,
        "artifacts": {
            "blend_blotter": str(out_blend_blotter),
            "canonical_blend_blotter": str(canonical_blend_blotter),
            "constituent_blotters": {
                str(k): str(v) for k, v in seed_blotter_paths.items()
            },
            "checkpoints": {k: str(v) for k, v in checkpoint_paths.items()},
        },
    }

    manifest_path = checkpoint_dir / "run_metadata.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, default=str)

    if canonical_dir.exists():
        canonical_manifest = canonical_dir / f"run_metadata_gen{generation}.json"
        shutil.copyfile(manifest_path, canonical_manifest)

    return manifest_path


def run_walk_forward_pipeline(
    generation: int = 21,
    seeds: Optional[List[int]] = None,
    force_retrain: bool = False,
    config: Optional[TradingConfig] = None,
    chunk_specs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Main Entrypoint: Runs expanding walk-forward engine with constituent independence,
    strictly isolated per-generation checkpointing, dynamic frontier resolution,
    ex-post capital ensembling, and deterministic metadata manifest persistence.
    """
    seeds = seeds or [42, 101, 777]
    config = config or TradingConfig()
    chunk_specs = chunk_specs or copy.deepcopy(CHUNK_SPECS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"🚀 Initializing Gen {generation} Walk-Forward Ex-Post Ensemble Engine (Seeds: {seeds}) on {device}"
    )

    data = load_processed_data()
    df_ohlcv = data.df_ohlcv
    macro_df = data.macro_df
    df_close = df_ohlcv["Adj Close"].unstack(level=0).sort_index()
    master_cal = get_master_trading_calendar(df_ohlcv, config.calendar_ticker)

    cache_file = LOCAL_DATA_DIR / CacheConfig.get_filename()
    feature_cube = pd.read_parquet(cache_file)
    valid_dates = set(feature_cube.index.get_level_values("Date").unique())
    trading_calendar = master_cal[master_cal.isin(valid_dates)]

    simple_ret_matrix = df_close.pct_change(1, fill_method=None).shift(-1)
    simple_ret_matrix["CASH"] = 0.0

    # Dedicated generation checkpoint directory
    checkpoint_dir = OUTPUT_DIR / "model_checkpoints" / f"walk_forward_gen{generation}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    prior_checkpoints: Dict[int, Optional[Path]] = {s: None for s in seeds}
    prior_scalers: Dict[int, Optional[Dict[str, Any]]] = {s: None for s in seeds}
    seed_chunk_blotters: Dict[int, List[List[Dict[str, Any]]]] = {s: [] for s in seeds}
    saved_checkpoints: Dict[str, Path] = {}

    obs_dim = 3 * feature_cube.shape[1] + len(macro_df.columns)
    action_dim = feature_cube.shape[1] + 4

    for chunk in chunk_specs:
        cid = chunk["chunk_id"]
        cname = chunk["name"]
        t_start = pd.Timestamp(chunk["train_start"])
        t_end = pd.Timestamp(chunk["train_end"])
        d_start = pd.Timestamp(chunk["deploy_start"])

        # Dynamic Frontier Resolution: None resolves to latest available session
        if chunk["deploy_end"] is not None:
            d_end = pd.Timestamp(chunk["deploy_end"])
            deploy_tag = d_end.strftime("%Y-%m-%d")
        else:
            d_end = trading_calendar.max()
            deploy_tag = f"{d_end.strftime('%Y-%m-%d')} (DYNAMIC LATEST)"

        cal_train_full = trading_calendar[
            (trading_calendar >= t_start) & (trading_calendar <= t_end)
        ]
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

        hist_cutoff_idx: Optional[int] = None
        if chunk["hist_cutoff_date"] is not None:
            cutoff_ts = pd.Timestamp(chunk["hist_cutoff_date"])
            hist_matches = np.where(cal_train <= cutoff_ts)[0]
            if len(hist_matches) > 0:
                hist_cutoff_idx = int(hist_matches[-1])

        val_start_idx = max(0, len(cal_train) - 252)
        cal_val = cal_train[val_start_idx:]

        print(f"\n{'='*75}")
        print(
            f"🔄 EXECUTING CHUNK {cid}: {cname} | Train: {len(cal_train)}d | "
            f"Deploy: {chunk['deploy_start']} -> {deploy_tag} ({len(cal_deploy)}d)"
        )
        print(f"{'='*75}")

        for seed in seeds:
            print(f"   ⚙️ Seed {seed}: Processing Chunk {cid}...")
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
                generation=generation,
                force_retrain=force_retrain,
            )
            prior_checkpoints[seed] = ckpt_path
            prior_scalers[seed] = scaler_state
            saved_checkpoints[f"chunk{cid}_s{seed}"] = ckpt_path

            agent_s = AbsoluteZeroAgent(obs_dim=obs_dim, action_dim=action_dim).to(
                device
            )
            payload = torch.load(ckpt_path, map_location=device, weights_only=False)
            agent_s.load_state_dict(payload["model_state_dict"])
            agent_s.eval()

            gym_deploy_seed = make_eval_env(
                feature_cube=feature_cube,
                simple_ret_matrix=simple_ret_matrix,
                calendar=cal_deploy,
                macro_df=macro_df,
                config=config,
                scaler_state=scaler_state,
                seed=seed,
            )
            eval_seed_res = AgentEvaluator.evaluate(
                agent_s, gym_deploy_seed, device=device, detailed_log=True
            )
            seed_chunk_blotters[seed].append(eval_seed_res["blotter"])

            print(
                f"      [Chunk {cid}|s{seed} Deploy] Sharpe: {eval_seed_res['sharpe_ratio']:.3f} | "
                f"Excess: {eval_seed_res['excess_return']*100:+.2f}% | IR: {eval_seed_res['information_ratio']:.3f}"
            )

    # =========================================================================
    # CONTINUOUS STITCHING & EX-POST CAPITAL ENSEMBLING
    # =========================================================================
    print("\n" + "=" * 75)
    print(f"🧵 STITCHING CONTINUOUS BLOTTERS & EX-POST ENSEMBLE (GEN {generation})")
    print("=" * 75)

    seed_continuous_blotters: Dict[int, pd.DataFrame] = {}
    seed_metrics: Dict[int, Dict[str, float]] = {}
    seed_blotter_paths: Dict[int, Path] = {}

    for s in seeds:
        df_s = stitch_continuous_blotters(seed_chunk_blotters[s])
        seed_continuous_blotters[s] = df_s
        seed_metrics[s] = compute_institutional_metrics(df_s)

        out_seed_blotter = (
            OUTPUT_DIR / f"blotter_continuous_s{s}_gen{generation}.parquet"
        )
        df_s.to_parquet(out_seed_blotter, index=False)
        seed_blotter_paths[s] = out_seed_blotter
        print(f"💾 Saved constituent continuous blotter: {out_seed_blotter.name}")

    df_blend_continuous = synthesize_ex_post_blend(seed_continuous_blotters)
    blend_metrics = compute_institutional_metrics(df_blend_continuous)

    # Dual-path persistence (Primary output root + canonical anchors directory)
    out_blend_blotter = (
        OUTPUT_DIR / f"blotter_continuous_ex_post_blend_gen{generation}.parquet"
    )
    df_blend_continuous.to_parquet(out_blend_blotter, index=False)
    print(f"💾 Saved ex-post blended continuous blotter: {out_blend_blotter.name}")

    canonical_dir = OUTPUT_DIR / "canonical_anchors"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    canonical_blend_blotter = (
        canonical_dir / f"blotter_continuous_ex_post_blend_gen{generation}.parquet"
    )
    df_blend_continuous.to_parquet(canonical_blend_blotter, index=False)
    print(f"💾 Synchronized canonical anchor: {canonical_blend_blotter.name}")

    # =========================================================================
    # DIVERSIFICATION RATIO & DECISION GATES
    # =========================================================================
    constituent_vols = [
        float(
            np.std(
                seed_continuous_blotters[s]["net_daily_simple_ret"].to_numpy(
                    dtype=float
                ),
                ddof=1,
            )
        )
        for s in seeds
    ]
    mean_constituent_vol = float(np.mean(constituent_vols))
    blend_vol = float(
        np.std(
            df_blend_continuous["net_daily_simple_ret"].to_numpy(dtype=float), ddof=1
        )
    )
    div_ratio = float(mean_constituent_vol / blend_vol) if blend_vol > 1e-8 else 1.0

    # =========================================================================
    # GEN 25 AUTHORITATIVE PRODUCTION BASELINE CONTROL (2015 CACHE, 1120 SESSIONS)
    # =========================================================================
    GEN25_BASELINE = {
        "sharpe_ratio": 0.8301,
        "excess_return": 0.4934,
        "max_drawdown": -0.2600,
        "beta": 1.3040,
        "alpha_multiplier": 1.2800,
    }

    mean_constituent_sharpe = float(
        np.mean([seed_metrics[s]["sharpe_ratio"] for s in seeds])
    )
    delta_sharpe_constituents = blend_metrics["sharpe_ratio"] - mean_constituent_sharpe
    delta_sharpe_baseline = (
        blend_metrics["sharpe_ratio"] - GEN25_BASELINE["sharpe_ratio"]
    )
    delta_excess_baseline = (
        blend_metrics["excess_return"] - GEN25_BASELINE["excess_return"]
    )

    # 1. Alpha Expansion: Outperform Gen 25 Sharpe (+0.03) or Excess Return
    gate_sharpe = delta_sharpe_baseline >= 0.030
    gate_excess = blend_metrics["excess_return"] >= GEN25_BASELINE["excess_return"]

    # 2. Risk Defense: Max DD must not worsen past -27.5% (-26.00% - 1.5%)
    gate_mdd = blend_metrics["max_drawdown"] >= (GEN25_BASELINE["max_drawdown"] - 0.015)

    # 3. Asymmetric Ratchet Beta Gate:
    # Passes if within the ideal corridor [0.90, 1.25], OR if beta is lower than
    # baseline while generating significant alpha expansion (delta_sharpe >= +0.03).
    cand_beta = blend_metrics["beta"]
    base_beta = GEN25_BASELINE["beta"]
    in_ideal_corridor = 0.900 <= cand_beta <= 1.250
    is_ratchet_improvement = (cand_beta <= base_beta) and (
        delta_sharpe_baseline >= 0.030
    )
    gate_beta = in_ideal_corridor or is_ratchet_improvement

    # 4. Ensemble Synergy: Blended Sharpe exceeds constituent seed average
    gate_div = div_ratio >= 1.050 and delta_sharpe_constituents > 0.0

    scorecard = {
        "gate_sharpe_expansion": {
            "pass": bool(gate_sharpe),
            "cand_sharpe": blend_metrics["sharpe_ratio"],
            "base_sharpe": GEN25_BASELINE["sharpe_ratio"],
            "delta_sharpe": delta_sharpe_baseline,
        },
        "gate_excess_spread": {
            "pass": bool(gate_excess),
            "cand_excess": blend_metrics["excess_return"],
            "base_excess": GEN25_BASELINE["excess_return"],
            "delta_excess": delta_excess_baseline,
        },
        "gate_drawdown_control": {
            "pass": bool(gate_mdd),
            "cand_mdd": blend_metrics["max_drawdown"],
            "floor_mdd": GEN25_BASELINE["max_drawdown"] - 0.015,
        },
        "gate_beta_discipline": {
            "pass": bool(gate_beta),
            "cand_beta": cand_beta,
            "base_beta": base_beta,
            "corridor": [0.900, 1.250],
            "in_corridor": bool(in_ideal_corridor),
            "ratchet_pass": bool(is_ratchet_improvement),
        },
        "gate_ensemble_synergy": {
            "pass": bool(gate_div),
            "div_ratio": div_ratio,
            "delta_sharpe_constituents": delta_sharpe_constituents,
        },
    }

    # Institutional Tri-State Verdict
    if (gate_sharpe or gate_excess) and gate_mdd and gate_beta and gate_div:
        verdict = "CONFIRMED"
    elif abs(delta_sharpe_baseline) < 0.020 and gate_mdd:
        verdict = "INCONCLUSIVE"  # Plateau detected: freeze sweeps and pivot
    else:
        verdict = "REFUTED"

    # Persist the complete run metadata manifest
    manifest_path = save_run_manifest(
        generation=generation,
        seeds=seeds,
        config=config,
        chunk_specs=chunk_specs,
        scorecard=scorecard,
        blend_metrics=blend_metrics,
        seed_metrics=seed_metrics,
        div_ratio=div_ratio,
        verdict=verdict,
        feature_cube=feature_cube,
        macro_df=macro_df,
        trading_calendar=trading_calendar,
        cache_file=cache_file,
        out_blend_blotter=out_blend_blotter,
        canonical_blend_blotter=canonical_blend_blotter,
        seed_blotter_paths=seed_blotter_paths,
        checkpoint_paths=saved_checkpoints,
        checkpoint_dir=checkpoint_dir,
        canonical_dir=canonical_dir,
    )
    print(f"💾 Saved generation manifest: {manifest_path.name}")

    print("\n" + "#" * 75)
    print(f"📊 GENERATION {generation} INSTITUTIONAL VERDICT: {verdict}")
    print("#" * 75)
    print(
        f"  • Blend Sharpe vs Gen 24 Base : {blend_metrics['sharpe_ratio']:.3f} vs {GEN25_BASELINE['sharpe_ratio']:.3f} (Δ: {delta_sharpe_baseline:+.3f}) -> {'✅ PASS' if gate_sharpe else '❌ FAIL'}"
    )
    print(
        f"  • Excess Return vs Gen 24 Base: {blend_metrics['excess_return']*100:+.2f}% vs {GEN25_BASELINE['excess_return']*100:+.2f}% (Δ: {delta_excess_baseline*100:+.2f}%) -> {'✅ PASS' if gate_excess else '❌ FAIL'}"
    )
    print(
        f"  • Max Drawdown Defense        : {blend_metrics['max_drawdown']*100:+.2f}% [Floor: {(GEN25_BASELINE['max_drawdown']-0.015)*100:+.2f}%] -> {'✅ PASS' if gate_mdd else '❌ FAIL'}"
    )
    beta_status = (
        "✅ PASS (In Corridor)"
        if in_ideal_corridor
        else ("✅ PASS (Ratchet Hedge)" if is_ratchet_improvement else "❌ FAIL")
    )
    print(
        f"  • Beta Ratchet / Corridor     : {cand_beta:.3f} [Base: {base_beta:.3f}, Target: 0.90-1.25] -> {beta_status}"
    )
    print(
        f"  • Ensemble Synergy (Div Ratio): {div_ratio:.4f} [Target: >= 1.050] -> {'✅ PASS' if gate_div else '❌ FAIL'}"
    )
    print(f"  • Total OOS Sessions          : {blend_metrics['sessions']} days")
    print("#" * 75)

    return {
        "verdict": verdict,
        "generation": generation,
        "scorecard": scorecard,
        "blend_metrics": blend_metrics,
        "seed_metrics": seed_metrics,
        "diversification_ratio": div_ratio,
        "blend_blotter_path": str(out_blend_blotter),
        "blotter_path": str(out_blend_blotter),
        "manifest_path": str(manifest_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Walk-Forward Multi-Seed Ex-Post Blend Engine Orchestrator"
    )
    parser.add_argument(
        "--generation",
        type=int,
        default=21,
        help="Generation number identifier (default: 21)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 101, 777],
        help="Constituent random seeds (default: 42 101 777)",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retrain models even if cached checkpoints exist",
    )
    parser.add_argument(
        "--min-active-tilt",
        type=float,
        default=None,
        help="Optional override for config.min_active_tilt (e.g. 0.20)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Optional override for config.benchmark (e.g. SPY, QQQ)",
    )
    args = parser.parse_args()

    run_cfg = TradingConfig()
    if args.min_active_tilt is not None:
        run_cfg.min_active_tilt = float(args.min_active_tilt)
    if args.benchmark is not None:
        run_cfg.benchmark = str(args.benchmark)

    start_t = time.time()
    res = run_walk_forward_pipeline(
        generation=args.generation,
        seeds=args.seeds,
        force_retrain=args.force_retrain,
        config=run_cfg,
    )
    print(
        f"\n⏱️ Execution completed in {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_t))}"
    )
