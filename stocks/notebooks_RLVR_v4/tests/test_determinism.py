"""
End-to-End Bit-Level Determinism Regression Tripwire.
Verifies SHA-256 state and IEEE 754 atol = 0.0 identity across independent training runs.
"""

import gc
import hashlib
import os
from pathlib import Path
import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest
import torch

from core.paths import LOCAL_DATA_DIR
from core.settings import CacheConfig, TradingConfig
from data_pipeline.loader import load_processed_data
from data_pipeline.utils import get_master_trading_calendar
from rl_discovery.adapter import ObservationScaler, make_stratified_train_envs
from rl_discovery.agent import AbsoluteZeroAgent
from rl_discovery.trainer import PPOTrainer, RolloutBuffer


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


def compute_agent_sha256(
    agent: AbsoluteZeroAgent, scaler: ObservationScaler
) -> Tuple[str, Dict[str, str]]:
    hasher = hashlib.sha256()
    tensor_hashes = {}
    for name, param in sorted(agent.state_dict().items()):
        arr = param.detach().cpu().numpy()
        h = hashlib.sha256(arr.tobytes()).hexdigest()
        tensor_hashes[name] = h
        hasher.update(name.encode("utf-8"))
        hasher.update(arr.tobytes())

    scaler_mean_h = hashlib.sha256(scaler.mean.tobytes()).hexdigest()
    scaler_var_h = hashlib.sha256(scaler.var.tobytes()).hexdigest()
    tensor_hashes["scaler.mean"] = scaler_mean_h
    tensor_hashes["scaler.var"] = scaler_var_h
    hasher.update(scaler.mean.tobytes())
    hasher.update(scaler.var.tobytes())

    return hasher.hexdigest(), tensor_hashes


def execute_mini_chunk_run(
    seed: int,
    epochs: int = 2,
    num_envs: int = 4,
    num_steps: int = 64,
    mini_batch_size: int = 64,
    device: torch.device = torch.device("cpu"),
) -> Tuple[AbsoluteZeroAgent, ObservationScaler, str, Dict[str, str], float]:
    set_seed(seed)
    config = TradingConfig()

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

    t_start = pd.Timestamp("2016-01-04")
    t_end = pd.Timestamp("2018-01-04")
    cal_train = trading_calendar[
        (trading_calendar >= t_start) & (trading_calendar <= t_end)
    ]
    cal_train = cal_train[: -config.holding_period]

    obs_dim = 3 * feature_cube.shape[1] + len(macro_df.columns)
    action_dim = feature_cube.shape[1] + 4

    agent = AbsoluteZeroAgent(obs_dim=obs_dim, action_dim=action_dim).to(device)

    envs = make_stratified_train_envs(
        feature_cube=feature_cube,
        simple_ret_matrix=simple_ret_matrix,
        calendar=cal_train,
        macro_df=macro_df,
        config=config,
        num_envs=num_envs,
        episode_steps=num_steps,
        hist_cutoff_idx=None,
        initial_scaler_state=None,
        seed=seed,
    )

    trainer = PPOTrainer(
        agent=agent,
        lr=2.0e-4,
        critic_lr=8.0e-4,
        clip_coef=config.clip_coef,
        clip_vloss=True,
        ent_coef=config.entropy_coef_start,
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

    total_observed_reward = 0.0

    for _ in range(epochs):
        obs, _ = envs.reset()
        for _ in range(num_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs_tensor)

            next_obs, rewards, terminations, truncations, _ = envs.step(
                action.cpu().numpy()
            )
            buffer.add(
                obs=obs,
                action=action,
                logprob=logprob,
                reward=rewards * 25.0,
                value=value,
                terminations=terminations,
                truncations=truncations,
            )
            total_observed_reward += float(np.sum(rewards))
            obs = next_obs

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
        next_term_tensor = torch.tensor(
            terminations, dtype=torch.float32, device=device
        )
        with torch.no_grad():
            next_values = agent.get_value(obs_tensor)
        buffer.compute_advantages(next_values, next_termination=next_term_tensor)

        trainer.update(
            buffer=buffer,
            update_epochs=2,
            mini_batch_size=mini_batch_size,
            stratified_sampling=False,
        )
        buffer.step = 0

    active_scaler = envs.get_attr("scaler")[0]
    envs.close()

    overall_hash, tensor_hashes = compute_agent_sha256(agent, active_scaler)
    return agent, active_scaler, overall_hash, tensor_hashes, total_observed_reward


@pytest.mark.integration
def test_chunk0_bit_level_determinism():
    target_seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent_1, scaler_1, hash_1, t_hashes_1, rew_1 = execute_mini_chunk_run(
        seed=target_seed, device=device
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    agent_2, scaler_2, hash_2, t_hashes_2, rew_2 = execute_mini_chunk_run(
        seed=target_seed, device=device
    )

    # 1. State Dict Weight Divergence Check
    for name, param1 in agent_1.state_dict().items():
        param2 = agent_2.state_dict()[name]
        diff = torch.max(torch.abs(param1 - param2)).item()
        assert diff == 0.0, f"Parameter drifted in layer {name}: max diff = {diff}"
        assert (
            t_hashes_1[name] == t_hashes_2[name]
        ), f"Tensor hash mismatch in layer {name}"

    # 2. Observation Scaler Divergence Check
    scaler_mean_diff = float(np.max(np.abs(scaler_1.mean - scaler_2.mean)))
    scaler_var_diff = float(np.max(np.abs(scaler_1.var - scaler_2.var)))
    assert scaler_mean_diff == 0.0, f"Scaler mean drifted: diff = {scaler_mean_diff}"
    assert scaler_var_diff == 0.0, f"Scaler variance drifted: diff = {scaler_var_diff}"

    # 3. Trajectory Reward and Master Checkpoint SHA-256 Identity
    assert (
        abs(rew_1 - rew_2) == 0.0
    ), f"Rollout reward divergence detected: {abs(rew_1 - rew_2)}"
    assert (
        hash_1 == hash_2
    ), f"Critical checkpoint SHA-256 mismatch: {hash_1} != {hash_2}"
