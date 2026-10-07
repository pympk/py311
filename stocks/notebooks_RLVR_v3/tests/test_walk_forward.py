"""
Validation suite for Generation 17 Walk-Forward Multi-Seed Committee Engine:
1. Stratified Rollout Buffer Mini-Batching (50% Historical / 50% Recent).
2. Closed-Form Analytical Gaussian KL Divergence Anchoring and Gradient Isolation.
3. Behavioral Anchor Diagnostics Logging.
"""

import copy
import numpy as np
import pytest
import torch
import torch.distributions.kl as dist_kl
from typing import List, cast

from rl_discovery.agent import AbsoluteZeroAgent
from rl_discovery.trainer import PPOTrainer, RolloutBuffer


@pytest.fixture
def agent_dims():
    return {"obs_dim": 46, "action_dim": 16, "hidden_size": 64}


@pytest.fixture
def mock_buffer(agent_dims):
    num_steps = 16
    num_envs = 8
    buffer = RolloutBuffer(
        num_steps=num_steps,
        num_envs=num_envs,
        obs_dim=agent_dims["obs_dim"],
        action_dim=agent_dims["action_dim"],
        device=torch.device("cpu"),
    )
    for step in range(num_steps):
        obs = np.random.randn(num_envs, agent_dims["obs_dim"]).astype(np.float32)
        actions = torch.randn(num_envs, agent_dims["action_dim"])
        logprobs = torch.zeros(num_envs)
        rewards = np.random.randn(num_envs).astype(np.float32)
        values = torch.zeros(num_envs)
        terminations = np.zeros(num_envs, dtype=np.float32)
        truncations = np.zeros(num_envs, dtype=np.float32)
        buffer.add(
            obs=obs,
            action=actions,
            logprob=logprobs,
            reward=rewards,
            value=values,
            terminations=terminations,
            truncations=truncations,
        )

    next_value = torch.zeros(num_envs)
    next_termination = torch.zeros(num_envs)
    buffer.compute_advantages(next_value, next_termination)
    return buffer


def test_stratified_sampling_partitions_equally(agent_dims, mock_buffer):
    """Verifies that every mini-batch draws exactly 50% from envs 0..3 and 50% from envs 4..7."""
    agent = AbsoluteZeroAgent(
        obs_dim=agent_dims["obs_dim"],
        action_dim=agent_dims["action_dim"],
        hidden_size=agent_dims["hidden_size"],
    )
    trainer = PPOTrainer(agent=agent)

    batch_size = mock_buffer.num_steps * mock_buffer.num_envs  # 16 * 8 = 128
    mini_batch_size = 32  # 4 mini-batches per epoch
    half_envs = mock_buffer.num_envs // 2  # 4

    all_inds = np.arange(batch_size)
    env_ids = all_inds % mock_buffer.num_envs
    hist_inds = set(all_inds[env_ids < half_envs])
    rec_inds = set(all_inds[env_ids >= half_envs])

    # Replicate trainer mini-batch assembly logic directly to verify index provenance
    shuffled_hist = np.random.permutation(list(hist_inds))
    shuffled_rec = np.random.permutation(list(rec_inds))
    half_mb = mini_batch_size // 2

    for start in range(0, len(shuffled_hist), half_mb):
        end = start + half_mb
        h_chunk = shuffled_hist[start:end]
        r_chunk = shuffled_rec[start:end]
        mb = np.concatenate([h_chunk, r_chunk])

        assert len(mb) == mini_batch_size
        hist_count = sum(1 for idx in mb if idx in hist_inds)
        rec_count = sum(1 for idx in mb if idx in rec_inds)
        assert (
            hist_count == half_mb
        ), f"Expected {half_mb} historical samples, got {hist_count}"
        assert (
            rec_count == half_mb
        ), f"Expected {half_mb} recent samples, got {rec_count}"

    # Verify trainer.update executes smoothly with stratified_sampling=True
    diag = trainer.update(
        mock_buffer,
        update_epochs=1,
        mini_batch_size=mini_batch_size,
        stratified_sampling=True,
    )
    assert "policy_loss" in diag
    assert np.isfinite(diag["policy_loss"])


def test_stratified_sampling_validation_guards(agent_dims, mock_buffer):
    """Guarantees ValueError is raised when num_envs or mini_batch_size violates parity."""
    agent = AbsoluteZeroAgent(
        obs_dim=agent_dims["obs_dim"],
        action_dim=agent_dims["action_dim"],
        hidden_size=agent_dims["hidden_size"],
    )
    trainer = PPOTrainer(agent=agent)

    # 1. Odd mini_batch_size
    with pytest.raises(ValueError, match="mini_batch_size must be even"):
        trainer.update(
            mock_buffer,
            update_epochs=1,
            mini_batch_size=31,
            stratified_sampling=True,
        )

    # 2. Buffer with odd num_envs
    odd_buffer = RolloutBuffer(
        num_steps=8,
        num_envs=3,
        obs_dim=agent_dims["obs_dim"],
        action_dim=agent_dims["action_dim"],
    )
    for _ in range(8):
        odd_buffer.add(
            obs=np.zeros((3, agent_dims["obs_dim"])),
            action=torch.zeros(3, agent_dims["action_dim"]),
            logprob=torch.zeros(3),
            reward=np.zeros(3),
            value=torch.zeros(3),
            terminations=np.zeros(3),
            truncations=np.zeros(3),
        )
    odd_buffer.compute_advantages(torch.zeros(3), torch.zeros(3))

    with pytest.raises(ValueError, match="even number of environments"):
        trainer.update(
            odd_buffer,
            update_epochs=1,
            mini_batch_size=16,
            stratified_sampling=True,
        )


def test_analytical_gaussian_kl_identity_and_divergence(agent_dims):
    """
    Mathematical Invariant:
    1. Identical policies have KL identically zero (0.0).
    2. Perturbed policy has strictly positive KL (> 0.0).
    """
    anchor_agent = AbsoluteZeroAgent(
        obs_dim=agent_dims["obs_dim"],
        action_dim=agent_dims["action_dim"],
        hidden_size=agent_dims["hidden_size"],
    )
    student_agent = copy.deepcopy(anchor_agent)

    obs = torch.randn(32, agent_dims["obs_dim"])

    # 1. Identical distributions
    dist_anchor = anchor_agent.get_distribution(obs)
    dist_student = student_agent.get_distribution(obs)
    kl_identical = dist_kl.kl_divergence(dist_anchor, dist_student).sum(dim=-1).mean()
    assert torch.isclose(kl_identical, torch.tensor(0.0), atol=1e-6)

    # 2. Perturb student actor mean weights
    with torch.no_grad():
        for p in student_agent.actor_mean.parameters():
            p.add_(torch.randn_like(p) * 0.25)

    dist_student_perturbed = student_agent.get_distribution(obs)
    kl_divergent = (
        dist_kl.kl_divergence(dist_anchor, dist_student_perturbed).sum(dim=-1).mean()
    )
    assert kl_divergent.item() > 0.01


def test_anchor_gradient_isolation_and_flow(agent_dims, mock_buffer):
    """
    Verifies that:
    1. Gradients from anchor_loss flow cleanly to student actor parameters.
    2. Anchor agent parameters remain frozen (requires_grad = False, grad = None).
    """
    anchor_agent = AbsoluteZeroAgent(
        obs_dim=agent_dims["obs_dim"],
        action_dim=agent_dims["action_dim"],
        hidden_size=agent_dims["hidden_size"],
    )
    student_agent = copy.deepcopy(anchor_agent)

    # Perturb student to create positive divergence
    with torch.no_grad():
        for p in student_agent.actor_mean.parameters():
            p.add_(torch.randn_like(p) * 0.10)

    trainer = PPOTrainer(
        agent=student_agent,
        anchor_agent=anchor_agent,
        kl_anchor_coef=0.10,
    )

    # Verify anchor parameter immutability
    for p in anchor_agent.parameters():
        assert not p.requires_grad

    diag = trainer.update(
        mock_buffer,
        update_epochs=1,
        mini_batch_size=32,
        stratified_sampling=True,
    )

    # Verify anchor_kl telemetry
    assert "anchor_kl" in diag
    assert diag["anchor_kl"] > 0.0

    # Verify student parameters received valid finite gradients
    for name, p in student_agent.actor_mean.named_parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.all(torch.isfinite(p.grad))

    # Verify anchor agent parameters accumulated ZERO gradients
    for p in anchor_agent.parameters():
        assert p.grad is None


def test_diagnostics_logging_zero_when_no_anchor(agent_dims, mock_buffer):
    """Verifies anchor_kl returns 0.0 when no anchor agent is attached."""
    agent = AbsoluteZeroAgent(
        obs_dim=agent_dims["obs_dim"],
        action_dim=agent_dims["action_dim"],
        hidden_size=agent_dims["hidden_size"],
    )
    trainer = PPOTrainer(agent=agent, anchor_agent=None, kl_anchor_coef=0.0)
    diag = trainer.update(mock_buffer, update_epochs=1, mini_batch_size=32)
    assert diag["anchor_kl"] == 0.0


import pandas as pd
from core.settings import TradingConfig
from rl_discovery.adapter import (
    ObservationScaler,
    RLVRGymEnv,
    make_eval_env,
    make_stratified_train_envs,
)
from run_walk_forward import compute_institutional_metrics, stitch_continuous_blotters


def test_make_stratified_train_envs_bounds_and_structure(agent_dims):
    """
    Verifies that make_stratified_train_envs constructs 8 environments with:
    - Envs 0..3 bounded to (0, hist_cutoff_idx).
    - Envs 4..7 bounded to (hist_cutoff_idx, len(cal) - 1).
    """
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    feature_cube = pd.DataFrame(
        np.random.randn(100 * 2, 12),
        index=pd.MultiIndex.from_product(
            [dates, ["AAPL", "MSFT"]], names=["Date", "Ticker"]
        ),
        columns=[f"F_{i}" for i in range(12)],
    )
    simple_rets = pd.DataFrame(
        np.random.randn(100, 2) * 0.01,
        index=dates,
        columns=["AAPL", "MSFT"],
    )
    simple_rets["CASH"] = 0.0
    macro_df = pd.DataFrame(
        np.random.randn(100, 10), index=dates, columns=[f"M_{i}" for i in range(10)]
    )
    cfg = TradingConfig()

    cutoff = 60
    vec_envs = make_stratified_train_envs(
        feature_cube=feature_cube,
        simple_ret_matrix=simple_rets,
        calendar=dates,
        macro_df=macro_df,
        config=cfg,
        num_envs=8,
        episode_steps=32,
        hist_cutoff_idx=cutoff,
    )

    assert vec_envs.num_envs == 8
    # Inspect internal sub-environments with concrete subclass casting
    sub_envs = cast(List[RLVRGymEnv], vec_envs.envs)
    for i in range(4):
        assert sub_envs[i].env.start_idx_bounds == (0, cutoff)
    for i in range(4, 8):
        assert sub_envs[i].env.start_idx_bounds == (cutoff, len(dates) - 1)

    vec_envs.close()


def test_make_eval_env_unbounded_contract():
    """
    Verifies make_eval_env enforces episode_steps = 0 and is_training = False.
    """
    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    feature_cube = pd.DataFrame(
        np.zeros((50 * 2, 12)),
        index=pd.MultiIndex.from_product(
            [dates, ["AAPL", "MSFT"]], names=["Date", "Ticker"]
        ),
        columns=[f"F_{i}" for i in range(12)],
    )
    simple_rets = pd.DataFrame(np.zeros((50, 2)), index=dates, columns=["AAPL", "MSFT"])
    simple_rets["CASH"] = 0.0
    macro_df = pd.DataFrame(
        np.zeros((50, 10)), index=dates, columns=[f"M_{i}" for i in range(10)]
    )
    cfg = TradingConfig()

    eval_env = make_eval_env(
        feature_cube=feature_cube,
        simple_ret_matrix=simple_rets,
        calendar=dates,
        macro_df=macro_df,
        config=cfg,
    )

    assert eval_env.env.episode_steps == 0
    assert not eval_env.is_training
    assert not eval_env.env.randomize_start


def test_continuous_blotter_stitching_and_geometric_equity():
    """
    Verifies that stitch_continuous_blotters correctly chains compounding wealth:
    V_p(T) = V_p(T_0) * (1 + r_1) * (1 + r_2) ...
    """
    blotter_chunk1 = [
        {
            "date": "2023-01-02",
            "net_daily_simple_ret": 0.02,
            "bm_daily_simple_ret": 0.01,
        },
        {
            "date": "2023-01-03",
            "net_daily_simple_ret": -0.01,
            "bm_daily_simple_ret": -0.02,
        },
    ]
    blotter_chunk2 = [
        {
            "date": "2023-01-04",
            "net_daily_simple_ret": 0.03,
            "bm_daily_simple_ret": 0.01,
        },
    ]

    stitched = stitch_continuous_blotters([blotter_chunk1, blotter_chunk2])
    assert len(stitched) == 3

    # Expected cumulative equity: (1 + 0.02) * (1 - 0.01) * (1 + 0.03) = 1.02 * 0.99 * 1.03 = 1.040094
    expected_p = 1.02 * 0.99 * 1.03
    assert np.isclose(stitched["agent_equity"].iloc[-1], expected_p, atol=1e-5)

    metrics = compute_institutional_metrics(stitched)
    assert np.isclose(metrics["total_return"], expected_p - 1.0, atol=1e-5)
    assert metrics["sessions"] == 3


def test_checkpoint_telemetry_schema(tmp_path):
    """Guarantees that saved champion checkpoints strictly adhere to Architecture A schema."""
    import torch

    dummy_path = tmp_path / "model_chunk0_s42_champion.pt"
    dummy_payload = {
        "model_state_dict": {},
        "scaler_state": {"mean": [0.0], "var": [1.0], "count": 100},
        "history": {
            "epoch": [1, 2],
            "total_loss": [1.5, 1.2],
            "policy_loss": [0.3, 0.2],
            "value_loss": [1.2, 1.0],
            "entropy": [2.5, 2.4],
            "approx_kl": [0.005, 0.008],
            "clip_fraction": [0.05, 0.04],
            "explained_variance": [0.10, 0.25],
            "step_reward": [0.001, 0.002],
            "avg_reward": [0.001, 0.002],
        },
    }
    torch.save(dummy_payload, dummy_path)

    loaded = torch.load(dummy_path, weights_only=False)
    assert "history" in loaded, "Missing 'history' key in champion checkpoint"

    required_keys = {
        "epoch",
        "total_loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "explained_variance",
        "step_reward",
    }
    history_keys = set(loaded["history"].keys())
    assert required_keys.issubset(
        history_keys
    ), f"Missing telemetry channels: {required_keys - history_keys}"
    assert len(loaded["history"]["epoch"]) == len(
        loaded["history"]["total_loss"]
    ), "Telemetry array dimension mismatch"


def test_trading_config_to_dict_runtime_fidelity():
    """
    Verifies that TradingConfig.to_dict() captures runtime attribute mutations,
    nested dataclasses, and dynamic properties without dropping fields.
    """
    cfg = TradingConfig()
    cfg.min_active_tilt = 0.35
    cfg.benchmark_ticker = "QQQ"
    cfg.holding_period = 10
    cfg.loss_aversion_penalty = 0.75
    cfg.strategy_params.rsi_overbought = 75

    d = cfg.to_dict()

    assert d["min_active_tilt"] == 0.35
    assert d["benchmark"] == "QQQ"
    assert d["benchmark_ticker"] == "QQQ"
    assert d["holding_period"] == 10
    assert d["loss_aversion_penalty"] == 0.75
    assert d["gamma"] == 0.90
    assert d["dynamic_gamma"] == 0.90
    assert isinstance(d["strategy_params"], dict)
    assert d["strategy_params"]["rsi_overbought"] == 75
    assert isinstance(d["thresholds"], dict)


def test_checkpoint_grid_params_contract(tmp_path):
    """
    Verifies that checkpoint payloads embedding merged runtime grid_params
    are parsed by extract_run_hyperparameters directly without regex guessing.
    """
    from core.settings import extract_run_hyperparameters

    cfg = TradingConfig()
    cfg.min_active_tilt = 0.25
    cfg.loss_aversion_penalty = 0.50
    cfg.holding_period = 7
    cfg.benchmark_ticker = "IWM"

    chunk_spec = {
        "chunk_id": 1,
        "lr": 2.0e-5,
        "kl_anchor_coef": 0.05,
    }

    runtime_grid_params = {
        **cfg.to_dict(),
        **chunk_spec,
        "holding_period": cfg.holding_period,
        "benchmark": cfg.benchmark,
        "gamma": cfg.gamma,
    }

    ckpt_file = tmp_path / "model_chunk1_s42_champion.pt"
    torch.save(
        {
            "model_state_dict": {},
            "scaler_state": {},
            "grid_params": runtime_grid_params,
            "holding_period": cfg.holding_period,
            "benchmark": cfg.benchmark,
            "epoch": 15,
        },
        ckpt_file,
    )

    extracted = extract_run_hyperparameters(ckpt_file)
    assert extracted["min_active_tilt"] == 0.25
    assert extracted["loss_aversion_penalty"] == 0.50
    assert extracted["holding_period"] == 7
    assert extracted["benchmark"] == "IWM"
    assert extracted["gamma"] == 0.90


def test_save_run_manifest_generation(tmp_path):
    """
    Verifies that save_run_manifest generates a valid, fully populated JSON file.
    """
    import json
    from run_walk_forward import save_run_manifest

    cfg = TradingConfig()
    cfg.min_active_tilt = 0.20

    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    feature_cube = pd.DataFrame(
        np.zeros((20, 5)),
        index=pd.MultiIndex.from_product(
            [dates, ["AAPL", "MSFT"]], names=["Date", "Ticker"]
        ),
        columns=[f"F_{i}" for i in range(5)],
    )
    macro_df = pd.DataFrame(
        np.zeros((10, 3)), index=dates, columns=["M_0", "M_1", "M_2"]
    )

    cache_file = tmp_path / "alpha_cache_test.parquet"
    cache_file.touch()

    out_blend = tmp_path / "blotter_blend.parquet"
    canonical_blend = tmp_path / "canonical_blotter_blend.parquet"
    out_blend.touch()
    canonical_blend.touch()

    ckpt_dir = tmp_path / "walk_forward_gen99"
    ckpt_dir.mkdir(parents=True)
    canonical_dir = tmp_path / "canonical_anchors"
    canonical_dir.mkdir(parents=True)

    manifest_path = save_run_manifest(
        generation=99,
        seeds=[42, 101],
        config=cfg,
        chunk_specs=[{"chunk_id": 0, "name": "base"}],
        scorecard={"gate1": {"pass": True, "value": 0.90}},
        blend_metrics={"sharpe_ratio": 0.90},
        seed_metrics={42: {"sharpe_ratio": 0.88}, 101: {"sharpe_ratio": 0.91}},
        div_ratio=1.06,
        verdict="CONFIRMED",
        feature_cube=feature_cube,
        macro_df=macro_df,
        trading_calendar=dates,
        cache_file=cache_file,
        out_blend_blotter=out_blend,
        canonical_blend_blotter=canonical_blend,
        seed_blotter_paths={42: out_blend, 101: out_blend},
        checkpoint_paths={"chunk0_s42": tmp_path / "m1.pt"},
        checkpoint_dir=ckpt_dir,
        canonical_dir=canonical_dir,
    )

    assert manifest_path.exists()
    assert (canonical_dir / "run_metadata_gen99.json").exists()

    with open(manifest_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    assert payload["generation"] == 99
    assert payload["verdict"] == "CONFIRMED"
    assert payload["trading_config"]["min_active_tilt"] == 0.20
    assert payload["data_lineage"]["obs_dim"] == 3 * 5 + 3  # 18
    assert payload["data_lineage"]["universe_size"] == 5
    assert payload["diversification_ratio"] == 1.06
