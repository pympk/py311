# In tests/test_mtm_invariants.py
import numpy as np
import pandas as pd
import pytest
from core.accounting import MTMPortfolioEngine
from core.settings import TradingConfig


def test_tripwire_t_plus_one_execution_lag():
    """TRIPWIRE: If stock doubles on Day 0 -> Day 1, portfolio CANNOT earn it on Day 0."""
    config = TradingConfig(holding_period=5, slippage_rate=0.0)
    engine = MTMPortfolioEngine(config)

    # Day 0 -> Day 1: AAPL spikes +100%, SPY is flat (0.0)
    day0_rets = {"AAPL": 1.00, "SPY": 0.0, "CASH": 0.0}
    # Day 1 -> Day 2: AAPL flat (0.0), SPY flat (0.0)
    day1_rets = {"AAPL": 0.00, "SPY": 0.0, "CASH": 0.0}

    # Step 0: Agent at T_0 selects AAPL
    res0 = engine.step(
        selected_tickers=["AAPL"],
        equity_exposure=1.0,
        active_tilt=1.0,
        stock_simple_rets=day0_rets,
        bm_daily_simple_ret=0.0,
    )

    # INVARIANT 1: Day 0 return MUST be 0.0 (Benchmark), NOT 1.0!
    assert res0.gross_stock_daily_simple_ret == pytest.approx(0.0), (
        f"CRITICAL LOOKAHEAD REGRESSION: Day 0 earned {res0.gross_stock_daily_simple_ret} "
        f"instead of 0.0! Order executed at T+0 instead of T+1."
    )
    assert res0.alpha_daily_simple_ret == pytest.approx(0.0)

    # Step 1: Holding Day 1 (T_1 -> T_2)
    res1 = engine.step(
        selected_tickers=["AAPL"],
        equity_exposure=1.0,
        active_tilt=1.0,
        stock_simple_rets=day1_rets,
        bm_daily_simple_ret=0.0,
    )

    # INVARIANT 2: On Day 1 -> Day 2, AAPL return was 0.0, active return must be 0.0
    assert res1.gross_stock_daily_simple_ret == pytest.approx(0.0)


def test_tripwire_constant_one_over_h_sleeve_allocation():
    """TRIPWIRE: A single active sleeve must have exactly 1/H weight, not 1.0."""
    H = 5
    config = TradingConfig(holding_period=H, slippage_rate=0.0)
    engine = MTMPortfolioEngine(config)

    # Day 0: Select stock XYZ
    engine.step(["XYZ"], 1.0, 1.0, {"XYZ": 0.0}, bm_daily_simple_ret=0.0)

    # Day 1: XYZ gains +10% over Day 1 -> Day 2. Benchmark is 0.0.
    # Portfolio gross return must be exactly (+10% / H) = +2.0%, NOT +10.0%!
    res1 = engine.step(["XYZ"], 1.0, 1.0, {"XYZ": 0.10}, bm_daily_simple_ret=0.0)

    expected_gross = 0.10 / H  # 0.02
    assert res1.gross_stock_daily_simple_ret == pytest.approx(
        expected_gross, rel=1e-4
    ), (
        f"CONCENTRATION REGRESSION: Single sleeve earned {res1.gross_stock_daily_simple_ret} "
        f"instead of {expected_gross} (1/{H} partition)."
    )


import torch
from rl_discovery.trainer import RolloutBuffer


def test_tripwire_gae_truncation_bootstrapping():
    """TRIPWIRE: Truncation must NOT zero out future value in GAE advantage."""
    buf = RolloutBuffer(
        num_steps=2,
        num_envs=1,
        obs_dim=4,
        action_dim=2,
        gamma=0.90,
        gae_lambda=0.95,
    )

    # Step 0: reward = 1.0, value = 1.0, termination = 0.0, truncation = 0.0
    buf.add(
        np.zeros((1, 4)),
        torch.zeros((1, 2)),
        torch.zeros(1),
        np.array([1.0]),
        torch.tensor([1.0]),
        np.array([0.0]),
        np.array([0.0]),
    )
    # Step 1: reward = 1.0, value = 1.0, termination = 0.0, TRUNCATION = 1.0
    buf.add(
        np.zeros((1, 4)),
        torch.zeros((1, 2)),
        torch.zeros(1),
        np.array([1.0]),
        torch.tensor([1.0]),
        np.array([0.0]),
        np.array([1.0]),
    )

    next_val = torch.tensor([10.0])  # Estimated V(s_2)
    next_term = torch.tensor([0.0])  # Truncated, NOT terminated

    buf.compute_advantages(next_val, next_term)

    # Advantage at step 1: delta = r_1 + gamma * V(s_2) - V(s_1) = 1.0 + 0.90*10.0 - 1.0 = 9.0
    assert buf.advantages[1, 0].item() == pytest.approx(9.0), (
        f"GAE BOOTSTRAP REGRESSION: Advantage was {buf.advantages[1, 0].item()} "
        f"instead of 9.0. Bootstrapping was zeroed out!"
    )
