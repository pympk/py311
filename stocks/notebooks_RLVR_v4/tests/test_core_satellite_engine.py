import numpy as np
import pandas as pd
import pytest
import torch

from core.settings import TradingConfig
from rl_discovery.adapter import ObservationAdapter, RLVRGymEnv
from rl_discovery.agent import AbsoluteZeroAgent
from rl_discovery.environment import DiscoveryEnv
from rl_discovery.trainer import RolloutBuffer

# =====================================================================
# DETERMINISTIC TEST FIXTURES
# =====================================================================


@pytest.fixture
def core_satellite_fixture():
    """
    Constructs a calibrated, deterministic synthetic environment
    to mathematically audit Tri-Asset returns, slippage, and observations.
    """
    config = TradingConfig(
        holding_period=1,
        min_basket_width=1,
        max_cash_pct=1.0,
        min_active_tilt=0.0,
        slippage_rate=0.0010,
        loss_aversion_penalty=0.0,
    )
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    tickers = [f"STOCK_{i}" for i in range(10)]
    benchmark = config.benchmark_ticker
    all_tickers = tickers + [benchmark]

    # 1. 12-Feature Cube (10 Stocks + Benchmark)
    feature_names = [
        "41d_Log Price Gain",
        "41d_Sharpe (TRP)",
        "41d_Momentum (21d)",
        "41d_Info Ratio (63d)",
        "41d_Oversold (-RSI)",
        "41d_Dip Buyer (-dd_21)",
        "41d_Range Position (20d)",
        "41d_Return Autocorr (15d)",
        "41d_Low Volatility (-ATRP)",
        "41d_Slope_P_5_Z",
        "41d_Slope_V_5_Z",
        "41d_Convexity",
    ]

    idx = pd.MultiIndex.from_product([all_tickers, dates], names=["Ticker", "Date"])
    cube_data = np.zeros((len(idx), 12))
    cube = pd.DataFrame(cube_data, index=idx, columns=feature_names)
    cube.loc[benchmark] = 2.0
    for t in tickers:
        cube.loc[t] = 1.0

    # 2. Simple Return Matrix (Stocks = +2.0%, Benchmark = +1.0%, CASH = +0.1%)
    simple_ret_matrix = pd.DataFrame(0.02, index=dates, columns=tickers)
    simple_ret_matrix[benchmark] = 0.010
    simple_ret_matrix["CASH"] = 0.001

    # 3. Clean 10-Column Macro DataFrame
    macro_cols = [
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
    macro_df = pd.DataFrame(0.5, index=dates, columns=macro_cols)
    macro_df["Mkt_Ret"] = 0.010

    return cube, simple_ret_matrix, macro_df, dates, config


# =====================================================================
# TEST TIER 1: TRI-ASSET PORTFOLIO MATH & SLIPPAGE
# =====================================================================


def test_pure_active_alpha_basket(core_satellite_fixture):
    """
    Allocating 100% Equity Exposure (E=1.0) and 100% Active Tilt (B=1.0):
    w_active = 1.0, w_benchmark = 0.0, w_cash = 0.0.
    Step 0 is pending warmup (earns benchmark return, zero slippage, zero alpha).
    Step 1 executes active sleeve: single-stock return (+2.0%) minus full slippage (-0.10%).
    """
    cube, simple_ret_matrix, macro_df, dates, config = core_satellite_fixture
    env = DiscoveryEnv(
        feature_cube=cube,
        simple_ret_matrix=simple_ret_matrix,
        calendar=dates,
        macro_df=macro_df,
        config=config,
    )
    env.reset()

    action = np.zeros(16)
    action[-4] = -1.0  # offset
    action[-3] = 1.0  # width (all stocks)
    action[-2] = 1.0  # E = +1.0
    action[-1] = 1.0  # B = +1.0

    # Step 0: Order is pending. Zero active exposure. Holds benchmark beta shelter.
    _, reward0, terminated0, truncated0, info0 = env.step(action)
    assert not terminated0 and not truncated0
    assert info0["weight_active"] == 1.0
    assert info0["weight_benchmark"] == 0.0
    assert info0["weight_cash"] == 0.0
    assert info0["slippage_daily_simple_loss"] == 0.0
    assert np.isclose(info0["net_daily_simple_ret"], 0.010)
    assert np.isclose(info0["alpha_daily_simple_ret"], 0.0)
    assert np.isclose(reward0, 0.0)

    # Step 1: Pending order executes. Realizes stock returns and turnover slippage.
    _, reward1, terminated1, truncated1, info1 = env.step(action)
    assert info1["weight_active"] == 1.0
    assert info1["weight_benchmark"] == 0.0
    assert info1["weight_cash"] == 0.0
    assert np.isclose(info1["slippage_daily_simple_loss"], 0.0010)

    # Stocks = 2.0%, Slippage = 0.10% -> Net = 1.90%
    # Benchmark = 1.0% -> Alpha = 1.90% - 1.0% = +0.90%
    assert np.isclose(info1["net_daily_simple_ret"], 0.0190)
    assert np.isclose(info1["alpha_daily_simple_ret"], 0.0090)
    assert np.isclose(reward1, 0.0090)


def test_pure_benchmark_beta_shelter(core_satellite_fixture):
    """
    Allocating 100% Equity Exposure (E=1.0) and 0% Active Tilt (B=0.0):
    w_active = 0.0, w_benchmark = 1.0, w_cash = 0.0.
    Proves that holding the benchmark produces ZERO tracking error, ZERO slippage, and ZERO alpha.
    """
    cube, simple_ret_matrix, macro_df, dates, config = core_satellite_fixture
    env = DiscoveryEnv(
        feature_cube=cube,
        simple_ret_matrix=simple_ret_matrix,
        calendar=dates,
        macro_df=macro_df,
        config=config,
    )
    env.reset()

    # Action: E=+1.0 (index -2), B=-1.0 (index -1 -> 0.0 active tilt)
    action = np.zeros(16)
    action[-4] = -1.0
    action[-3] = 1.0
    action[-2] = 1.0
    action[-1] = -1.0

    # Step 0: Warmup
    env.step(action)
    # Step 1: Benchmark Shelter Active
    _, reward, _, _, info = env.step(action)

    assert info["weight_active"] == 0.0
    assert info["weight_benchmark"] == 1.0
    assert info["weight_cash"] == 0.0
    assert info["slippage_daily_simple_loss"] == 0.0

    assert np.isclose(info["net_daily_simple_ret"], 0.010)
    assert np.isclose(info["alpha_daily_simple_ret"], 0.0)
    assert np.isclose(reward, 0.0)


def test_pure_risk_off_cash_in_market_crash(core_satellite_fixture):
    """
    Allocating 0% Equity Exposure (E=0.0):
    w_active = 0.0, w_benchmark = 0.0, w_cash = 1.0.
    In a crashing market (Benchmark = -5.0%), Cash (+0.1%) generates +5.1% Pure Positive Alpha.
    """
    cube, simple_ret_matrix, macro_df, dates, config = core_satellite_fixture
    benchmark = config.benchmark_ticker
    simple_ret_matrix[benchmark] = -0.050

    env = DiscoveryEnv(
        feature_cube=cube,
        simple_ret_matrix=simple_ret_matrix,
        calendar=dates,
        macro_df=macro_df,
        config=config,
    )
    env.reset()

    # Action: E=-1.0 (index -2 -> 0.0 equity exposure, 100% Cash)
    action = np.zeros(16)
    action[-2] = -1.0

    _, reward, _, _, info = env.step(action)

    assert info["weight_active"] == 0.0
    assert info["weight_benchmark"] == 0.0
    assert info["weight_cash"] == 1.0
    assert info["slippage_daily_simple_loss"] == 0.0

    # Return = Cash Return (+0.1%)
    # Alpha = +0.001 - (-0.050) = +0.051 (+5.1% Alpha)
    assert np.isclose(info["net_daily_simple_ret"], 0.001)
    assert np.isclose(info["alpha_daily_simple_ret"], 0.051)
    assert np.isclose(reward, 0.051)


def test_balanced_core_satellite_split(core_satellite_fixture):
    """
    Audits proportional arithmetic:
    E = 0.80 (80% Equity, 20% Cash)
    B = 0.50 (50% of Equity in Active, 50% in Benchmark)
    Result: w_active = 0.40, w_benchmark = 0.40, w_cash = 0.20.
    """
    cube, simple_ret_matrix, macro_df, dates, config = core_satellite_fixture
    env = DiscoveryEnv(
        feature_cube=cube,
        simple_ret_matrix=simple_ret_matrix,
        calendar=dates,
        macro_df=macro_df,
        config=config,
    )
    env.reset()

    # Action: offset=-1.0, width=1.0, E=0.60 (interp to 0.80), B=0.0 (interp to 0.50)
    action = np.zeros(16)
    action[-4] = -1.0
    action[-3] = 1.0
    action[-2] = 0.60
    action[-1] = 0.00

    # Step 0: Warmup
    env.step(action)
    # Step 1: Active
    _, reward, _, _, info = env.step(action)

    assert np.isclose(info["weight_active"], 0.40)
    assert np.isclose(info["weight_benchmark"], 0.40)
    assert np.isclose(info["weight_cash"], 0.20)

    # Gross Return = (0.40 * 0.02) + (0.40 * 0.01) + (0.20 * 0.001) = 0.0122
    # Slippage = 0.40 * 0.0010 = 0.0004
    # Net Sleeve Return = 0.0122 - 0.0004 = 0.0118 (1.18%)
    # Alpha = 0.0118 - 0.010 = +0.0018 (+0.18% Alpha)
    assert np.isclose(info["slippage_daily_simple_loss"], 0.0004)
    assert np.isclose(info["net_daily_simple_ret"], 0.0118)
    assert np.isclose(info["alpha_daily_simple_ret"], 0.0018)
    assert np.isclose(reward, 0.0018)


def test_asymmetric_loss_aversion_penalty(core_satellite_fixture):
    """
    Ensures negative alpha is scaled by (1 + loss_aversion_penalty),
    while positive alpha remains unpenalized.
    """
    cube, simple_ret_matrix, macro_df, dates, config = core_satellite_fixture
    config.loss_aversion_penalty = 1.5  # Multiplier: 1.0 + 1.5 = 2.5

    env = DiscoveryEnv(
        feature_cube=cube,
        simple_ret_matrix=simple_ret_matrix,
        calendar=dates,
        macro_df=macro_df,
        config=config,
    )
    env.reset()

    # Force Negative Alpha: 100% Cash when Market is UP (+1.0%)
    action_cash = np.zeros(16)
    action_cash[-2] = -1.0  # 100% Cash

    _, reward_neg, _, _, info_neg = env.step(action_cash)

    # Alpha = 0.001 (Cash) - 0.010 (Benchmark) = -0.009 (-0.9% Alpha)
    # Penalized Alpha = -0.009 * (1.0 + 1.5) = -0.0225
    assert np.isclose(info_neg["alpha_daily_simple_ret"], -0.009)
    assert np.isclose(info_neg["penalized_alpha_daily_simple_ret"], -0.0225)
    assert np.isclose(reward_neg, -0.0225)


# =====================================================================
# TEST TIER 2: 46-DIM OBSERVATION INTEGRITY & BENCHMARK ALIGNMENT
# =====================================================================


def test_observation_tensor_dimension_and_benchmark_alignment(core_satellite_fixture):
    """
    Verifies that the observation space contains EXACTLY 46 dimensions:
    [0..11]   Universe Mean
    [12..23]  Universe Std
    [24..35]  Benchmark Vector
    [36..45]  Macro Context
    And verifies Benchmark features are precisely aligned.
    """
    cube, simple_ret_matrix, macro_df, dates, config = core_satellite_fixture
    env = DiscoveryEnv(
        feature_cube=cube,
        simple_ret_matrix=simple_ret_matrix,
        calendar=dates,
        macro_df=macro_df,
        config=config,
    )
    gym_env = RLVRGymEnv(env, macro_df)

    obs, _ = gym_env.reset()

    assert obs.shape == (46,), f"Expected 46 observation dimensions, got {obs.shape}"
    assert gym_env.observation_space.shape == (46,)
    assert gym_env.action_space.shape == (16,)

    # Direct Raw Observation Verification via ObservationAdapter
    obs_dict = env._get_observation()
    raw_obs = ObservationAdapter.process(
        ensemble=obs_dict["ensemble"],
        macro_row=obs_dict["macro_row"],
        expected_strats=12,
        bm_row=obs_dict["bm_row"],
    )

    assert len(raw_obs) == 46
    assert isinstance(raw_obs, np.ndarray)
    assert raw_obs.dtype == np.float32

    # Benchmark slice is indices [24:36]
    benchmark_slice = raw_obs[24:36]
    assert np.allclose(
        benchmark_slice, 2.0
    ), "Benchmark feature vector in observation does not match cube data"


def test_agent_forward_pass_46_to_16():
    """
    Audits the Neural Network architecture:
    Accepts 46-dim observations and outputs 16-dim actions.
    """
    agent = AbsoluteZeroAgent(obs_dim=46, action_dim=16, hidden_size=256)
    dummy_obs = torch.randn(8, 46)

    action, logprob, entropy, value = agent.get_action_and_value(dummy_obs)

    assert action.shape == (8, 16)
    assert logprob.shape == (8,)
    assert entropy.shape == (8,)
    assert value.shape == (8, 1)


def test_rollout_buffer_allocation_46_to_16():
    """
    Verifies RolloutBuffer correctly initializes and logs steps under
    terminations and truncations separation.
    """
    buffer = RolloutBuffer(
        num_steps=16, num_envs=4, obs_dim=46, action_dim=16, gamma=0.90
    )

    assert buffer.obs.shape == (16, 4, 46)
    assert buffer.actions.shape == (16, 4, 16)
    assert buffer.rewards.shape == (16, 4)

    buffer.add(
        obs=np.zeros((4, 46)),
        action=torch.zeros((4, 16)),
        logprob=torch.zeros(4),
        reward=np.ones(4),
        value=torch.zeros((4, 1)),
        terminations=np.zeros(4),
        truncations=np.zeros(4),
    )

    assert buffer.step == 1
