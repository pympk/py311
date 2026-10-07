from collections import deque

import numpy as np
import pandas as pd
import pytest
import torch

from core.accounting import MTMPortfolioEngine
from core.logic import SelectionLogic
from core.quant import QuantUtils
from core.settings import TradingConfig
from rl_discovery.adapter import RLVRGymEnv
from rl_discovery.environment import DiscoveryEnv
from rl_discovery.validator import AgentEvaluator


# ----------------------------------------------------------------------
# 1. FIXTURES: Minimal Deterministic Environment & Synthetic Market
# ----------------------------------------------------------------------
@pytest.fixture
def mock_mtm_setup():
    """
    Creates a deterministic 6-day market with dynamic benchmark configuration:
    - TICKER_A: Outperformer
    - TICKER_B: Mean-reverting / Volatile
    - Dynamic Benchmark: Ticker defined in TradingConfig (e.g. SPY)
    """
    config = TradingConfig()
    benchmark = config.benchmark_ticker
    tickers = ["TICKER_A", "TICKER_B", benchmark]

    dates = pd.date_range(start="2024-01-01", periods=6, freq="B")

    # 1. MultiIndex Feature Cube (Date x Ticker)
    idx = pd.MultiIndex.from_product([dates, tickers], names=["Date", "Ticker"])
    cube = pd.DataFrame(
        0.0,
        index=idx,
        columns=[f"feat_{i}" for i in range(12)],
    )

    # 2. Deterministic Daily Simple Returns
    daily_returns = pd.DataFrame(
        [
            [0.020, 0.040, 0.010],
            [0.010, -0.020, 0.005],
            [0.030, 0.010, 0.000],
            [-0.010, 0.050, -0.005],
            [0.000, 0.000, 0.000],
            [0.020, 0.020, 0.010],
        ],
        index=dates,
        columns=tickers,
    )
    daily_returns["CASH"] = 0.0

    # Stationary macro mock (10 dimensions)
    macro_df = pd.DataFrame(0.0, index=dates, columns=[f"macro_{i}" for i in range(10)])

    config.holding_period = 3
    config.rank_max_width = 1
    config.min_basket_width = 1
    config.slippage_rate = 0.0030  # 30 bps roundtrip -> 10 bps amortized per day
    config.loss_aversion_penalty = 0.0

    env = DiscoveryEnv(
        feature_cube=cube,
        simple_ret_matrix=daily_returns,
        calendar=dates,
        macro_df=macro_df,
        config=config,
        randomize_start=False,
        episode_steps=0,
    )

    return env, daily_returns, dates, config, macro_df


# ----------------------------------------------------------------------
# 2. TEST CASES: ENVIRONMENT & MTM ENGINE
# ----------------------------------------------------------------------


def test_fifo_queue_capacity_and_eviction(mock_mtm_setup):
    """
    INVARIANT 1: FIFO queue must never exceed holding_period (H=3).
    Verifies T+1 pending order state machine:
    - Step 0: Order queued (0 active sleeves)
    - Step 1: 1 active sleeve
    - Step 2: 2 active sleeves
    - Step 3: 3 active sleeves (Capacity)
    - Step 4: 3 active sleeves (Day 0 sleeve evicted)
    """
    env, _, _, config, _ = mock_mtm_setup
    env.reset()
    assert len(env.active_sleeves) == 0

    action_pick_top = np.zeros(16)
    action_pick_top[-4:] = [-1.0, 1.0, 1.0, 1.0]

    # Step 0 -> Pending order placed (0 active sleeves)
    env.step(action_pick_top)
    assert len(env.active_sleeves) == 0

    # Step 1 -> 1 active sleeve
    env.step(action_pick_top)
    assert len(env.active_sleeves) == 1

    # Step 2 -> 2 active sleeves
    env.step(action_pick_top)
    assert len(env.active_sleeves) == 2

    # Step 3 -> 3 active sleeves (Capacity)
    env.step(action_pick_top)
    assert len(env.active_sleeves) == 3

    # Step 4 -> 3 active sleeves (Evicts Step 0 sleeve)
    env.step(action_pick_top)
    assert len(env.active_sleeves) == 3


def test_step_zero_warmup_and_net_slippage_e2e(mock_mtm_setup):
    """
    INVARIANT 2: Step 0 is pending warmup (100% benchmark return, zero slippage, zero alpha).
    Step 1 executes the active sleeve: 1/H partition on stock + (H-1)/H on benchmark minus slippage.
    """
    env, daily_rets, dates, config, _ = mock_mtm_setup
    env.reset()

    benchmark = config.benchmark_ticker
    H = config.holding_period
    expected_slippage = config.slippage_rate / H  # 0.0030 / 3 = 0.0010 (10 bps)

    # Force action to pick TICKER_A
    action_pick_a = np.zeros(16)
    action_pick_a[-4:] = [-1.0, 1.0, 1.0, 1.0]

    # --- STEP 0: PENDING ORDER WARMUP ---
    obs0, reward0, term0, trunc0, info0 = env.step(action_pick_a)

    expected_bm_ret_0 = daily_rets.loc[dates[0], benchmark]  # 0.010
    assert info0["slippage_daily_simple_loss"] == pytest.approx(0.0)
    assert info0["net_daily_simple_ret"] == pytest.approx(expected_bm_ret_0)
    assert info0["alpha_daily_simple_ret"] == pytest.approx(0.0)
    assert reward0 == pytest.approx(0.0)
    assert env.equity_curve[-1] == pytest.approx(1.0 * (1.0 + expected_bm_ret_0))
    assert env.alpha_equity_curve[-1] == pytest.approx(1.0)

    # --- STEP 1: SLEEVE ACTIVE EXECUTION ---
    obs1, reward1, term1, trunc1, info1 = env.step(action_pick_a)

    # Day 1 Returns: TICKER_A = +0.010, Benchmark = +0.005
    # Active Stock Return = (1 * 0.010 + 2 * 0.005) / 3 = 0.020 / 3 = 0.0066667
    # Net Daily Simple Return = (0.020 / 3) - 0.0010 = 0.0056667
    # Alpha = 0.0056667 - 0.005 = 0.0006667
    expected_gross_stock = (0.010 + (H - 1) * 0.005) / H
    expected_net_ret = expected_gross_stock - expected_slippage
    expected_alpha = expected_net_ret - 0.005

    assert info1["gross_stock_daily_simple_ret"] == pytest.approx(expected_gross_stock)
    assert info1["slippage_daily_simple_loss"] == pytest.approx(expected_slippage)
    assert info1["net_daily_simple_ret"] == pytest.approx(expected_net_ret)
    assert info1["alpha_daily_simple_ret"] == pytest.approx(expected_alpha)
    assert reward1 == pytest.approx(expected_alpha)


def test_slippage_is_deducted_from_equity_curve(mock_mtm_setup):
    """
    INVARIANT 3: Verifies that equity_curve strictly reflects Net returns
    (Gross return minus transaction slippage).
    """
    env, _, _, config, _ = mock_mtm_setup
    env.config.slippage_rate = 0.0060  # 60 bps -> 20 bps amortized per executing sleeve
    env.reset()

    action = np.zeros(16)
    action[-4:] = [-1.0, 1.0, 1.0, 1.0]

    # Step 0: Warmup
    env.step(action)
    equity_step0 = env.equity_curve[-1]

    # Step 1: Active execution
    _, _, _, _, info1 = env.step(action)
    gross_ret = info1["gross_daily_simple_ret"]
    net_ret = info1["net_daily_simple_ret"]
    slippage = info1["slippage_daily_simple_loss"]

    assert slippage == pytest.approx(0.0060 / config.holding_period)
    assert net_ret == pytest.approx(gross_ret - slippage)
    assert env.equity_curve[-1] == pytest.approx(equity_step0 * (1.0 + net_ret))
    assert env.equity_curve[-1] < (equity_step0 * (1.0 + gross_ret))


def test_zero_tracking_error_on_pure_benchmark_selection(mock_mtm_setup):
    """
    INVARIANT 4: If the agent selects the dynamic benchmark every day with 0 slippage,
    the active excess reward must be identically 0.0 on every step.
    """
    env, daily_rets, dates, config, _ = mock_mtm_setup
    benchmark = config.benchmark_ticker
    engine = MTMPortfolioEngine(config=config)
    engine.config.slippage_rate = 0.0

    # Pre-fill active sleeves with benchmark
    engine.active_sleeves = deque([[benchmark], [benchmark], [benchmark]], maxlen=3)

    for date in dates[:4]:
        bm_ret = float(daily_rets.loc[date, benchmark])
        res = engine.step(
            selected_tickers=[benchmark],
            equity_exposure=1.0,
            active_tilt=1.0,
            stock_simple_rets={benchmark: bm_ret},
            bm_daily_simple_ret=bm_ret,
        )
        assert res.alpha_daily_simple_ret == pytest.approx(0.0, abs=1e-9)


def test_empty_basket_safe_handling(mock_mtm_setup):
    """
    INVARIANT 5: If an action produces an empty selection ([]),
    sleeve return defaults to 0.0 without slippage or NaN crash.
    """
    engine = MTMPortfolioEngine()
    engine.reset()

    sleeve_ret = engine.calculate_sleeve_simple_ret([], {"AAPL": 0.05})
    assert sleeve_ret == 0.0


# ----------------------------------------------------------------------
# 3. TEST CASES: VALIDATOR INTEGRATION & SHARPE ACCURACY
# ----------------------------------------------------------------------


class MockActorCritic(torch.nn.Module):
    """Deterministic mock agent conforming to get_deterministic_action and get_value contracts."""

    def __init__(self, action_dim: int = 16):
        super().__init__()
        self.action_dim = action_dim

    def forward(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        act = torch.zeros((x.shape[0], self.action_dim), dtype=torch.float32)
        act[:, -4:] = torch.tensor([-1.0, 1.0, 1.0, 1.0])
        return act

    def get_deterministic_action(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, deterministic=True)

    def actor_mean(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_deterministic_action(x)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros((x.shape[0], 1), dtype=torch.float32)


def test_validator_evaluator_sharpe_accuracy(mock_mtm_setup):
    """
    INVARIANT 6: Verifies AgentEvaluator.evaluate() computes Sharpe and Information Ratio
    accurately on daily net simple returns without double scaling.
    """
    env, _, dates, _, macro_df = mock_mtm_setup
    gym_env = RLVRGymEnv(discovery_env=env, macro_df=macro_df)
    mock_agent = MockActorCritic(action_dim=16)

    results = AgentEvaluator.evaluate(
        agent=mock_agent,
        env=gym_env,
        device=torch.device("cpu"),
        detailed_log=True,
    )

    # 1. Check result keys
    assert "total_return" in results
    assert "sharpe_ratio" in results
    assert "information_ratio" in results
    assert "blotter" in results

    # 2. Check blotter length matches execution steps (dates - 2)
    blotter = results["blotter"]
    assert len(blotter) == len(dates) - 2

    # 3. Verify manual Sharpe matching on net_daily_simple_ret
    actual_returns = pd.Series([row["net_daily_simple_ret"] for row in blotter])
    expected_sharpe = (actual_returns.mean() / actual_returns.std(ddof=1)) * np.sqrt(
        252
    )

    assert results["sharpe_ratio"] == pytest.approx(float(expected_sharpe), abs=1e-5)


# ----------------------------------------------------------------------
# 4. TEST CASES: INTRA-SLEEVE RISK PARITY & SELECTION LOGIC
# ----------------------------------------------------------------------


def test_intra_sleeve_inverse_volatility_weighting():
    """
    INVARIANT 7: Intra-sleeve weights must strictly equal inverse volatility:
    Stock A: ATRP = 0.01 (1%), Stock B: ATRP = 0.04 (4%)
    Inv Vol: A = 100, B = 25 -> Normalized: A = 0.80, B = 0.20
    """
    vols = np.array([0.01, 0.04])
    weights = QuantUtils.calculate_inv_vol_weights(vols)

    assert weights[0] == pytest.approx(0.80, abs=1e-6)
    assert weights[1] == pytest.approx(0.20, abs=1e-6)
    assert np.sum(weights) == pytest.approx(1.0, abs=1e-6)

    # Calculate weighted sleeve return: A earns +2%, B drops -1%
    # Expected sleeve return = 0.80 * 0.02 + 0.20 * (-0.01) = 0.016 - 0.002 = 0.014
    sleeve = {"TICKER_A": weights[0], "TICKER_B": weights[1]}
    rets = {"TICKER_A": 0.02, "TICKER_B": -0.01}
    sleeve_ret = MTMPortfolioEngine.calculate_sleeve_simple_ret(sleeve, rets)
    assert sleeve_ret == pytest.approx(0.014, abs=1e-6)


def test_intra_sleeve_delisting_weight_renormalization():
    """
    INVARIANT 8: If an asset in a risk-parity sleeve has NaN return (missing/delisted),
    surviving weights are renormalized to 1.0 without capital leakage.
    """
    sleeve = {"A": 0.50, "B": 0.30, "C": 0.20}
    # C is delisted/missing
    rets = {"A": 0.04, "B": 0.02, "C": np.nan}

    # Surviving weights: A = 0.50/0.80 = 0.625, B = 0.30/0.80 = 0.375
    # Expected return = 0.625 * 0.04 + 0.375 * 0.02 = 0.025 + 0.0075 = 0.0325
    sleeve_ret = MTMPortfolioEngine.calculate_sleeve_simple_ret(sleeve, rets)
    assert sleeve_ret == pytest.approx(0.0325, abs=1e-6)


def test_selection_logic_compute_intra_sleeve_weights():
    """
    INVARIANT 9: SelectionLogic extracts ATRP and maps to normalized weights.
    """
    ensemble = pd.DataFrame(
        {
            "feat_0": [0.1, 0.2],
            "ATRP": [0.02, 0.08],
        },
        index=["TICKER_X", "TICKER_Y"],
    )
    weights = SelectionLogic.compute_intra_sleeve_weights(
        ["TICKER_X", "TICKER_Y"], ensemble
    )

    # ATRP: 0.02 vs 0.08 -> Inv: 50 vs 12.5 -> Total: 62.5 -> X=0.80, Y=0.20
    assert weights["TICKER_X"] == pytest.approx(0.80, abs=1e-6)
    assert weights["TICKER_Y"] == pytest.approx(0.20, abs=1e-6)
