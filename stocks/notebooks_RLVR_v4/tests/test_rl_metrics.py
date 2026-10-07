import pytest
import pandas as pd
import numpy as np
from core.settings import TradingConfig
from core.logic import SelectionLogic
from rl_discovery.environment import DiscoveryEnv


@pytest.fixture
def dummy_environment_data():
    """Generates synthetic cross-sectional data, rewards, and macro indices with dynamic benchmark."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
    benchmark = TradingConfig().benchmark_ticker

    # 1. Feature Cube (12 features)
    idx = pd.MultiIndex.from_product([tickers, dates], names=["Ticker", "Date"])
    cube = pd.DataFrame(np.random.randn(len(idx), 12), index=idx)

    # 2. Simple Return Matrix (Flat 1% stock return, 0.5% benchmark return, 0% cash)
    ret_cols = tickers + [benchmark, "CASH"]
    reward_matrix = pd.DataFrame(0.01, index=dates, columns=ret_cols)
    reward_matrix[benchmark] = 0.005
    reward_matrix["CASH"] = 0.0

    # 3. Macro Market Trend
    macro_df = pd.DataFrame({"Mkt_Ret": [0.005] * 10}, index=dates)

    return cube, reward_matrix, macro_df, dates


def test_selection_logic_zero_width(dummy_environment_data):
    """Ensures Portfolio Constraints allow buying 0 stocks when min_basket_width=0."""
    cube, _, _, dates = dummy_environment_data
    ensemble = cube.xs(dates[0], level="Date")
    config = TradingConfig(
        rank_max_width=10,
        min_basket_width=0,
        max_cash_pct=1.0,
        min_active_tilt=0.0,
    )

    num_features = ensemble.shape[1]
    action_size = num_features + 4
    action = np.full(action_size, -1.0)

    (
        selected,
        _,
        _,
        width,
        equity_exposure,
        active_tilt,
        _,
        _,
    ) = SelectionLogic.apply_action(
        ensemble,
        action,
        rank_max_offset_percentile=1.0,
        rank_max_width=config.rank_max_width,
        min_basket_width=config.min_basket_width,
        max_cash_pct=config.max_cash_pct,
        min_active_tilt=config.min_active_tilt,
    )

    assert width == 0
    assert len(selected) == 0
    assert equity_exposure == 0.0
    assert active_tilt == 0.0


def test_environment_alpha_and_slippage(dummy_environment_data):
    """
    Proves slippage deduction and positive alpha under strict T+1 execution.
    - Day T (Step 0): Order placed, zero active sleeves -> benchmark return during cold start.
    - Day T+1 (Step 1): Pending sleeve executes and earns active sleeve returns with slippage.
    """
    cube, reward_matrix, macro_df, dates = dummy_environment_data
    config = TradingConfig(
        slippage_rate=0.0010,
        loss_aversion_penalty=1.0,
        holding_period=1,
        min_basket_width=1,
        max_cash_pct=1.0,
        min_active_tilt=0.0,
    )

    env = DiscoveryEnv(cube, reward_matrix, dates, macro_df, config)
    env.reset()

    num_features = cube.shape[1]
    action = np.full(num_features + 4, 1.0)
    action[-4] = -1.0  # offset = 0
    action[-3] = 1.0  # max width (5 tickers)
    action[-2] = 1.0  # equity_exposure = 1.0
    action[-1] = 1.0  # active_tilt = 1.0

    # STEP 0: Day T Close -> Decision stored in pending_sleeve; 0 active sleeves held
    _, reward0, terminated0, truncated0, info0 = env.step(action)
    assert len(info0["tickers"]) == 5
    assert info0["slippage_daily_simple_loss"] == 0.0
    assert info0["equity_exposure"] == 1.0
    assert info0["weight_active"] == 1.0
    assert info0["weight_benchmark"] == 0.0
    assert info0["weight_cash"] == 0.0
    assert np.isclose(info0["net_daily_simple_ret"], 0.0050)
    assert np.isclose(info0["alpha_daily_simple_ret"], 0.0)
    assert np.isclose(reward0, 0.0)

    # STEP 1: Day T+1 Close -> Pending sleeve executes, becomes active over (T+1 -> T+2)
    _, reward1, terminated1, truncated1, info1 = env.step(action)
    assert len(info1["tickers"]) == 5
    assert np.isclose(info1["slippage_daily_simple_loss"], 0.0010)
    assert info1["equity_exposure"] == 1.0
    assert info1["weight_active"] == 1.0
    assert info1["weight_benchmark"] == 0.0
    assert info1["weight_cash"] == 0.0

    # EXPECTED QUANT REALIZATION:
    # Stock return = 0.01 -> gross_stock_daily_simple_ret = 0.01
    # Slippage = 0.0010 -> net_daily_simple_ret = 0.0090
    # Benchmark = 0.0050 -> alpha = 0.0090 - 0.0050 = +0.0040
    assert np.isclose(info1["net_daily_simple_ret"], 0.0090)
    assert np.isclose(info1["alpha_daily_simple_ret"], 0.0040)
    assert np.isclose(info1["penalized_alpha_daily_simple_ret"], 0.0040)
    assert np.isclose(reward1, 0.0040)

    # Compounding Equity Curves (Strict Geometric Ratio V_p / V_bm):
    # Step 0: V_p = 1.0 * 1.0050 = 1.0050, V_bm = 1.0 * 1.0050 = 1.0050 -> Alpha = 1.0
    # Step 1: V_p = 1.0050 * (1.0 + 0.0090) = 1.014045, V_bm = 1.0050 * (1.0 + 0.0050) = 1.010025
    expected_p_equity = 1.0050 * (1.0 + 0.0090)
    expected_bm_equity = 1.0050 * (1.0 + 0.0050)
    expected_alpha_equity = expected_p_equity / expected_bm_equity
    assert np.isclose(env.equity_curve[-1], expected_p_equity)
    assert np.isclose(env.alpha_equity_curve[-1], expected_alpha_equity)


def test_selection_logic_dynamic_boundaries():
    """Tests continuous actions [-1, 1] interpolation into portfolio hyperparameters."""
    tickers = [f"TICKER_{i}" for i in range(200)]
    ensemble = pd.DataFrame(
        np.random.randn(200, 2), index=tickers, columns=["Feature_1", "Feature_2"]
    )

    num_features = ensemble.shape[1]
    test_percentile = 0.5
    test_width = 20

    # CASE 1: Minimum Action [-1.0]
    action_min = np.full(num_features + 4, -1.0)
    *_, offset, width, equity_exposure, active_tilt, _, _ = SelectionLogic.apply_action(
        ensemble,
        action_min,
        rank_max_offset_percentile=test_percentile,
        rank_max_width=test_width,
        min_basket_width=0,
        max_cash_pct=1.0,
        min_active_tilt=0.0,
    )
    assert offset == 0
    assert width == 0
    assert equity_exposure == 0.0
    assert active_tilt == 0.0

    # CASE 2: Maximum Action [+1.0]
    action_max = np.full(num_features + 4, 1.0)
    *_, offset, width, equity_exposure, active_tilt, _, _ = SelectionLogic.apply_action(
        ensemble,
        action_max,
        rank_max_offset_percentile=test_percentile,
        rank_max_width=test_width,
        min_basket_width=0,
        max_cash_pct=1.0,
        min_active_tilt=0.0,
    )
    assert offset == 100
    assert width == 20
    assert equity_exposure == 1.0
    assert active_tilt == 1.0

    # CASE 3: Neutral Action [0.0]
    action_mid = np.full(num_features + 4, 0.0)
    *_, offset, width, equity_exposure, active_tilt, _, _ = SelectionLogic.apply_action(
        ensemble,
        action_mid,
        rank_max_offset_percentile=test_percentile,
        rank_max_width=test_width,
        min_basket_width=0,
        max_cash_pct=1.0,
        min_active_tilt=0.0,
    )
    assert offset == 50
    assert width == 10
    assert equity_exposure == 0.5
    assert active_tilt == 0.5


def test_environment_downside_penalty_and_cash(dummy_environment_data):
    """Proves Loss Aversion Penalty triggers properly & 100% Cash retreat avoids slippage."""
    cube, reward_matrix, macro_df, dates = dummy_environment_data
    config = TradingConfig(
        slippage_rate=0.0010,
        loss_aversion_penalty=1.0,  # 1.0 penalty -> 2.0x loss multiplier
        holding_period=1,
        min_basket_width=0,
        max_cash_pct=1.0,
        min_active_tilt=0.0,
    )

    env = DiscoveryEnv(cube, reward_matrix, dates, macro_df, config)
    env.reset()

    # Agent forces width to 0 and exposure to 0 (100% Cash)
    num_features = cube.shape[1]
    action = np.full(num_features + 4, -1.0)

    _, reward, terminated, truncated, info = env.step(action)

    assert len(info["tickers"]) == 0
    assert info["slippage_daily_simple_loss"] == 0.0
    assert info["equity_exposure"] == 0.0
    assert info["weight_cash"] == 1.0
    assert info["weight_active"] == 0.0
    assert info["weight_benchmark"] == 0.0

    # EXPECTED QUANT REALIZATION:
    # 100% Cash -> net_daily_simple_ret = 0.0
    # benchmark return = +0.005 -> alpha = -0.005
    # loss multiplier = 1.0 + 1.0 = 2.0 -> penalized_alpha = -0.005 * 2.0 = -0.010
    assert np.isclose(info["net_daily_simple_ret"], 0.0)
    assert np.isclose(info["alpha_daily_simple_ret"], -0.005)
    assert np.isclose(info["penalized_alpha_daily_simple_ret"], -0.010)
    assert np.isclose(reward, -0.010)


def test_environment_dynamic_cash_scaling(dummy_environment_data):
    """Proves 50% Exposure scales equity return and slippage proportionally under T+1 execution."""
    cube, reward_matrix, macro_df, dates = dummy_environment_data
    config = TradingConfig(
        slippage_rate=0.0010,
        loss_aversion_penalty=0.0,
        holding_period=1,
        min_basket_width=1,
        max_cash_pct=1.0,
        min_active_tilt=0.0,
    )

    env = DiscoveryEnv(cube, reward_matrix, dates, macro_df, config)
    env.reset()

    num_features = cube.shape[1]
    action = np.zeros(num_features + 4)
    action[-4] = -1.0  # offset = 0
    action[-3] = 1.0  # width = max
    action[-2] = 0.0  # equity_exposure = 0.50
    action[-1] = 1.0  # active_tilt = 1.0 (100% of equity in active stocks)

    # Step 0: Order queued into pending_sleeve
    env.step(action)

    # Step 1: Active sleeve executes
    _, reward, terminated, truncated, info = env.step(action)

    assert len(info["tickers"]) == 5
    assert np.isclose(info["equity_exposure"], 0.5)
    assert np.isclose(info["weight_active"], 0.5)
    assert np.isclose(info["weight_cash"], 0.5)
    assert np.isclose(info["weight_benchmark"], 0.0)

    # EXPECTED QUANT REALIZATION:
    # Active Stock return = 0.01 * 0.50 = 0.0050
    # Cash return = 0.0 * 0.50 = 0.0
    # Slippage = (0.0010 / 1) * 0.50 = 0.0005
    # Net return = 0.0050 - 0.0005 = 0.0045
    # Benchmark = 0.0050 -> Alpha = 0.0045 - 0.0050 = -0.0005
    assert np.isclose(info["slippage_daily_simple_loss"], 0.0005)
    assert np.isclose(info["net_daily_simple_ret"], 0.0045)
    assert np.isclose(info["alpha_daily_simple_ret"], -0.0005)
    assert np.isclose(reward, -0.0005)


def test_environment_benchmark_allocation(dummy_environment_data):
    """Proves allocating to Benchmark (active_tilt=0.0) yields 0.0 Alpha with zero slippage."""
    cube, reward_matrix, macro_df, dates = dummy_environment_data
    config = TradingConfig(
        slippage_rate=0.0010,
        loss_aversion_penalty=0.0,
        holding_period=1,
        min_basket_width=1,
        max_cash_pct=1.0,
        min_active_tilt=0.0,
    )

    env = DiscoveryEnv(cube, reward_matrix, dates, macro_df, config)
    env.reset()

    num_features = cube.shape[1]
    action = np.zeros(num_features + 4)
    action[-4] = -1.0  # offset = 0
    action[-3] = 1.0  # width = max
    action[-2] = 1.0  # equity_exposure = 1.0
    action[-1] = -1.0  # active_tilt = 0.0 (100% benchmark)

    _, reward, terminated, truncated, info = env.step(action)

    assert np.isclose(info["weight_active"], 0.0)
    assert np.isclose(info["weight_benchmark"], 1.0)
    assert np.isclose(info["weight_cash"], 0.0)
    assert np.isclose(info["slippage_daily_simple_loss"], 0.0)
    assert np.isclose(info["net_daily_simple_ret"], 0.0050)
    assert np.isclose(info["alpha_daily_simple_ret"], 0.0)
    assert np.isclose(reward, 0.0)


def test_selection_logic_factor_variance_equalization():
    """
    Tripwire Invariant: Proves cross-sectional Z-scoring equalizes factor voice.
    A feature with 10,000x raw variance must NOT drown out a micro-scale feature.
    """
    tickers = [f"T{i}" for i in range(100)]
    np.random.seed(42)

    # Feature 0: Massive scale variance (sigma ~ 100.0)
    # Feature 1: Micro scale variance (sigma ~ 0.01)
    f0 = np.random.randn(100) * 100.0
    f1 = np.random.randn(100) * 0.01
    ensemble = pd.DataFrame({"F_Macro": f0, "F_Micro": f1}, index=tickers)

    # Action placing 100% weight on the micro feature (Feature 1)
    # Action structure: [w0, w1, offset, width, equity_exp, active_tilt]
    action = np.array([0.0, 1.0, -1.0, 1.0, 1.0, 1.0])

    selected, top_3, *_, max_s, min_s = SelectionLogic.apply_action(
        ensemble=ensemble,
        action=action,
        rank_max_width=5,
        min_basket_width=5,
    )

    # If unstandardized, F_Macro noise would corrupt sorting.
    # Under Z-scoring, top_3 must identically match F_Micro's highest standardized values.
    expected_top_3 = ensemble["F_Micro"].sort_values(ascending=False).index[:3].tolist()
    assert (
        top_3 == expected_top_3
    ), f"Factor equalization failed: expected {expected_top_3}, got {top_3}"
    assert np.isclose(
        max_s, np.clip((f1 - f1.mean()) / f1.std(), -4.0, 4.0).max(), atol=1e-4
    )
