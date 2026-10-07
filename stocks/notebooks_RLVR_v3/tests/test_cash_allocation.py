# ==============================================================================
# tests/test_cash_allocation.py
# ==============================================================================
import pytest
import numpy as np
import pandas as pd

from core.settings import TradingConfig
from core.logic import SelectionLogic
from rl_discovery.environment import DiscoveryEnv


# ------------------------------------------------------------------------------
# FIXTURES
# ------------------------------------------------------------------------------
@pytest.fixture
def mock_config():
    """Provides a baseline TradingConfig fixture with dynamic fields."""
    cfg = TradingConfig()
    cfg.holding_period = 5
    cfg.max_cash_pct = 0.0
    return cfg


@pytest.fixture
def mock_ensemble():
    """Generates a synthetic 12-feature cross-sectional ensemble for 5 tickers."""
    tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
    features = [f"feat_{i}" for i in range(12)]
    data = np.random.randn(len(tickers), len(features))
    return pd.DataFrame(data, index=tickers, columns=features)


@pytest.fixture
def mock_environment_setup(mock_config):
    """Creates a minimal synthetic environment dataset for integration tests."""
    dates = pd.date_range("2023-01-01", periods=20, freq="B")
    tickers = ["AAPL", "MSFT", "NVDA", mock_config.benchmark_ticker]
    features = [f"feat_{i}" for i in range(12)]

    # 1. Feature Cube (MultiIndex [Date, Ticker])
    idx = pd.MultiIndex.from_product([dates, tickers], names=["Date", "Ticker"])
    cube_data = np.random.randn(len(idx), len(features))
    feature_cube = pd.DataFrame(cube_data, index=idx, columns=features)

    # 2. Simple Return Matrix (Shifted 1-day forward returns + CASH)
    ret_cols = tickers + ["CASH"]
    ret_data = np.random.normal(0.0005, 0.015, size=(len(dates), len(ret_cols)))
    simple_ret_matrix = pd.DataFrame(ret_data, index=dates, columns=ret_cols)
    simple_ret_matrix["CASH"] = 0.0

    # 3. Macro DataFrame
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
    macro_df = pd.DataFrame(
        np.random.randn(len(dates), len(macro_cols)),
        index=dates,
        columns=macro_cols,
    )

    return feature_cube, simple_ret_matrix, dates, macro_df


# ------------------------------------------------------------------------------
# 1. UNIT TESTS: SelectionLogic.apply_action Bounding
# ------------------------------------------------------------------------------
def test_zero_cash_forces_full_equity_exposure(mock_ensemble, mock_config):
    """
    When max_cash_pct = 0.0, equity_exposure must be exactly 1.0 regardless
    of whether the agent outputs minimum (-1.0), neutral (0.0), or maximum (1.0) actions.
    """
    num_features = mock_ensemble.shape[1]

    for exposure_action in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        action = np.zeros(num_features + 4)
        action[-2] = exposure_action  # Equity Exposure control dial
        action[-1] = 0.5  # Active Tilt control dial

        (
            selected,
            top_3,
            offset,
            width,
            equity_exposure,
            active_tilt,
            max_s,
            min_s,
        ) = SelectionLogic.apply_action(
            ensemble=mock_ensemble,
            action=action,
            rank_max_offset_percentile=mock_config.rank_max_offset_percentile,
            rank_max_width=mock_config.rank_max_width,
            max_cash_pct=0.0,
        )

        assert equity_exposure == pytest.approx(1.0, abs=1e-7), (
            f"Expected equity_exposure=1.0 when max_cash_pct=0.0, but got {equity_exposure} "
            f"for raw action {exposure_action}."
        )


def test_bounded_cash_interpolation(mock_ensemble, mock_config):
    """
    When max_cash_pct = 0.10 (10% cash cap), equity_exposure must strictly
    interpolate between [0.90, 1.0].
    """
    num_features = mock_ensemble.shape[1]

    test_cases = [
        (-1.0, 0.90),  # Minimum equity exposure / Maximum cash (10%)
        (0.0, 0.95),  # Midpoint exposure / 5% cash
        (1.0, 1.00),  # Maximum equity exposure / 0% cash
    ]

    for exposure_action, expected_exposure in test_cases:
        action = np.zeros(num_features + 4)
        action[-2] = exposure_action
        action[-1] = 0.0

        *_, equity_exposure, _, _, _ = SelectionLogic.apply_action(
            ensemble=mock_ensemble,
            action=action,
            rank_max_offset_percentile=mock_config.rank_max_offset_percentile,
            rank_max_width=mock_config.rank_max_width,
            max_cash_pct=0.10,
        )

        assert equity_exposure == pytest.approx(expected_exposure, abs=1e-6), (
            f"Action {exposure_action} with max_cash_pct=0.10 resulted in {equity_exposure}, "
            f"expected {expected_exposure}."
        )


def test_legacy_cash_backward_compatibility(mock_ensemble, mock_config):
    """
    When max_cash_pct = 1.0, equity_exposure must span the full legacy range [0.0, 1.0].
    """
    num_features = mock_ensemble.shape[1]

    action_min = np.zeros(num_features + 4)
    action_min[-2] = -1.0
    *_, exp_min, _, _, _ = SelectionLogic.apply_action(
        ensemble=mock_ensemble,
        action=action_min,
        max_cash_pct=1.0,
    )
    assert exp_min == pytest.approx(0.0, abs=1e-7)

    action_max = np.zeros(num_features + 4)
    action_max[-2] = 1.0
    *_, exp_max, _, _, _ = SelectionLogic.apply_action(
        ensemble=mock_ensemble,
        action=action_max,
        max_cash_pct=1.0,
    )
    assert exp_max == pytest.approx(1.0, abs=1e-7)


# ------------------------------------------------------------------------------
# 2. INTEGRATION TESTS: DiscoveryEnv End-to-End Accounting
# ------------------------------------------------------------------------------
def test_environment_step_zero_cash_guarantee(mock_environment_setup, mock_config):
    """
    Verifies that stepping through DiscoveryEnv with config.max_cash_pct = 0.0
    strictly produces w_cash == 0.0 and weight_active + weight_benchmark == 1.0
    under Gymnasium 5-tuple step protocol.
    """
    feature_cube, simple_ret_matrix, calendar, macro_df = mock_environment_setup
    mock_config.max_cash_pct = 0.0

    env = DiscoveryEnv(
        feature_cube=feature_cube,
        simple_ret_matrix=simple_ret_matrix,
        calendar=calendar,
        macro_df=macro_df,
        config=mock_config,
    )

    env.reset()
    num_features = feature_cube.shape[1]

    tilt_actions = [-1.0, -0.5, 0.0, 0.5, 1.0]
    for tilt in tilt_actions:
        action = np.random.uniform(-1.0, 1.0, size=num_features + 4)
        action[-2] = -1.0  # Attempt to force 100% cash
        action[-1] = tilt  # Active vs benchmark tilt

        obs, reward, terminated, truncated, info = env.step(action)

        w_active = info["weight_active"]
        w_bench = info["weight_benchmark"]
        w_cash = info["weight_cash"]

        assert w_cash == pytest.approx(
            0.0, abs=1e-7
        ), f"w_cash must be exactly 0.0 when max_cash_pct=0.0, got {w_cash}"
        assert (w_active + w_bench) == pytest.approx(
            1.0, abs=1e-6
        ), f"Simplex budget violated: w_active ({w_active}) + w_bench ({w_bench}) != 1.0"
        assert info["equity_exposure"] == pytest.approx(1.0, abs=1e-6)
        assert w_active >= 0.0 and w_bench >= 0.0


def test_environment_step_bounded_cash_allocation(mock_environment_setup, mock_config):
    """
    Verifies that setting config.max_cash_pct = 0.05 caps cash at exactly 5%
    under Gymnasium 5-tuple step protocol.
    """
    feature_cube, simple_ret_matrix, calendar, macro_df = mock_environment_setup
    mock_config.max_cash_pct = 0.05

    env = DiscoveryEnv(
        feature_cube=feature_cube,
        simple_ret_matrix=simple_ret_matrix,
        calendar=calendar,
        macro_df=macro_df,
        config=mock_config,
    )

    env.reset()
    num_features = feature_cube.shape[1]

    # Action demanding max cash (-1.0) and 50/50 active/benchmark tilt (0.0)
    action = np.zeros(num_features + 4)
    action[-2] = -1.0
    action[-1] = 0.0

    _, _, _, _, info = env.step(action)

    assert info["weight_cash"] == pytest.approx(0.05, abs=1e-6)
    assert info["equity_exposure"] == pytest.approx(0.95, abs=1e-6)
    assert (
        info["weight_active"] + info["weight_benchmark"] + info["weight_cash"]
    ) == pytest.approx(1.0, abs=1e-6)
