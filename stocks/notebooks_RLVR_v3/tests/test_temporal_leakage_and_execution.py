import numpy as np
import pandas as pd
import pytest
from core.accounting import MTMPortfolioEngine
from core.quant import QuantUtils
from core.settings import TradingConfig
from rl_discovery.environment import DiscoveryEnv


def test_mtm_staggered_sleeve_forward_execution():
    """
    Verifies T+1 Next-Day Close execution lag and 1/H portfolio ramp-up partitioning:
    - Step 0 (Decision at T_0): AAPL is queued into pending_sleeve. Over T_0 -> T_1, active sleeves = 0,
      earning benchmark return (0.0).
    - Step 1 (Decision at T_1): AAPL executes into active_sleeves (1 active, H-1 unfilled hold benchmark).
      MSFT is queued into pending_sleeve.
    - Step 2 (Decision at T_2): MSFT executes into active_sleeves (2 active, H-2 unfilled hold benchmark).
    """
    config = TradingConfig(holding_period=3, slippage_rate=0.0)
    H = config.holding_period
    engine = MTMPortfolioEngine(config=config)

    # 1. Forward returns for Step 0 (T_0 -> T_1)
    rets_step_0 = pd.Series({"AAPL": 0.05, "MSFT": -0.02})

    # Step 0: Agent picks AAPL
    res0 = engine.step(
        selected_tickers=["AAPL"],
        equity_exposure=1.0,
        active_tilt=1.0,
        stock_simple_rets=rets_step_0,
        bm_daily_simple_ret=0.0,
    )

    # INVARIANT: Order is pending; active sleeves count is 0, return is pure benchmark (0.0)
    assert len(engine.active_sleeves) == 0
    assert engine.pending_sleeve == ["AAPL"]
    assert pytest.approx(res0.gross_stock_daily_simple_ret, rel=1e-5) == 0.0
    assert pytest.approx(res0.alpha_daily_simple_ret, rel=1e-5) == 0.0

    # 2. Forward returns for Step 1 (T_1 -> T_2)
    rets_step_1 = pd.Series({"AAPL": 0.02, "MSFT": 0.04})

    # Step 1: Agent picks MSFT
    res1 = engine.step(
        selected_tickers=["MSFT"],
        equity_exposure=1.0,
        active_tilt=1.0,
        stock_simple_rets=rets_step_1,
        bm_daily_simple_ret=0.0,
    )

    # INVARIANT: AAPL is now active (1/H weight), MSFT is pending. Unfilled (H-1) hold benchmark (0.0).
    assert len(engine.active_sleeves) == 1
    assert engine.pending_sleeve == ["MSFT"]
    expected_ret_step_1 = 0.02 / H
    assert (
        pytest.approx(res1.gross_stock_daily_simple_ret, rel=1e-5)
        == expected_ret_step_1
    )

    # 3. Forward returns for Step 2 (T_2 -> T_3)
    rets_step_2 = pd.Series({"AAPL": 0.03, "MSFT": 0.06})

    # Step 2: Agent picks GOOG
    res2 = engine.step(
        selected_tickers=["GOOG"],
        equity_exposure=1.0,
        active_tilt=1.0,
        stock_simple_rets=rets_step_2,
        bm_daily_simple_ret=0.0,
    )

    # INVARIANT: AAPL (0.03) + MSFT (0.06) are active. Unfilled 1 holds benchmark (0.0).
    # Portfolio return = (0.03 + 0.06 + 0.0) / 3 = 0.03
    assert len(engine.active_sleeves) == 2
    assert engine.pending_sleeve == ["GOOG"]
    expected_ret_step_2 = (0.03 + 0.06) / H
    assert (
        pytest.approx(res2.gross_stock_daily_simple_ret, rel=1e-5)
        == expected_ret_step_2
    )


def test_forward_return_matrix_builder():
    """Verify build_forward_return_matrix computes forward returns without lookahead."""
    config = TradingConfig()
    benchmark = config.benchmark_ticker
    dates = pd.date_range("2021-01-01", periods=4, freq="B")
    df_close = pd.DataFrame(
        {
            "AAPL": [100.0, 110.0, 121.0, 121.0],
            benchmark: [200.0, 202.0, 204.02, 204.02],
        },
        index=dates,
    )

    fwd_ret = QuantUtils.build_forward_return_matrix(df_close, horizon=1)

    # AAPL Day 0 forward return (100 -> 110) is +0.10
    assert pytest.approx(fwd_ret.loc[dates[0], "AAPL"], rel=1e-5) == 0.10
    # CASH column must exist and be 0.0
    assert "CASH" in fwd_ret.columns
    assert fwd_ret.loc[dates[0], "CASH"] == 0.0


def test_discovery_env_backward_return_tripwire():
    """Verify DiscoveryEnv catches contemporaneous backward-looking returns via simple_ret_matrix."""
    config = TradingConfig()
    benchmark = config.benchmark_ticker
    dates = pd.date_range("2020-01-01", periods=20, freq="B")
    tickers = ["AAPL", "MSFT", "GOOG", "NVDA", "AMZN", "META", benchmark]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["Date", "Ticker"])

    # Create dummy cube with distinct varying values per ticker
    values = np.arange(len(idx), dtype=float)
    cube = pd.DataFrame({"41d_Log Price Gain": values}, index=idx)

    # Vectorized unstack to simulate backward returns identical to feature values
    backward_matrix = cube["41d_Log Price Gain"].unstack(level="Ticker").astype(float)
    backward_matrix["CASH"] = 0.0

    with pytest.raises(ValueError, match="CRITICAL DATA LEAKAGE DETECTED"):
        DiscoveryEnv(
            feature_cube=cube,
            simple_ret_matrix=backward_matrix,
            calendar=dates,
            config=config,
        )
