import pandas as pd
import numpy as np
import pytest

from core.quant import QuantUtils


def test_compute_returns_boundary_integrity(debug: bool = False):
    """Validates math kernels before execution (Leading NaNs)."""
    # Test 1: Series Boundary
    mock_s = pd.Series([100.0, 102.0, 101.0])
    rets_s = QuantUtils.compute_returns(mock_s)
    assert pd.isna(rets_s.iloc[0]), "Math Integrity: Series Leading NaN missing"

    # Test 2: DataFrame Boundary
    mock_df = pd.DataFrame({"A": [100, 101], "B": [200, 202]})
    rets_df = QuantUtils.compute_returns(mock_df)

    if debug:
        print(f"mock_s:\n{mock_s}\n")
        print(f"rets_s:\n{rets_s}\n")
        print(f"mock_df:\n{mock_df}\n")
        print(f"rets_df:\n{rets_df}\n")

    assert rets_df.iloc[0].isna().all(), "Math Integrity: DF Leading NaN missing"


def test_ranking_integrity_sharpe_vol(debug: bool = False):
    """
    Prevents 'Momentum Collapse' in Volatility-Adjusted Ranking.
    Ensures Sharpe(Vol) distinguishes between High-Vol and Low-Vol stocks.
    """
    # VOLATILE: 10% ret / 10% Vol = 1.0 Sharpe
    # STABLE:   2% ret / 1% Vol   = 2.0 Sharpe (Winner)
    data = {"VOLATILE": [1.0, 1.10], "STABLE": [1.0, 1.02]}
    df_returns = pd.DataFrame(data).pct_change().dropna()
    vol_series = pd.Series({"VOLATILE": 0.10, "STABLE": 0.01})

    results = QuantUtils.calc_sharpe_cross_section(df_returns, vol_series)

    if debug:
        print(f"data {type(data)}:\n{data}\n")
        print(f"df_returns {type(df_returns)}:\n{df_returns}\n")
        print(f"vol_series {type(vol_series)}:\n{vol_series}\n")
        print(f"results:\n{results}\n")

    assert not np.isclose(
        results["VOLATILE"], results["STABLE"]
    ), "RANKING COLLAPSE: No differentiation"
    assert (
        results["STABLE"] > results["VOLATILE"]
    ), "MOMENTUM REGRESSION: Volatility ignored"
    assert np.isclose(
        results["STABLE"], 2.0
    ), f"MATH ERROR: Expected 2.0, got {results['STABLE']}"


def test_volatility_alignment_temporal_coupling(debug: bool = False):
    """
    Verifies Temporal Coupling between Returns and Volatility.
    Ensures denominator only counts days where a valid return exists.
    """
    # Day 1: NaN Return, 0.90 Vol
    # Day 2: 0.10 Return, 0.10 Vol
    rets_s = pd.Series([np.nan, 0.10])
    vol_s = pd.Series([0.90, 0.10])
    res_series = QuantUtils.calc_sharpe_univariate(rets_s, vol_s)

    assert np.isclose(
        res_series, 1.0
    ), f"DENOMINATOR MISMATCH: Series {res_series:.2f} != 1.0"
    rets_df = pd.DataFrame({"A": [np.nan, 0.10], "B": [np.nan, 0.20]})
    vol_df = pd.DataFrame({"A": [0.90, 0.10], "B": [0.05, 0.20]})
    res_df = QuantUtils.calc_sharpe_multivariate_aligned(rets_df, vol_df)

    if debug:
        print(f"rets_s:\n{rets_s}\n")
        print(f"vol_s:\n{vol_s}\n")
        print(f"res_series:\n{res_series}\n")
        print(f"rets_df:\n{rets_df}\n")
        print(f"vol_df:\n{vol_df}\n")
        print(f"res_df:\n{res_df}\n")

    assert np.isclose(res_df["A"], 1.0) and np.isclose(
        res_df["B"], 1.0
    ), "VECTORIZED MISMATCH: Column alignment failed"


def test_sharpe_alignment():
    """
    Validates that QuantUtils kernels enforce index/column alignment
    and handle mathematical coupling correctly.

    ### What this test enforces:
    1.  **Univariate:** Ensures that if the series have different date indices,
        the function refuses to guess and crashes instead (preventing "look-ahead" or
        "date-shifted" errors).
    2.  **Cross-Section:** Ensures that if `returns` has `[AAPL, GOOGL]` and the
        `vol_vector` has `[GOOGL, AAPL]`, the code raises an error before performing
        the `.to_numpy()` calculation.
    3.  **Multivariate (Coupling):** This is the most important one. It verifies that
        if an asset has a missing return on a specific day, the volatility for that
        specific day is **not** included in the average volatility. This prevents
        "Volatility Dilution" where an asset looks safer than it is because it didn't
        trade during a high-vol period.
    """

    # --- SETUP MOCK DATA ---
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    tickers = ["AAPL", "GOOGL"]

    # Returns: AAPL (0.01, 0.02, 0.03) | GOOGL (0.1, 0.2, 0.3)
    df_returns = pd.DataFrame(
        {"AAPL": [0.01, 0.02, 0.03], "GOOGL": [0.1, 0.2, 0.3]}, index=dates
    )

    # Vol Vector for Cross Section
    ser_vol = pd.Series({"AAPL": 0.01, "GOOGL": 0.1}, name="Vol")

    # Vol Grid for Multivariate
    df_vol_grid = pd.DataFrame(
        {"AAPL": [0.01, 0.01, 0.01], "GOOGL": [0.1, 0.1, 0.1]}, index=dates
    )

    # 1. TEST: calc_sharpe_univariate (Temporal Alignment)
    # Correct Alignment
    res_uni = QuantUtils.calc_sharpe_univariate(df_returns["AAPL"], df_vol_grid["AAPL"])
    assert np.isclose(res_uni, 2.0), f"Univariate Math: Expected 2.0, got {res_uni}"

    # Mismatched Dates -> Should Raise Error
    ser_bad_dates = df_vol_grid["AAPL"].copy()
    ser_bad_dates.index = pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"])
    try:
        QuantUtils.calc_sharpe_univariate(df_returns["AAPL"], ser_bad_dates)
        pytest.fail("Univariate failed to catch date mismatch!")
    except (ValueError, AssertionError):
        pass  # Success: Error caught

    # 2. TEST: calc_sharpe_cross_section (Ticker Alignment)
    # Correct Alignment
    res_cs = QuantUtils.calc_sharpe_cross_section(df_returns, ser_vol)
    assert np.isclose(res_cs["AAPL"], 2.0)
    assert np.isclose(res_cs["GOOGL"], 2.0)

    # Mismatched Tickers (Swapped) -> Should Raise Error
    ser_swapped = pd.Series({"GOOGL": 0.1, "AAPL": 0.01})
    # Note: Even if values match, if the index order isn't identical, .to_numpy() will swap them
    try:
        # This will fail the 'assert returns.columns.equals(vol_vector.index)'
        QuantUtils.calc_sharpe_cross_section(df_returns, ser_swapped)
        pytest.fail("Cross-section failed to catch ticker order mismatch!")
    except (ValueError, AssertionError):
        pass  # Success: Error caught

    # 3. TEST: calc_sharpe_multivariate_aligned (Temporal Coupling)
    # We introduce a NaN in AAPL returns on Day 2.
    # AAPL returns: [0.01, NaN, 0.03] -> Mean = 0.02
    # AAPL vol:    [0.01, 99.0, 0.01]
    # If coupling works, the 99.0 is ignored. Result: 0.02 / 0.01 = 2.0
    # If coupling fails, the 99.0 is included. Result: 0.02 / 33.0 = ~0.0006

    df_ret_nan = df_returns.copy()
    df_ret_nan.iloc[1, 0] = np.nan  # NaN for AAPL on Day 2

    df_vol_extreme = df_vol_grid.copy()
    df_vol_extreme.iloc[1, 0] = 99.0  # Extreme vol on the day AAPL didn't trade

    res_multi = QuantUtils.calc_sharpe_multivariate_aligned(df_ret_nan, df_vol_extreme)

    assert np.isclose(
        res_multi["AAPL"], 2.0
    ), f"Temporal Coupling Failed: Expected 2.0, got {res_multi['AAPL']}. (Vol not masked)"

    # 4. TEST: Column Mismatch in Multivariate
    df_vol_bad_cols = df_vol_grid.rename(columns={"AAPL": "MSFT"})
    try:
        QuantUtils.calc_sharpe_multivariate_aligned(df_returns, df_vol_bad_cols)
        pytest.fail("Multivariate failed to catch column name mismatch!")
    except (ValueError, AssertionError):
        pass  # Success

    print("✅ All QuantUtils Alignment and Math tests passed!")


# drop in at the bottom of tests/test_quant.py


def test_calculate_information_ratio():
    # 1. Normal active alpha scenario
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    bench = pd.Series(
        [0.01, -0.01, 0.02, 0.00, 0.01, -0.02, 0.01, 0.00, -0.01, 0.02], index=dates
    )
    # Portfolio consistently beats benchmark by 10 bps with some active volatility
    port = bench + pd.Series(
        [0.001, 0.002, 0.001, 0.003, -0.001, 0.002, 0.001, 0.002, 0.000, 0.001],
        index=dates,
    )

    ir = QuantUtils.calculate_information_ratio(port, bench, periods=252)
    assert np.isfinite(ir)
    assert ir > 0.0

    # 2. Identical returns (Zero tracking error / zero active return)
    ir_zero = QuantUtils.calculate_information_ratio(bench, bench, periods=252)
    assert ir_zero == 0.0

    # 3. Degenerate edge cases (< 2 observations, all NaNs)
    short_s = pd.Series([0.01], index=pd.date_range("2020-01-01", periods=1))
    assert QuantUtils.calculate_information_ratio(short_s, short_s) == 0.0

    nan_s = pd.Series([np.nan, np.nan], index=pd.date_range("2020-01-01", periods=2))
    assert QuantUtils.calculate_information_ratio(nan_s, bench.iloc[:2]) == 0.0


def test_compute_composite_fitness():
    # 1. Positive Alpha, Positive IR
    fitness_pos = QuantUtils.compute_composite_fitness(
        excess_return=0.05, information_ratio=1.2
    )
    assert fitness_pos == pytest.approx(0.05 * 1.2)
    assert fitness_pos > 0.0

    # 2. Positive Alpha, Micro IR (Clamped to ir_floor)
    fitness_clamped = QuantUtils.compute_composite_fitness(
        excess_return=0.04, information_ratio=0.01, ir_floor=0.05
    )
    assert fitness_clamped == pytest.approx(0.04 * 0.05)

    # 3. Negative Alpha (Steep penalty applied)
    fitness_neg = QuantUtils.compute_composite_fitness(
        excess_return=-0.03, information_ratio=-0.8, ir_floor=0.05
    )
    assert fitness_neg < 0.0
    assert fitness_neg == pytest.approx(-0.03 * (1.0 / 0.05))

    # 4. Non-finite values safe trap
    assert QuantUtils.compute_composite_fitness(np.nan, 1.0) == -10.0
    assert QuantUtils.compute_composite_fitness(0.05, np.inf) == -10.0


def test_calculate_residual_momentum():
    dates = pd.date_range("2020-01-01", periods=150, freq="D")
    bm_rets = pd.Series(np.random.normal(0.0005, 0.01, size=150), index=dates)

    # Stock A: Beta = 1.0, Constant Positive Alpha (+10 bps/day)
    stock_a = bm_rets + 0.001
    # Stock B: Pure Beta = 1.0, Zero Alpha
    stock_b = bm_rets.copy()

    df_rets = pd.DataFrame({"Alpha_Stock": stock_a, "Beta_Stock": stock_b}, index=dates)
    res_mom = QuantUtils.calculate_residual_momentum(df_rets, bm_rets, window=126)

    assert isinstance(res_mom, pd.DataFrame)
    assert (
        res_mom["Alpha_Stock"].iloc[-1] > 0.0
    ), "Alpha stock must have positive residual momentum"
    assert np.isclose(
        res_mom["Beta_Stock"].iloc[-1], 0.0, atol=1e-3
    ), "Pure beta stock residual momentum should be near 0"


def test_calculate_range_pos_52w():
    dates = pd.date_range("2020-01-01", periods=260, freq="D")
    # Monotonically increasing prices -> always at 52w high
    p_high = pd.Series(np.linspace(100, 200, 260), index=dates)
    pos_high = QuantUtils.calculate_range_pos_52w(p_high, window=252)
    assert np.isclose(pos_high.iloc[-1], 1.0)

    # Stock crashed 20% below peak
    p_dip = p_high.copy()
    p_dip.iloc[-1] = 160.0  # 160 / 200 = 0.80
    pos_dip = QuantUtils.calculate_range_pos_52w(p_dip, window=252)
    assert np.isclose(pos_dip.iloc[-1], 0.80)
    assert (pos_dip >= 0.0).all() and (pos_dip <= 1.0).all()


def test_calculate_downside_beta():
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    bm_rets = pd.Series(np.where(np.arange(100) % 2 == 0, -0.02, 0.02), index=dates)

    # Stock doubles market down-moves: -0.04 on down days, +0.02 on up days
    stock_down = pd.Series(np.where(np.arange(100) % 2 == 0, -0.04, 0.02), index=dates)

    d_beta = QuantUtils.calculate_downside_beta(stock_down, bm_rets, window=63)
    assert np.isclose(
        d_beta.iloc[-1], 2.0, atol=1e-2
    ), f"Expected downside beta ~2.0, got {d_beta.iloc[-1]}"


def test_calculate_efficiency_ratio():
    dates = pd.date_range("2020-01-01", periods=80, freq="D")

    # 1. Smooth straight line -> ER = 1.0
    p_smooth = pd.Series(np.linspace(100, 180, 80), index=dates)
    er_smooth = QuantUtils.calculate_efficiency_ratio(p_smooth, window=63)
    assert np.isclose(er_smooth.iloc[-1], 1.0)

    # 2. Ping-pong price (100 -> 101 -> 100 -> 101) -> Net change ~0, Total path high -> ER ~ 0.0
    p_choppy = pd.Series(
        [100.0 if i % 2 == 0 else 101.0 for i in range(80)], index=dates
    )
    er_choppy = QuantUtils.calculate_efficiency_ratio(p_choppy, window=63)
    assert er_choppy.iloc[-1] < 0.05


#
