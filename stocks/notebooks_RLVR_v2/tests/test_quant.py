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


#
