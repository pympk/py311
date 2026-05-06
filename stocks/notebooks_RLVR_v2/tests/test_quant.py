import pytest
import pandas as pd
import numpy as np

# Adjust imports based on your structure
from core.quant import QuantUtils


def test_compute_returns_boundary_integrity():
    """Validates math kernels before execution (Leading NaNs)."""
    # Test 1: Series Boundary
    mock_s = pd.Series([100.0, 102.0, 101.0])
    rets_s = QuantUtils.compute_returns(mock_s)
    assert pd.isna(rets_s.iloc[0]), "Math Integrity: Series Leading NaN missing"

    # Test 2: DataFrame Boundary
    mock_df = pd.DataFrame({"A": [100, 101], "B": [200, 202]})
    rets_df = QuantUtils.compute_returns(mock_df)
    assert rets_df.iloc[0].isna().all(), "Math Integrity: DF Leading NaN missing"


def test_ranking_integrity_sharpe_vol():
    """
    Prevents 'Momentum Collapse' in Volatility-Adjusted Ranking.
    Ensures Sharpe(Vol) distinguishes between High-Vol and Low-Vol stocks.
    """
    # VOLATILE: 10% ret / 10% Vol = 1.0 Sharpe
    # STABLE:   2% ret / 1% Vol   = 2.0 Sharpe (Winner)
    data = {"VOLATILE": [1.0, 1.10], "STABLE": [1.0, 1.02]}
    df_returns = pd.DataFrame(data).pct_change().dropna()
    vol_series = pd.Series({"VOLATILE": 0.10, "STABLE": 0.01})

    results = QuantUtils.calculate_sharpe_vol(df_returns, vol_series)

    assert not np.isclose(
        results["VOLATILE"], results["STABLE"]
    ), "RANKING COLLAPSE: No differentiation"
    assert (
        results["STABLE"] > results["VOLATILE"]
    ), "MOMENTUM REGRESSION: Volatility ignored"
    assert np.isclose(
        results["STABLE"], 2.0
    ), f"MATH ERROR: Expected 2.0, got {results['STABLE']}"


def test_volatility_alignment_temporal_coupling():
    """
    Verifies Temporal Coupling between Returns and Volatility.
    Ensures denominator only counts days where a valid return exists.
    """
    # Day 1: NaN Return, 0.90 Vol
    # Day 2: 0.10 Return, 0.10 Vol
    rets_s = pd.Series([np.nan, 0.10])
    vol_s = pd.Series([0.90, 0.10])

    res_series = QuantUtils.calculate_sharpe_vol(rets_s, vol_s)
    assert np.isclose(
        res_series, 1.0
    ), f"DENOMINATOR MISMATCH: Series {res_series:.2f} != 1.0"

    rets_df = pd.DataFrame({"A": [np.nan, 0.10], "B": [np.nan, 0.20]})
    vol_df = pd.DataFrame({"A": [0.90, 0.10], "B": [0.05, 0.20]})

    res_df = QuantUtils.calculate_sharpe_vol(rets_df, vol_df)
    assert np.isclose(res_df["A"], 1.0) and np.isclose(
        res_df["B"], 1.0
    ), "VECTORIZED MISMATCH: Column alignment failed"
