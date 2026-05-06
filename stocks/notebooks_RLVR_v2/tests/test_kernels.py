import pytest
import pandas as pd
import numpy as np

from core.kernel import QuantUtils

def test_math_integrity():
    """🛡️ Guardrail: Validates math kernels before execution."""
    
    # Test 1: Series Boundary
    mock_s = pd.Series([100.0, 102.0, 101.0])
    rets_s = QuantUtils.compute_returns(mock_s)
    assert pd.isna(rets_s.iloc[0]), "Math Integrity: Series Leading NaN missing"

    # Test 2: DataFrame Boundary
    mock_df = pd.DataFrame({"A": [100, 101], "B": [200, 202]})
    rets_df = QuantUtils.compute_returns(mock_df)
    assert rets_df.iloc[0].isna().all(), "Math Integrity: DF Leading NaN missing"

def test_ranking_integrity():
    """🛡️ Guardrail: Prevents 'Momentum Collapse' in Volatility-Adjusted Ranking."""

    # 1. Setup Mock Universe
    # VOLATILE: 10% ret / 10% Vol = 1.0 Sharpe
    # STABLE:   2% ret / 1% Vol   = 2.0 Sharpe (Winner)
    data = {"VOLATILE": [1.0, 1.10], "STABLE": [1.0, 1.02]}
    df_returns = pd.DataFrame(data).pct_change().dropna()
    vol_series = pd.Series({"VOLATILE": 0.10, "STABLE": 0.01})

    # 2. Run Kernel
    results = QuantUtils.calculate_sharpe_vol(df_returns, vol_series)

    # 3. Validation Logic
    assert not np.isclose(results["VOLATILE"], results["STABLE"]), "RANKING COLLAPSE: No differentiation."
    assert results["STABLE"] > results["VOLATILE"], "MOMENTUM REGRESSION: Volatility ignored."
    assert np.isclose(results["STABLE"], 2.0), f"MATH ERROR: Expected 2.0, got {results['STABLE']}"

def test_vol_alignment_integrity():
    """🛡️ Guardrail: Verifies Temporal Coupling between Returns and Volatility."""
    
    # 1. SETUP SYNTHETIC DATA (Day 1 is a Trap)
    # Day 1: NaN  Return, 0.90 Vol
    # Day 2: 0.10 Return, 0.10 Vol
    rets_s = pd.Series([np.nan, 0.10])
    vol_s = pd.Series([0.90, 0.10])

    # 2. RUN KERNELS
    res_series = QuantUtils.calculate_sharpe_vol(rets_s, vol_s)

    rets_df = pd.DataFrame({"A": [np.nan, 0.10], "B": [np.nan, 0.20]})
    vol_df = pd.DataFrame({"A": [0.90, 0.10], "B": [0.05, 0.20]})
    res_df = QuantUtils.calculate_sharpe_vol(rets_df, vol_df)

    # 3. VALIDATION
    assert np.isclose(res_series, 1.0), f"DENOMINATOR MISMATCH: Series {res_series:.2f} != 1.0"
    assert np.isclose(res_df["A"], 1.0) and np.isclose(res_df["B"], 1.0), "VECTORIZED MISMATCH: Column alignment failed."

def test_feature_engineering_integrity():
    """🛡️ Guardrail: Validates Feature Engineering Logic."""
    from core.features import generate_features

    # 1. Create Synthetic Data (3 Days)
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    idx = pd.MultiIndex.from_product([["TEST"], dates], names=["Ticker", "Date"])

    df_mock = pd.DataFrame(
        {
            "Adj Open": [100, 110, 110],
            "Adj High": [110, 130, 120],
            "Adj Low": [100, 110, 110],
            "Adj Close": [105, 120, 115],
            "Volume": [1000, 1000, 1000],
        },
        index=idx,
    )

    # 2. Run the Generator
    feats_df, _ = generate_features(
        df_mock, atr_period=2, rsi_period=2, quality_min_periods=1
    )
    atr_series = feats_df["ATR"]

    # 3. ASSERTIONS
    assert np.isnan(atr_series.iloc[0]), "Day 1 Regression: Expected NaN"
    assert np.isclose(atr_series.iloc[1], 25.0), f"Initialization Regression: Expected 25.0, got {atr_series.iloc[1]}"
    assert np.isclose(atr_series.iloc[2], 17.5), f"Wilder's Logic Regression: Expected 17.5, got {atr_series.iloc[2]}"
