import pandas as pd
import numpy as np

from core.settings import TradingConfig
from data_pipeline.builder import generate_features


def test_feature_engineering_wilders_atr():
    """
    Validates Feature Engineering Logic.
    Enforces: Day 1 ATR must be NaN, Initialization, and Wilder's Smoothing.
    """
    # 1. Create Synthetic Data (3 Days)
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    idx = pd.MultiIndex.from_product([["TEST"], dates], names=["Ticker", "Date"])

    df_mock = pd.DataFrame(
        {
            "Adj Open": [100, 110, 110],
            "Adj High": [110, 130, 120],
            "Adj Low": [100, 110, 110],
            "Adj Close": [105, 120, 115],  # PrevClose: NaN, 105, 120
            "Volume": [1000, 1000, 1000],
        },
        index=idx,
    )

    # 2. Run the Generator (Period=2 means Alpha = 1/2 = 0.5)
    test_config = TradingConfig()
    test_config.atr_period = 2
    test_config.rsi_period = 2
    test_config.quality_min_periods = 1

    feats_df, _ = generate_features(df_mock, config=test_config)

    atr_series = feats_df["ATR"]

    # 3. Assertions
    # Check Day 1 (No PrevClose)
    assert np.isnan(
        atr_series.iloc[0]
    ), f"Day 1 Regression: Expected NaN, got {atr_series.iloc[0]}"

    # Check Day 2 (Initialization: H-L=20, |130-105|=25, |110-105|=5 -> TR=25)
    assert np.isclose(
        atr_series.iloc[1], 25.0
    ), f"Initialization Regression: Expected 25.0, got {atr_series.iloc[1]}"

    # Check Day 3 (Recursion: TR=10. ATR_3 = (10 * 0.5) + (25 * 0.5) = 17.5)
    assert np.isclose(
        atr_series.iloc[2], 17.5
    ), f"Wilder's Logic Regression: Expected 17.5, got {atr_series.iloc[2]}"
