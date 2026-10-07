import os
import pandas as pd
import pytest

from data_pipeline.utils import get_chronological_splits

# Assuming your split utility is located in the appropriate path, e.g.:
# from rl_discovery.utils import get_chronological_splits


@pytest.fixture
def sample_dates():
    """Generates a base series of 150 sequential business days for testing."""
    return pd.date_range(start="2026-01-01", periods=150, freq="B")


def test_standard_splits(sample_dates, monkeypatch):
    """
    Verifies that under normal conditions with sufficient dates:
    1. Splits are strictly chronological and ordered.
    2. Split ratios approximate the default 70/15/15 distribution.
    """
    # Ensure any residual environment variables are unset for this test
    monkeypatch.delenv("DEBUG_MODE", raising=False)

    trading_calendar = sample_dates
    feature_cube_dates = sample_dates

    train, val, test = get_chronological_splits(
        trading_calendar=trading_calendar,
        feature_cube_dates=feature_cube_dates,
        holding_period=0,
        min_dates_threshold=100,
    )

    # Verify strict chronological order
    assert train.max() < val.min()
    assert val.max() < test.min()

    # Total dates = 150.
    # Train end: 150 * 0.70 = 105 -> train has indices 0 to 104 (len 105)
    # Val end: 150 * 0.85 = 127 -> val has indices 105 to 126 (len 22)
    # Test: indices 127 onwards (len 23)
    assert len(train) == 105
    assert len(val) == 22
    assert len(test) == 23


def test_debug_mode_fallback(sample_dates, monkeypatch):
    """
    Verifies that when DEBUG_MODE is active, the splitting function bypasses
    partitioning and returns identical copies of the full calendar.
    """
    monkeypatch.setenv("DEBUG_MODE", "True")

    trading_calendar = sample_dates
    feature_cube_dates = sample_dates

    train, val, test = get_chronological_splits(
        trading_calendar=trading_calendar,
        feature_cube_dates=feature_cube_dates,
        holding_period=5,
        min_dates_threshold=100,
    )

    # Confirm that each split contains all dates and is identical
    pd.testing.assert_index_equal(train, pd.Index(sample_dates))
    pd.testing.assert_index_equal(val, pd.Index(sample_dates))
    pd.testing.assert_index_equal(test, pd.Index(sample_dates))


def test_scarcity_mode_fallback(sample_dates, monkeypatch):
    """
    Verifies that when the length of valid dates is below the min_dates_threshold,
    the function automatically uses identical splits to prevent downstream IndexErrors.
    """
    monkeypatch.delenv("DEBUG_MODE", raising=False)

    # Use only 50 dates, which is below our specified threshold of 100
    short_dates = sample_dates[:50]
    trading_calendar = short_dates
    feature_cube_dates = short_dates

    train, val, test = get_chronological_splits(
        trading_calendar=trading_calendar,
        feature_cube_dates=feature_cube_dates,
        holding_period=10,
        min_dates_threshold=100,
    )

    # Confirm identical calendars are returned despite DEBUG_MODE not being explicitly True
    pd.testing.assert_index_equal(train, pd.Index(short_dates))
    pd.testing.assert_index_equal(val, pd.Index(short_dates))
    pd.testing.assert_index_equal(test, pd.Index(short_dates))


def test_purge_gap_leakage_protection(sample_dates, monkeypatch):
    """
    Verifies that when a holding period is defined, gap periods are properly introduced
    between the Train/Val and Val/Test splits to prevent data leakage.
    """
    monkeypatch.delenv("DEBUG_MODE", raising=False)

    holding_period = 10
    trading_calendar = sample_dates
    feature_cube_dates = sample_dates

    train, val, test = get_chronological_splits(
        trading_calendar=trading_calendar,
        feature_cube_dates=feature_cube_dates,
        holding_period=holding_period,
        min_dates_threshold=100,
    )

    full_index = pd.Index(sample_dates)

    # Map back to positions in the continuous calendar to measure the physical gap size
    train_last_idx = full_index.get_loc(train.max())
    val_first_idx = full_index.get_loc(val.min())

    val_last_idx = full_index.get_loc(val.max())
    test_first_idx = full_index.get_loc(test.min())

    # Enforce integer type narrowing for the static type checker
    assert isinstance(train_last_idx, int) and isinstance(val_first_idx, int)
    assert isinstance(val_last_idx, int) and isinstance(test_first_idx, int)

    # The gap (number of elements skipped between splits) should equal the configured holding period
    assert val_first_idx - train_last_idx - 1 == holding_period
    assert test_first_idx - val_last_idx - 1 == holding_period


def test_filtering_mismatched_dates(sample_dates, monkeypatch):
    """
    Verifies that the splitting function only splits dates present in both
    the trading calendar and the feature cube.
    """
    monkeypatch.delenv("DEBUG_MODE", raising=False)

    # Remove some dates from feature_cube_dates to create a mismatch
    feature_cube_dates = sample_dates[::2]  # Keep every other date

    train, val, test = get_chronological_splits(
        trading_calendar=sample_dates,
        feature_cube_dates=feature_cube_dates,
        holding_period=0,
        min_dates_threshold=50,  # Lower threshold because dates are cut in half
    )

    # Check that none of the missing dates (odd indices) exist in any split
    all_assigned_dates = pd.Index(train).union(pd.Index(val)).union(pd.Index(test))

    assert len(all_assigned_dates) == len(feature_cube_dates)
    assert all_assigned_dates.isin(feature_cube_dates).all()
