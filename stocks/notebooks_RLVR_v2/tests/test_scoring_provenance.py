import pytest
import pickle
import numpy as np
import pandas as pd

from core.paths import OUTPUT_DIR, LOCAL_DATA_DIR
from core.settings import CacheConfig, TradingConfig
from core.logic import SelectionLogic

# =====================================================================
# PATHS & CONFIGURATION
# =====================================================================

PKL_FILENAME = "oos_results_ent_0.01_pen_1.0_lr_0.0003.pkl"
PKL_PATH = OUTPUT_DIR / PKL_FILENAME
PARQUET_PATH = LOCAL_DATA_DIR / CacheConfig.get_filename()

FILES_EXIST = PKL_PATH.exists() and PARQUET_PATH.exists()

# Define the target dates discovered in Notebook 03 to keep the test fast
TARGET_DATES = ["2022-04-07", "2026-07-09"]


# =====================================================================
# FIXTURES
# =====================================================================


@pytest.fixture(scope="module")
def trading_config():
    return TradingConfig()


@pytest.fixture(scope="module")
def system_artifacts():
    """Loads the real PKL blotter and corresponding Parquet slices."""
    if not FILES_EXIST:
        pytest.skip(f"Missing real data files. Need {PKL_FILENAME} and cache parquet.")

    # 1. Load Blotter
    with open(PKL_PATH, "rb") as f:
        results = pickle.load(f)

    blotter_df = pd.DataFrame(results["blotter"])
    blotter_df["decision_date"] = pd.to_datetime(blotter_df["decision_date"])

    # Filter blotter to only the target dates to speed up the test
    target_timestamps = pd.to_datetime(TARGET_DATES)
    test_blotter = blotter_df[blotter_df["decision_date"].isin(target_timestamps)]

    # 2. Load Feature Cube (Only loading necessary dates to save RAM/Time)
    cube = pd.read_parquet(PARQUET_PATH)

    # Extract only the dates we need from the MultiIndex
    test_cube = cube[cube.index.get_level_values("Date").isin(target_timestamps)]

    return test_blotter, test_cube


# =====================================================================
# TIER 1: UNIVERSE ALIGNMENT & BARE-METAL MATH CHECK
# =====================================================================


@pytest.mark.skipif(not FILES_EXIST, reason="Real system artifacts not found.")
def test_bare_metal_provenance(system_artifacts):
    """
    Strips away all system architecture. Calculates scores entirely independently
    using raw numpy/pandas math to verify the recorded blotter outputs.
    """
    blotter_df, feature_cube = system_artifacts

    for _, row in blotter_df.iterrows():
        decision_date = row["decision_date"]

        # Extract the state at that exact moment
        ensemble = feature_cube.xs(decision_date, level="Date")
        raw_actions = np.array(row["raw_actions"])
        weights = raw_actions[:-2]

        print(f"decision_date:\n{decision_date}\n")
        print(f"ensemble:\n{ensemble}\n")
        print(f"raw_actions:\n{raw_actions}\n")
        print(f"weights:\n{weights}\n")

        # 1. UNIVERSE ALIGNMENT CHECK
        # Did the environment see the exact same number of tickers recorded in the blotter?
        assert (
            len(ensemble) == row["universe_size"]
        ), f"Universe size mismatch on {decision_date}. Blotter: {row['universe_size']}, Cube: {len(ensemble)}"

        # 2. BARE-METAL MATH RECALCULATION
        clean_matrix = ensemble.fillna(0.0)

        # Raw matrix multiplication (dot product)
        scores = clean_matrix.values @ weights
        score_series = pd.Series(scores, index=clean_matrix.index)

        # Descending sort
        sorted_tickers = score_series.sort_values(ascending=False)

        print(f"scores shape: {scores.shape}")
        print(f"scores:\n{scores}\n")
        print(f"score_series:\n{score_series}\n")
        print(f"sorted_tickers:\n{sorted_tickers}\n")

        expected_top_3 = sorted_tickers.index[:3].tolist()
        expected_max = float(sorted_tickers.max())
        expected_min = float(sorted_tickers.min())

        # 3. ASSERTIONS AGAINST RECORDED SYSTEM TRUTH
        assert (
            row["top_3_tickers"] == expected_top_3
        ), f"Top 3 math deviation on {decision_date.date()}"

        assert np.isclose(
            row["max_score"], expected_max, atol=1e-5
        ), f"Max score drift on {decision_date.date()}"

        assert np.isclose(
            row["min_score"], expected_min, atol=1e-5
        ), f"Min score drift on {decision_date.date()}"


# =====================================================================
# TIER 2: SELECTION LOGIC SYSTEM REPLAY
# =====================================================================


@pytest.mark.skipif(not FILES_EXIST, reason="Real system artifacts not found.")
def test_system_logic_replay(system_artifacts, trading_config):
    """
    Passes the exact historical inputs back into `SelectionLogic.apply_action`
    and verifies the logic layer outputs perfectly match the blotter.
    """
    blotter_df, feature_cube = system_artifacts

    for _, row in blotter_df.iterrows():
        decision_date = row["decision_date"]
        ensemble = feature_cube.xs(decision_date, level="Date")
        raw_actions = np.array(row["raw_actions"])

        # Execute System Logic
        selected, top_3, offset, width, max_s, min_s = SelectionLogic.apply_action(
            ensemble=ensemble,
            action=raw_actions,
            rank_max_offset=trading_config.rank_max_offset,
            rank_max_width=trading_config.rank_max_width,
        )

        # Verify decoding logic and slicing
        assert (
            offset == row["decoded_offset"]
        ), f"Offset logic failed on {decision_date}"
        assert width == row["decoded_width"], f"Width logic failed on {decision_date}"

        # Verify lists
        assert top_3 == row["top_3_tickers"], f"Top 3 list failed on {decision_date}"
        assert (
            selected == row["chosen_tickers"]
        ), f"Selected list failed on {decision_date}"
