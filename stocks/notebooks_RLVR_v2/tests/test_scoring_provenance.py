import pickle
import numpy as np
import pandas as pd
import pytest

from core.logic import SelectionLogic
from core.paths import LOCAL_DATA_DIR, OUTPUT_DIR
from core.settings import CacheConfig, TradingConfig, extract_run_hyperparameters

# =====================================================================
# PATHS & CONFIGURATION
# =====================================================================

pkl_files = list(OUTPUT_DIR.glob("results_*.pkl"))
if not pkl_files:
    pkl_files = list(OUTPUT_DIR.glob("*.pkl"))

latest_pkl = max(pkl_files, key=lambda p: p.stat().st_mtime) if pkl_files else None
PKL_PATH = latest_pkl
PKL_FILENAME = PKL_PATH.name if PKL_PATH else None
PARQUET_PATH = LOCAL_DATA_DIR / CacheConfig.get_filename()

FILES_EXIST = PKL_PATH is not None and PKL_PATH.exists() and PARQUET_PATH.exists()

TARGET_DATES = ["2022-04-07", "2026-07-09"]

# =====================================================================
# FIXTURES
# =====================================================================


@pytest.fixture(scope="module")
def system_artifacts():
    """Loads the real PKL blotter and corresponding Parquet slices."""
    if not FILES_EXIST:
        pytest.skip(f"Missing real data files. Need {PKL_FILENAME} and cache parquet.")

    assert PKL_PATH is not None, "PKL_PATH cannot be None when loading artifacts."
    with open(PKL_PATH, "rb") as f:
        results = pickle.load(f)

    meta = results.get("metadata", {})
    blotter_df = pd.DataFrame(results["blotter"])
    blotter_df["decision_date"] = pd.to_datetime(blotter_df["decision_date"])

    params = extract_run_hyperparameters(PKL_PATH, metadata=meta, blotter_df=blotter_df)

    target_timestamps = pd.to_datetime(TARGET_DATES)
    test_blotter = blotter_df[
        blotter_df["decision_date"].isin(target_timestamps)
    ].copy()
    test_blotter.attrs["metadata"] = meta
    test_blotter.attrs["extracted_params"] = params

    cube = pd.read_parquet(PARQUET_PATH)
    test_cube = cube[cube.index.get_level_values("Date").isin(target_timestamps)]

    return test_blotter, test_cube


@pytest.fixture(scope="module")
def trading_config():
    """Loads the core configuration updated with hyperparameters extracted from the artifact."""
    if PKL_PATH is not None and PKL_PATH.exists():
        return TradingConfig.from_artifact(PKL_PATH)
    return TradingConfig()


# =====================================================================
# TIER 1: UNIVERSE ALIGNMENT & BARE-METAL MATH CHECK
# =====================================================================


@pytest.mark.skipif(not FILES_EXIST, reason="Real system artifacts not found.")
def test_bare_metal_provenance(system_artifacts):
    """
    Calculates scores independently using bare-metal numpy/pandas matrix operations
    to verify recorded blotter outputs.
    """
    blotter_df, feature_cube = system_artifacts

    for _, row in blotter_df.iterrows():
        decision_date = row["decision_date"]

        ensemble = feature_cube.xs(decision_date, level="Date")
        raw_actions = np.array(row["raw_actions"])

        clean_matrix = ensemble.fillna(0.0)
        num_features = clean_matrix.shape[1]
        clipped_action = np.clip(np.nan_to_num(raw_actions, nan=0.0), -1.0, 1.0)
        weights = clipped_action[:num_features]

        # 1. UNIVERSE ALIGNMENT CHECK
        assert (
            len(ensemble) == row["universe_size"]
        ), f"Universe size mismatch on {decision_date}. Blotter: {row['universe_size']}, Cube: {len(ensemble)}"

        # 2. BARE-METAL MATH RECALCULATION
        norm = np.linalg.norm(weights)
        norm_weights = weights / norm if norm > 1e-6 else weights

        vals = clean_matrix.values.astype(np.float64)
        # Auto-detect whether blotter artifact was produced under Phase 1+ Factor Equalization
        # or legacy unstandardized scoring
        mu = np.mean(vals, axis=0, keepdims=True)
        sigma = np.maximum(np.std(vals, axis=0, keepdims=True), 1e-8)
        z_vals = np.clip((vals - mu) / sigma, -4.0, 4.0)

        scores_z = pd.Series(z_vals @ norm_weights, index=clean_matrix.index)
        scores_raw = pd.Series(vals @ norm_weights, index=clean_matrix.index)

        # Reconcile against recorded max_score
        if np.isclose(row["max_score"], float(scores_z.max()), atol=1e-4):
            sorted_tickers = scores_z.sort_values(ascending=False)
        else:
            sorted_tickers = scores_raw.sort_values(ascending=False)

        expected_top_3 = sorted_tickers.index[:3].tolist()
        expected_max = float(sorted_tickers.max())
        expected_min = float(sorted_tickers.min())

        # 3. ASSERTIONS AGAINST RECORDED SYSTEM TRUTH
        assert (
            row["top_3"] == expected_top_3
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
    blotter_df, feature_cube = system_artifacts

    rows_info = []
    for _, row in blotter_df.iterrows():
        raw_a = np.array(row["raw_actions"])
        rows_info.append(
            f"Date: {row['decision_date']} | Width: {row['width']} | Offset: {row['offset']} | "
            f"Actions[-4:]: {raw_a[-4:].tolist()}"
        )

    meta = blotter_df.attrs.get("metadata", {})

    for _, row in blotter_df.iterrows():
        decision_date = row["decision_date"]
        ensemble = feature_cube.xs(decision_date, level="Date")
        raw_actions = np.array(row["raw_actions"])

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
            ensemble=ensemble,
            action=raw_actions,
            rank_max_offset_percentile=trading_config.rank_max_offset_percentile,
            rank_max_width=trading_config.rank_max_width,
            min_basket_width=trading_config.min_basket_width,
            max_cash_pct=trading_config.max_cash_pct,
            min_active_tilt=trading_config.min_active_tilt,
        )

        assert (
            offset == row["offset"]
        ), f"Offset drift on {decision_date}: expected {row['offset']}, got {offset}"

        assert width == row["width"], (
            f"\n--- [TRAP] SCORING REPLAY DIAGNOSTICS ---\n"
            f"PKL File: {PKL_FILENAME}\n"
            f"Metadata in PKL: {meta}\n"
            f"All Target Rows:\n  " + "\n  ".join(rows_info) + "\n"
            f"Failed on: {decision_date}\n"
            f"  Calculated width: {width} (min_basket_width={trading_config.min_basket_width}, rank_max_width={trading_config.rank_max_width})\n"
            f"  Recorded width in blotter: {row['width']}\n"
            f"  raw_action[-3] (width action): {raw_actions[-3]}\n"
        )
