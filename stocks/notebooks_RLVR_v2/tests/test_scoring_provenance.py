import pytest
import pickle
import numpy as np
import pandas as pd
import re

from core.paths import OUTPUT_DIR, LOCAL_DATA_DIR
from core.settings import CacheConfig, TradingConfig
from core.logic import SelectionLogic

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


def _extract_run_hyperparameters(filename: str, metadata: dict | None = None) -> dict:
    metadata = metadata or {}
    params: dict = {}

    # 1. min_basket_width: from metadata or tokens like _Width5_, _w5_
    w_match = re.search(r"_(?:Width|width|w)([0-9]+)(?:_|$)", filename)
    if "min_basket_width" in metadata:
        params["min_basket_width"] = int(metadata["min_basket_width"])
    elif w_match:
        params["min_basket_width"] = int(w_match.group(1))

    # 2. min_active_tilt: from metadata or tokens like _tilt0.85_, _Tilt85_
    tilt_match = re.search(r"_(?:tilt|Tilt)([0-9.]+)(?:_|$)", filename)
    if "min_active_tilt" in metadata:
        params["min_active_tilt"] = float(metadata["min_active_tilt"])
    elif tilt_match:
        val = float(tilt_match.group(1))
        params["min_active_tilt"] = val / 100.0 if val > 1.0 else val

    # 3. upside_alpha_mult: from metadata or tokens like _mult2.5_, _upside_2.5_, _up_2.5_
    mult_match = re.search(r"_(?:mult|upside|up)_?([0-9.]+)(?:_|$)", filename)
    if "upside_alpha_mult" in metadata:
        params["upside_alpha_mult"] = float(metadata["upside_alpha_mult"])
    elif mult_match:
        params["upside_alpha_mult"] = float(mult_match.group(1))

    # 4. loss_aversion_penalty: from metadata or tokens like _lossav_1.0_, _pen_2.0_, _loss_0.5_
    lossav_match = re.search(r"_(?:lossav|loss)_?([0-9.]+)(?:_|$)", filename)
    pen_match = re.search(r"_pen_?([0-9.]+)(?:_|$)", filename)
    if "loss_aversion_penalty" in metadata:
        params["loss_aversion_penalty"] = float(metadata["loss_aversion_penalty"])
    elif lossav_match:
        params["loss_aversion_penalty"] = float(lossav_match.group(1))
    elif pen_match:
        params["loss_aversion_penalty"] = max(0.0, float(pen_match.group(1)) - 1.0)

    # 5. holding_period: from metadata or token like _T3_
    t_match = re.search(r"_T([0-9]+)(?:_|$)", filename)
    if "holding_period" in metadata:
        params["holding_period"] = int(metadata["holding_period"])
    elif t_match:
        params["holding_period"] = int(t_match.group(1))

    return params


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
    params = _extract_run_hyperparameters(PKL_PATH.name, meta)

    blotter_df = pd.DataFrame(results["blotter"])
    blotter_df["decision_date"] = pd.to_datetime(blotter_df["decision_date"])

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
    config = TradingConfig()
    meta = {}
    if PKL_PATH is not None and PKL_PATH.exists():
        try:
            with open(PKL_PATH, "rb") as f:
                results = pickle.load(f)
            meta = results.get("metadata", {})
        except Exception:
            pass
        params = _extract_run_hyperparameters(PKL_PATH.name, meta)
        for k, v in params.items():
            if hasattr(config, k):
                setattr(config, k, v)
    return config


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

        scores = clean_matrix.values @ norm_weights
        score_series = pd.Series(scores, index=clean_matrix.index)
        sorted_tickers = score_series.sort_values(ascending=False)

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
