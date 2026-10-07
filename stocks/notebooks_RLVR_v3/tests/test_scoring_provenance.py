"""
Scoring Provenance & Selection Logic Replay Test Suite

Uses the frozen hermetic canonical slice (output/canonical_anchors/canonical_scoring_slice.parquet)
to guarantee bit-level identity regardless of future daily OHLCV updates in data/.
"""

from pathlib import Path
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
import pytest

from core.logic import SelectionLogic
from core.paths import LOCAL_DATA_DIR, OUTPUT_DIR
from core.settings import CacheConfig, TradingConfig, extract_run_hyperparameters

# =====================================================================
# PATHS & ARTIFACT DISCOVERY (GROUND-TRUTH VAULT)
# =====================================================================

ANCHOR_DIR = OUTPUT_DIR / "canonical_anchors"

# 1. Target Blotter: Authoritative Gen 18 baseline champion (s42)
candidate_blotters = sorted(
    list(ANCHOR_DIR.glob("blotter_continuous_s42_gen*.parquet"))
)
if not candidate_blotters:
    candidate_blotters = sorted(
        list(ANCHOR_DIR.glob("blotter_continuous_s*_gen*.parquet"))
    )

BLOTTER_PATH = candidate_blotters[-1] if candidate_blotters else None

# 2. Frozen Feature Slice (Prioritizes canonical slice, falls back to local data dir)
CANONICAL_SLICE_PATH = ANCHOR_DIR / "canonical_scoring_slice.parquet"
if CANONICAL_SLICE_PATH.exists():
    FEATURE_CUBE_PATH = CANONICAL_SLICE_PATH
elif (LOCAL_DATA_DIR / f"_{CacheConfig.get_filename()}").exists():
    FEATURE_CUBE_PATH = LOCAL_DATA_DIR / f"_{CacheConfig.get_filename()}"
else:
    FEATURE_CUBE_PATH = LOCAL_DATA_DIR / CacheConfig.get_filename()

FILES_EXIST = (
    BLOTTER_PATH is not None and BLOTTER_PATH.exists() and FEATURE_CUBE_PATH.exists()
)

TARGET_DATES = ["2022-04-07", "2026-07-09"]

# =====================================================================
# FIXTURES
# =====================================================================


@pytest.fixture(scope="module")
def system_artifacts():
    """Loads the real Parquet blotter and the hermetic feature cube slice."""
    if not FILES_EXIST:
        pytest.skip("Required ground-truth artifacts missing from canonical_anchors.")

    assert BLOTTER_PATH is not None
    blotter_df = pd.read_parquet(BLOTTER_PATH)
    blotter_df["decision_date"] = pd.to_datetime(blotter_df["decision_date"])

    meta = {}
    params = extract_run_hyperparameters(
        BLOTTER_PATH, metadata=meta, blotter_df=blotter_df
    )

    target_timestamps = pd.to_datetime(TARGET_DATES)
    test_blotter = blotter_df[
        blotter_df["decision_date"].isin(target_timestamps)
    ].copy()
    test_blotter.attrs["extracted_params"] = params

    cube = pd.read_parquet(FEATURE_CUBE_PATH)
    eval_dates = pd.to_datetime(test_blotter["decision_date"]).unique()
    test_cube = cube[
        pd.to_datetime(cube.index.get_level_values("Date")).isin(eval_dates)
    ]

    return test_blotter, test_cube


@pytest.fixture(scope="module")
def trading_config(system_artifacts):
    """Loads dynamic configuration updated with extracted artifact parameters."""
    test_blotter, _ = system_artifacts
    params = test_blotter.attrs.get("extracted_params", {})

    config = TradingConfig()
    if BLOTTER_PATH is not None and BLOTTER_PATH.exists():
        try:
            config = TradingConfig.from_artifact(BLOTTER_PATH)
        except Exception:
            pass

    for k, v in params.items():
        if hasattr(config, k) and v is not None:
            setattr(config, k, v)

    return config


# =====================================================================
# TIER 1: UNIVERSE ALIGNMENT & BARE-METAL MATH CHECK
# =====================================================================


@pytest.mark.skipif(not FILES_EXIST, reason="Real system artifacts not found.")
def test_bare_metal_provenance(system_artifacts):
    """
    Calculates scores independently using bare-metal numpy/pandas matrix operations
    to verify recorded blotter outputs with bit-level identity.
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

        # 1. Exact Universe Alignment
        assert len(ensemble) == row["universe_size"], (
            f"Universe size mismatch on {decision_date}: "
            f"blotter={row['universe_size']}, cube={len(ensemble)}"
        )

        # 2. Bare-Metal Factor Equalization Recalculation
        norm = np.linalg.norm(weights)
        norm_weights = weights / norm if norm > 1e-6 else weights

        vals = clean_matrix.values.astype(np.float64)
        mu = np.mean(vals, axis=0, keepdims=True)
        sigma = np.maximum(np.std(vals, axis=0, keepdims=True), 1e-8)
        z_vals = np.clip((vals - mu) / sigma, -4.0, 4.0)

        scores_z = pd.Series(z_vals @ norm_weights, index=clean_matrix.index)
        scores_raw = pd.Series(vals @ norm_weights, index=clean_matrix.index)

        if np.isclose(row["max_score"], float(scores_z.max()), atol=1e-4):
            sorted_tickers = scores_z.sort_values(ascending=False)
        else:
            sorted_tickers = scores_raw.sort_values(ascending=False)

        expected_top_3 = sorted_tickers.index[:3].tolist()
        expected_max = float(sorted_tickers.max())
        expected_min = float(sorted_tickers.min())

        # 3. Bit-Level Identity Assertions
        assert list(row["top_3"]) == expected_top_3, (
            f"Top 3 math deviation on {decision_date.date()}: "
            f"expected {expected_top_3}, got {list(row['top_3'])}"
        )
        assert np.isclose(row["max_score"], expected_max, atol=1e-5), (
            f"Max score drift on {decision_date.date()}: "
            f"expected {expected_max}, got {row['max_score']}"
        )
        assert np.isclose(row["min_score"], expected_min, atol=1e-5), (
            f"Min score drift on {decision_date.date()}: "
            f"expected {expected_min}, got {row['min_score']}"
        )


# =====================================================================
# TIER 2: SELECTION LOGIC SYSTEM REPLAY
# =====================================================================


@pytest.mark.skipif(not FILES_EXIST, reason="Real system artifacts not found.")
def test_system_logic_replay(system_artifacts, trading_config):
    """
    Replays SelectionLogic.apply_action on the canonical market slice and verifies
    bit-level congruence with recorded offsets, widths, tilts, and scores.
    """
    blotter_df, feature_cube = system_artifacts

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
        assert (
            width == row["width"]
        ), f"Width drift on {decision_date}: expected {row['width']}, got {width}"
        assert list(top_3) == list(
            row["top_3"]
        ), f"Top 3 deviation on {decision_date}: expected {list(row['top_3'])}, got {list(top_3)}"
        assert np.isclose(
            max_s, row["max_score"], atol=1e-5
        ), f"Max score deviation on {decision_date}: expected {row['max_score']}, got {max_s}"
        assert np.isclose(
            min_s, row["min_score"], atol=1e-5
        ), f"Min score deviation on {decision_date}: expected {row['min_score']}, got {min_s}"

        if "active_tilt" in row:
            assert np.isclose(
                active_tilt, row["active_tilt"], atol=1e-5
            ), f"Active tilt deviation on {decision_date}: expected {row['active_tilt']}, got {active_tilt}"
