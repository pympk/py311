"""
Generalized Mathematical Invariant Test Suite for Production Walk-Forward Blotters.
Dynamically validates constituent and blended blotters without hardcoding arbitrary
metric numbers. Asserts mathematical consistency, geometric wealth tracking,
dynamic ex-post capital compounding, and ensembling diversification laws.
"""

from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import pytest

from core.paths import OUTPUT_DIR


@pytest.fixture
def gen_blotters(gen: int) -> Dict[str, pd.DataFrame]:
    """
    Dynamically locates and loads all constituent and blended blotters for generation `gen`.
    Prioritizes output/canonical_anchors/ ground-truth vault for promoted production
    baselines, falling back to OUTPUT_DIR for active in-flight experiments.
    """
    if not OUTPUT_DIR.exists():
        pytest.skip(f"Output directory {OUTPUT_DIR} does not exist.")

    canonical_dir = OUTPUT_DIR / "canonical_anchors"
    has_canonical = (
        canonical_dir.exists()
        and (
            canonical_dir / f"blotter_continuous_ex_post_blend_gen{gen}.parquet"
        ).exists()
        and len(list(canonical_dir.glob(f"blotter_continuous_s*_gen{gen}.parquet")))
        >= 2
    )
    search_dir = canonical_dir if has_canonical else OUTPUT_DIR

    blend_file = search_dir / f"blotter_continuous_ex_post_blend_gen{gen}.parquet"
    if not blend_file.exists():
        pytest.skip(
            f"Gen {gen} blended blotter not found in {search_dir.name}: {blend_file.name}"
        )

    seed_files = sorted(
        list(search_dir.glob(f"blotter_continuous_s*_gen{gen}.parquet"))
    )
    if not seed_files:
        pytest.skip(f"No constituent seed blotters found for Gen {gen} in {search_dir}")

    blotters: Dict[str, pd.DataFrame] = {}
    for p in seed_files:
        # Extract seed tag (e.g., 's42' from 'blotter_continuous_s42_gen18.parquet')
        seed_tag = p.stem.split("_")[2]
        df = pd.read_parquet(p)
        date_col = "date" if "date" in df.columns else "decision_date"
        df["date"] = pd.to_datetime(df[date_col])
        df = df.sort_values("date").reset_index(drop=True)
        blotters[seed_tag] = df

    df_blend = pd.read_parquet(blend_file)
    date_col = "date" if "date" in df_blend.columns else "decision_date"
    df_blend["date"] = pd.to_datetime(df_blend[date_col])
    df_blend = df_blend.sort_values("date").reset_index(drop=True)
    blotters["blend"] = df_blend

    return blotters


# =====================================================================
# GENERALIZED PARAMETERIZED INVARIANT TESTS
# =====================================================================


@pytest.mark.parametrize("gen", [17, 18])
def test_temporal_alignment_and_monotonicity(
    gen: int, gen_blotters: Dict[str, pd.DataFrame]
):
    """
    INVARIANT 1: Date Alignment & Monotonic Continuity
    All constituent and blended blotters must have identical date indices with
    strictly monotonic ordering and zero temporal duplicates.
    """
    ref_dates = gen_blotters["blend"]["date"]
    assert len(ref_dates) > 0, f"Gen {gen} blend blotter is empty"
    assert ref_dates.is_monotonic_increasing, f"Gen {gen} blend dates are not monotonic"
    assert not ref_dates.duplicated().any(), f"Gen {gen} contains duplicate dates"

    for name, df in gen_blotters.items():
        assert len(df) == len(
            ref_dates
        ), f"{name} length mismatch: {len(df)} vs {len(ref_dates)}"
        assert df["date"].equals(
            ref_dates
        ), f"{name} date index diverges from blend reference"


@pytest.mark.parametrize("gen", [17, 18])
def test_linear_return_blending_invariant(
    gen: int, gen_blotters: Dict[str, pd.DataFrame]
):
    """
    INVARIANT 2: Portfolio Return Combination Law
    Validates that the blended portfolio return strictly obeys either:
      1. Equal-Weight Daily Rebalancing: r_blend(t) == (1/M) * sum(r_m(t))
      2. Ex-Post Capital Compounding:    r_blend(t) == sum(w_{m, t-1} * r_m(t))
    where w_{m, t-1} = Equity_{m, t-1} / sum_k Equity_{k, t-1}.
    """
    seeds = [k for k in gen_blotters.keys() if k != "blend"]
    assert (
        len(seeds) >= 2
    ), f"Gen {gen} has insufficient seeds ({len(seeds)}) to test ensembling"

    rets_stack = np.stack(
        [gen_blotters[s]["net_daily_simple_ret"].to_numpy(dtype=float) for s in seeds],
        axis=0,
    )
    recorded_blend = gen_blotters["blend"]["net_daily_simple_ret"].to_numpy(dtype=float)

    # Formulation 1: Daily Arithmetic Rebalancing
    expected_arithmetic = np.mean(rets_stack, axis=0)

    # Formulation 2: Ex-Post Capital Blend (Wealth-Weighted dynamic sleeve compounding)
    equities_stack = np.cumprod(1.0 + rets_stack, axis=1)  # shape (M, T)
    lagged_equities = np.pad(
        equities_stack[:, :-1], ((0, 0), (1, 0)), constant_values=1.0
    )
    capital_weights = lagged_equities / np.sum(lagged_equities, axis=0, keepdims=True)
    expected_capital_blend = np.sum(capital_weights * rets_stack, axis=0)

    diff_arith = float(np.max(np.abs(recorded_blend - expected_arithmetic)))
    diff_capital = float(np.max(np.abs(recorded_blend - expected_capital_blend)))

    assert (diff_arith < 1e-5) or (diff_capital < 1e-5), (
        f"Ex-post blend return diverges from constituent combination laws in Gen {gen}!\n"
        f"  Max residual vs Daily Arithmetic Mean: {diff_arith:.6e}\n"
        f"  Max residual vs Ex-Post Capital Blend: {diff_capital:.6e}"
    )


@pytest.mark.parametrize("gen", [17, 18])
def test_benchmark_return_identity_invariant(
    gen: int, gen_blotters: Dict[str, pd.DataFrame]
):
    """
    INVARIANT 3: Benchmark Uniformity
    All constituent seed blotters and the blend must execute against the identical benchmark series.
    """
    bm_blend = gen_blotters["blend"]["bm_daily_simple_ret"].to_numpy(dtype=float)
    for name, df in gen_blotters.items():
        if name == "blend":
            continue
        bm_seed = df["bm_daily_simple_ret"].to_numpy(dtype=float)
        np.testing.assert_allclose(
            bm_seed,
            bm_blend,
            rtol=1e-7,
            atol=1e-9,
            err_msg=f"Benchmark return mismatch detected in {name} (Gen {gen})",
        )


@pytest.mark.parametrize("gen", [17, 18])
def test_geometric_alpha_ratio_invariant(
    gen: int, gen_blotters: Dict[str, pd.DataFrame]
):
    """
    INVARIANT 4: Geometric Wealth Multiplier Invariant
    Alpha Equity must strictly track:
        Alpha Multiplier(t) = Agent Equity(t) / Benchmark Equity(t).
    Additive compounding (1 + sum(alpha)) is strictly forbidden.
    """
    for name, df in gen_blotters.items():
        agent_equity = (1.0 + df["net_daily_simple_ret"]).cumprod().to_numpy()
        bm_equity = (1.0 + df["bm_daily_simple_ret"]).cumprod().to_numpy()
        expected_alpha = agent_equity / np.maximum(bm_equity, 1e-8)

        recorded_alpha = df["alpha_equity"].to_numpy(dtype=float)
        np.testing.assert_allclose(
            recorded_alpha,
            expected_alpha,
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"Geometric alpha ratio violated in {name} (Gen {gen})",
        )


@pytest.mark.parametrize("gen", [17, 18])
def test_diversification_ratio_superiority(
    gen: int, gen_blotters: Dict[str, pd.DataFrame]
):
    """
    INVARIANT 5: Ensembling Diversification Superiority Law
    1. Diversification Ratio: DR = sum(w_i * sigma_i) / sigma_blend > 1.0.
       Guarantees ensemble volatility is strictly lower than the constituent weighted volatility.
    2. Blended Sharpe strictly expands beyond the arithmetic mean constituent Sharpe.
    """
    seeds = [k for k in gen_blotters.keys() if k != "blend"]
    blend_df = gen_blotters["blend"]

    seed_vols = [
        float(
            np.std(
                gen_blotters[s]["net_daily_simple_ret"].to_numpy(dtype=float), ddof=1
            )
        )
        for s in seeds
    ]
    mean_seed_vol = float(np.mean(seed_vols))
    blend_vol = float(
        np.std(blend_df["net_daily_simple_ret"].to_numpy(dtype=float), ddof=1)
    )

    div_ratio = mean_seed_vol / blend_vol
    assert (
        div_ratio > 1.0
    ), f"Diversification Ratio failed in Gen {gen}: {div_ratio:.4f} <= 1.0"

    seed_sharpes = []
    for s in seeds:
        rets = gen_blotters[s]["net_daily_simple_ret"].to_numpy(dtype=float)
        s_std = float(np.std(rets, ddof=1))
        s_sharpe = float((np.mean(rets) / s_std) * np.sqrt(252)) if s_std > 0 else 0.0
        seed_sharpes.append(s_sharpe)

    blend_rets = blend_df["net_daily_simple_ret"].to_numpy(dtype=float)
    b_std = float(np.std(blend_rets, ddof=1))
    blend_sharpe = (
        float((np.mean(blend_rets) / b_std) * np.sqrt(252)) if b_std > 0 else 0.0
    )

    mean_constituent_sharpe = float(np.mean(seed_sharpes))
    assert blend_sharpe > mean_constituent_sharpe, (
        f"Ensemble failed to expand Sharpe in Gen {gen}! Blend: {blend_sharpe:.3f}, "
        f"Constituent Mean: {mean_constituent_sharpe:.3f}"
    )


@pytest.mark.parametrize("gen", [17, 18])
def test_gross_exposure_and_zero_cash_mandate(
    gen: int, gen_blotters: Dict[str, pd.DataFrame]
):
    """
    INVARIANT 6: Zero Cash & Full Market Exposure Mandate
    Asserts equity_exposure == 1.0 and weight_cash == 0.0 across all sessions.
    """
    for name, df in gen_blotters.items():
        if "equity_exposure" in df.columns:
            exp = df["equity_exposure"].to_numpy(dtype=float)
            np.testing.assert_allclose(
                exp,
                1.0,
                atol=1e-4,
                err_msg=f"Gross equity exposure is not 1.0 in {name} (Gen {gen})",
            )
        if "weight_cash" in df.columns:
            cash = df["weight_cash"].to_numpy(dtype=float)
            np.testing.assert_allclose(
                cash,
                0.0,
                atol=1e-4,
                err_msg=f"Cash hoarding detected in {name} (Gen {gen}): weight_cash != 0.0",
            )
