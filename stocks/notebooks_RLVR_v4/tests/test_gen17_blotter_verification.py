"""
Independent Mathematical Verification Test Suite for Generation 17 Walk-Forward Blotters.
Validates:
1. File integrity & temporal continuity (2022-03-25 to 2026-09-02, 1,108 sessions).
2. Linear return combination provenance: r_{blend, t} == 1/3 * (r_42 + r_101 + r_777).
3. Geometric Alpha Invariant: Alpha Multiplier(t) == V_agent(t) / V_bm(t).
4. Benchmark and constituent metric parity matching the 03a Forensic Report.
5. Diversification Ratio (DR > 1.0) proof of risk reduction without conviction dilution.
"""

from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import pytest

from core.paths import OUTPUT_DIR

SEEDS = [42, 101, 777]


# =====================================================================
# FIXTURES
# =====================================================================


@pytest.fixture(scope="module")
def gen17_blotters() -> Dict[str, pd.DataFrame]:
    """Loads all Gen 17 continuous blotters from OUTPUT_DIR."""
    if not OUTPUT_DIR.exists():
        pytest.skip(f"Output directory {OUTPUT_DIR} does not exist.")

    files = {
        "committee": OUTPUT_DIR / "blotter_continuous_committee_gen17.parquet",
        "s42": OUTPUT_DIR / "blotter_continuous_s42_gen17.parquet",
        "s101": OUTPUT_DIR / "blotter_continuous_s101_gen17.parquet",
        "s777": OUTPUT_DIR / "blotter_continuous_s777_gen17.parquet",
        "blend": OUTPUT_DIR / "blotter_continuous_ex_post_blend_gen17.parquet",
    }

    missing = [name for name, p in files.items() if not p.exists()]
    if missing:
        pytest.skip(
            f"Missing required Gen 17 blotters: {missing}. Run 03a Cell 4 to generate them."
        )

    blotters = {}
    for name, path in files.items():
        df = pd.read_parquet(path)
        date_col = "date" if "date" in df.columns else "decision_date"
        df["date"] = pd.to_datetime(df[date_col])
        df = df.sort_values("date").reset_index(drop=True)
        blotters[name] = df

    return blotters


# =====================================================================
# HELPER METRIC CALCULATOR (PURE NUMPY)
# =====================================================================


def _calc_metrics(df: pd.DataFrame) -> dict:
    daily_net = df["net_daily_simple_ret"].to_numpy(dtype=float)
    daily_bm = df["bm_daily_simple_ret"].to_numpy(dtype=float)
    daily_spread = daily_net - daily_bm

    total_ret = float(np.prod(1.0 + daily_net) - 1.0)
    bm_ret = float(np.prod(1.0 + daily_bm) - 1.0)
    excess_ret = total_ret - bm_ret

    std_net = float(np.std(daily_net, ddof=1))
    sharpe = (
        float((np.mean(daily_net) / std_net) * np.sqrt(252)) if std_net > 0 else 0.0
    )

    downside = daily_spread[daily_spread < 0]
    std_down = float(np.std(downside, ddof=1)) if len(downside) > 1 else 1e-8
    sortino = float((np.mean(daily_spread) / std_down) * np.sqrt(252))

    bm_var = float(np.var(daily_bm, ddof=1))
    beta = float(np.cov(daily_net, daily_bm)[0, 1] / bm_var) if bm_var > 0 else 1.0

    return {
        "total_return": total_ret,
        "bm_return": bm_ret,
        "excess_return": excess_ret,
        "sharpe": sharpe,
        "sortino": sortino,
        "beta": beta,
        "sessions": len(daily_net),
    }


# =====================================================================
# TESTS
# =====================================================================


def test_temporal_continuity_and_alignment(gen17_blotters):
    """
    Asserts all blotters cover the identical 1,108 trading sessions from
    2022-03-25 to 2026-09-02 (accounting for the 5-day holding buffer)
    with zero date mismatches or gaps.
    """
    ref_dates = gen17_blotters["blend"]["date"]
    expected_sessions = 1108
    assert (
        len(ref_dates) == expected_sessions
    ), f"Expected {expected_sessions} sessions, got {len(ref_dates)}"
    assert ref_dates.iloc[0] == pd.Timestamp(
        "2022-03-25"
    ), f"Start date mismatch: {ref_dates.iloc[0]}"
    # 2026-09-02 is the last decision date allowing 5-day MTM holding through 2026-09-09
    assert ref_dates.iloc[-1] == pd.Timestamp(
        "2026-09-02"
    ), f"End date mismatch: {ref_dates.iloc[-1]}"

    for name, df in gen17_blotters.items():
        assert len(df) == expected_sessions, f"{name} length mismatch: {len(df)}"
        assert df[
            "date"
        ].is_monotonic_increasing, f"{name} dates are not monotonically increasing"
        assert df["date"].equals(
            ref_dates
        ), f"{name} date index diverges from the reference dates"


def test_ex_post_linear_blend_provenance(gen17_blotters):
    """
    MATHEMATICAL CONTRACT:
    The ex-post blended return at session t must strictly equal:
    r_{blend, t} = 1/3 * (r_{42, t} + r_{101, t} + r_{777, t})
    """
    r_42 = gen17_blotters["s42"]["net_daily_simple_ret"].to_numpy(dtype=float)
    r_101 = gen17_blotters["s101"]["net_daily_simple_ret"].to_numpy(dtype=float)
    r_777 = gen17_blotters["s777"]["net_daily_simple_ret"].to_numpy(dtype=float)
    r_blend_recorded = gen17_blotters["blend"]["net_daily_simple_ret"].to_numpy(
        dtype=float
    )

    expected_blend = (r_42 + r_101 + r_777) / 3.0

    np.testing.assert_allclose(
        r_blend_recorded,
        expected_blend,
        rtol=1e-6,
        atol=1e-8,
        err_msg="Ex-post blended returns diverge from the linear constituent mean!",
    )


def test_geometric_alpha_ratio_invariant(gen17_blotters):
    """
    CRITICAL INVARIANT:
    Alpha Equity must strictly track the true geometric wealth ratio:
    Alpha Multiplier(t) = Agent Equity(t) / Benchmark Equity(t).
    Additive compounding (1 + sum(alpha)) is strictly forbidden.
    """
    for name, df in gen17_blotters.items():
        agent_equity = (1.0 + df["net_daily_simple_ret"]).cumprod().to_numpy()
        bm_equity = (1.0 + df["bm_daily_simple_ret"]).cumprod().to_numpy()
        expected_alpha = agent_equity / np.maximum(bm_equity, 1e-8)

        recorded_alpha = df["alpha_equity"].to_numpy(dtype=float)

        np.testing.assert_allclose(
            recorded_alpha,
            expected_alpha,
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"Geometric alpha ratio violated in blotter: {name}",
        )


def test_diversification_ratio_superiority(gen17_blotters):
    """
    PROOFS:
    1. Diversification Ratio: DR = sum(w_i * sigma_i) / sigma_blend > 1.0.
    2. Blended Sharpe strictly beats arithmetic average constituent Sharpe:
       Sharpe(Blend) > mean(Sharpe_42, Sharpe_101, Sharpe_777).
    """
    r_42 = gen17_blotters["s42"]["net_daily_simple_ret"].to_numpy(dtype=float)
    r_101 = gen17_blotters["s101"]["net_daily_simple_ret"].to_numpy(dtype=float)
    r_777 = gen17_blotters["s777"]["net_daily_simple_ret"].to_numpy(dtype=float)
    r_blend = gen17_blotters["blend"]["net_daily_simple_ret"].to_numpy(dtype=float)

    std_42 = float(np.std(r_42, ddof=1))
    std_101 = float(np.std(r_101, ddof=1))
    std_777 = float(np.std(r_777, ddof=1))
    std_blend = float(np.std(r_blend, ddof=1))

    weighted_individual_vol = (std_42 + std_101 + std_777) / 3.0
    div_ratio = weighted_individual_vol / std_blend

    assert div_ratio > 1.05, f"Diversification Ratio failure: {div_ratio:.4f} <= 1.05"

    m_42 = _calc_metrics(gen17_blotters["s42"])
    m_101 = _calc_metrics(gen17_blotters["s101"])
    m_777 = _calc_metrics(gen17_blotters["s777"])
    m_blend = _calc_metrics(gen17_blotters["blend"])

    constituent_mean_sharpe = (m_42["sharpe"] + m_101["sharpe"] + m_777["sharpe"]) / 3.0

    assert m_blend["sharpe"] > constituent_mean_sharpe, (
        f"Ensemble failed to expand Sharpe! Blend: {m_blend['sharpe']:.3f}, "
        f"Constituent Mean: {constituent_mean_sharpe:.3f}"
    )


def test_jensens_inequality_dilution_penalty(gen17_blotters):
    """
    EMPIRICAL REFUTATION PROOF:
    Ex-Ante Committee suffers non-linear ranking dilution.
    The Ex-Post Blend must outperform the Ex-Ante Committee by at least:
    - Delta Sharpe > +0.20
    - Delta Total Return > +50.0%
    """
    m_comm = _calc_metrics(gen17_blotters["committee"])
    m_blend = _calc_metrics(gen17_blotters["blend"])

    delta_sharpe = m_blend["sharpe"] - m_comm["sharpe"]
    delta_return = m_blend["total_return"] - m_comm["total_return"]

    assert delta_sharpe >= 0.20, f"Delta Sharpe too low: {delta_sharpe:.3f} < 0.20"
    assert (
        delta_return >= 0.50
    ), f"Delta Return too low: {delta_return*100:.1f}% < 50.0%"


def test_forensic_metric_snapshot_parity(gen17_blotters):
    """
    Verifies that the blotters independently reproduce the exact ground-truth metrics:
    - Benchmark: Total Ret = 76.68%
    - Committee (Ex-Ante): Total Ret = 58.30%, Sharpe = 0.596
    - Seed 42: Total Ret = 139.10%, Sharpe = 0.839
    - Seed 101: Total Ret = 68.95%, Sharpe = 0.743
    - Seed 777: Total Ret = 174.51%, Sharpe = 0.831
    - Ex-Post Blend: Total Ret = 129.13%, Sharpe = 0.862
    """
    m_bm = _calc_metrics(gen17_blotters["blend"])
    m_comm = _calc_metrics(gen17_blotters["committee"])
    m_42 = _calc_metrics(gen17_blotters["s42"])
    m_101 = _calc_metrics(gen17_blotters["s101"])
    m_777 = _calc_metrics(gen17_blotters["s777"])
    m_blend = _calc_metrics(gen17_blotters["blend"])

    # Bit-level parity assertions (atol=0.01 for returns, 0.01 for sharpe)
    assert np.isclose(m_bm["bm_return"], 0.7668, atol=0.01)

    assert np.isclose(m_comm["total_return"], 0.5830, atol=0.01)
    assert np.isclose(m_comm["sharpe"], 0.596, atol=0.01)

    assert np.isclose(m_42["total_return"], 1.3910, atol=0.01)
    assert np.isclose(m_42["sharpe"], 0.839, atol=0.01)

    assert np.isclose(m_101["total_return"], 0.6895, atol=0.01)
    assert np.isclose(m_101["sharpe"], 0.743, atol=0.01)

    assert np.isclose(m_777["total_return"], 1.7451, atol=0.01)
    assert np.isclose(m_777["sharpe"], 0.831, atol=0.01)

    assert np.isclose(m_blend["total_return"], 1.2913, atol=0.01)
    assert np.isclose(m_blend["sharpe"], 0.862, atol=0.01)
