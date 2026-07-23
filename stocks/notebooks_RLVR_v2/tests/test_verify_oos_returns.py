import pytest

from pathlib import Path

from core.auditor import SystemAuditor
from core.paths import GLOBAL_PROCESSED_DIR, OUTPUT_DIR


def test_real_oos_returns_integration():
    """
    INTEGRATION TEST: Finds the most recent RL output pickle file and
    audits it against the actual market parquet data to ensure 0% math leakage.
    """
    # 1. Verify market data exists
    market_data_path = GLOBAL_PROCESSED_DIR / "df_ohlcv.parquet"
    if not market_data_path.exists():
        pytest.skip(
            f"Market data not found at {market_data_path}. Skipping integration test."
        )

    # 2. Find the most recent OOS pickle file in the output directory
    if not OUTPUT_DIR.exists():
        pytest.skip("Output directory does not exist yet. Run training notebook first.")

    pkl_files = list(OUTPUT_DIR.glob("oos_results_*.pkl"))
    if not pkl_files:
        pytest.skip("No OOS results pickle files found in output directory.")

    # Get the latest file by modification time
    latest_pkl = max(pkl_files, key=lambda p: p.stat().st_mtime)

    print(f"\n[Auditor] Auditing latest RL run: {latest_pkl.name}")

    # 3. Run the Auditor
    verification_df = SystemAuditor.audit_oos_results(
        pkl_path=latest_pkl, df_ohlcv_path=market_data_path, slippage_bps=5.0
    )

    # 4. Assert the maximum divergence is negligible (less than 1 basis point)
    max_diff = verification_df["Difference"].abs().max()

    assert max_diff < 1e-4, (
        f"❌ FAILED: RL Environment contains forward-looking leaks or price misalignment! "
        f"Max divergence was {max_diff:.6f}"
    )

    print(f"✅ PASSED: Integration math verified. Max divergence: {max_diff:.6f}")
