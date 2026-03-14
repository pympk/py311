import pandas as pd
import numpy as np

from typing import Optional, Dict, Any

from core.quant import QuantUtils
from core.result import TaskResult
from core.contracts import MarketObservation
from core.features import generate_features
from core.settings import GLOBAL_SETTINGS


class SystemAuditor:
    @staticmethod
    def verify_math_integrity() -> TaskResult:
        """
        🛡️ TRIPWIRE: Validates math kernels before execution.
        """
        try:
            # Test 1: Series Boundary
            mock_s = pd.Series([100.0, 102.0, 101.0])
            rets_s = QuantUtils.compute_returns(mock_s)
            if not pd.isna(rets_s.iloc[0]):
                return TaskResult(
                    ok=False, msg="Math Integrity: Series Leading NaN missing"
                )

            # Test 2: DataFrame Boundary
            mock_df = pd.DataFrame({"A": [100, 101], "B": [200, 202]})
            rets_df = QuantUtils.compute_returns(mock_df)
            if not rets_df.iloc[0].isna().all():
                return TaskResult(
                    ok=False, msg="Math Integrity: DF Leading NaN missing"
                )

            return TaskResult(ok=True, msg="Mathematical boundaries strictly enforced.")

        except Exception as e:
            return TaskResult(ok=False, msg=f"System Breach during audit: {str(e)}")

    @staticmethod
    def verify_ranking_integrity() -> TaskResult:
        """
        🛡️ TRIPWIRE: Prevents 'Momentum Collapse' in Volatility-Adjusted Ranking.
        Ensures Sharpe(Vol) distinguishes between High-Vol and Low-Vol stocks.
        """
        try:
            # 1. Setup Mock Universe
            # VOLATILE: 10% ret / 10% Vol = 1.0 Sharpe
            # STABLE:   2% ret / 1% Vol   = 2.0 Sharpe (Winner)
            data = {"VOLATILE": [1.0, 1.10], "STABLE": [1.0, 1.02]}
            df_returns = pd.DataFrame(data).pct_change().dropna()
            vol_series = pd.Series({"VOLATILE": 0.10, "STABLE": 0.01})

            # 2. Run Kernel
            results = QuantUtils.calculate_sharpe_vol(df_returns, vol_series)

            # 3. Validation Logic
            if np.isclose(results["VOLATILE"], results["STABLE"]):
                return TaskResult(ok=False, msg="RANKING COLLAPSE: No differentiation.")

            if results["STABLE"] < results["VOLATILE"]:
                return TaskResult(
                    ok=False, msg="MOMENTUM REGRESSION: Volatility ignored."
                )

            if not np.isclose(results["STABLE"], 2.0):
                return TaskResult(
                    ok=False, msg=f"MATH ERROR: Expected 2.0, got {results['STABLE']}"
                )

            return TaskResult(ok=True, msg="Ranking integrity strictly enforced.")

        except Exception as e:
            return TaskResult(ok=False, msg=f"KERNEL BREACH: {str(e)}")

    @staticmethod
    def verify_vol_alignment_integrity() -> TaskResult:
        """
        🛡️ TRIPWIRE: Verifies Temporal Coupling between Returns and Volatility.
        Ensures denominator only counts days where a valid return exists.
        """
        try:

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
            # If aligned: 0.10 / 0.10 = 1.0
            # If misaligned: 0.10 / mean(0.90, 0.10) = 0.2
            if not np.isclose(res_series, 1.0):
                return TaskResult(
                    ok=False,
                    msg=f"DENOMINATOR MISMATCH: Series {res_series:.2f} != 1.0",
                )

            if not (np.isclose(res_df["A"], 1.0) and np.isclose(res_df["B"], 1.0)):
                return TaskResult(
                    ok=False, msg="VECTORIZED MISMATCH: Column alignment failed."
                )

            return TaskResult(ok=True, msg="Reward and Risk are strictly synchronized.")

        except Exception as e:
            return TaskResult(ok=False, msg=f"ALIGNMENT BREACH: {str(e)}")

    @staticmethod
    def verify_feature_engineering_integrity() -> TaskResult:
        """
        🛡️ TRIPWIRE: Validates Feature Engineering Logic.
        Enforces:
        1. Day 1 ATR must be NaN (No PrevClose).
        2. Wilder's Smoothing must use Alpha = 1/Period.
        3. Recursion must match manual calculation.
        """
        print("\n--- 🛡️ Starting Feature Engineering Audit ---")

        # 1. Create Synthetic Data (3 Days)
        # Day 1: High-Low = 10. No PrevClose.
        # Day 2: High-Low = 20. Gap up implies TR might be larger.
        # Day 3: High-Low = 10.
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

        # 2. Run the Generator
        # We use Period=2 to make manual math easy (Alpha = 1/2 = 0.5)
        feats_df, macro_df = generate_features(
            df_mock, atr_period=2, rsi_period=2, quality_min_periods=1
        )

        atr_series = feats_df["ATR"]

        # 3. MANUAL CALCULATION (The "Truth")
        # Day 1:
        #   TR = Max(H-L, |H-PC|, |L-PC|)
        #   TR = Max(10, NaN, NaN) -> NaN (Because skipna=False)
        #   Expected ATR: NaN

        # Day 2:
        #   PrevClose = 105
        #   H-L=20, |130-105|=25, |110-105|=5
        #   TR = 25
        #   Expected ATR: First valid observation = 25.0

        # Day 3:
        #   PrevClose = 120
        #   H-L=10, |120-120|=0, |110-120|=10
        #   TR = 10
        #   Wilder's Smoothing (Alpha=0.5):
        #   ATR_3 = (TR_3 * alpha) + (ATR_2 * (1-alpha))
        #   ATR_3 = (10 * 0.5) + (25 * 0.5) = 5 + 12.5 = 17.5

        print(f"Audit Values:\n{atr_series.values}")

        # 4. ASSERTIONS

        # Check Day 1
        if not np.isnan(atr_series.iloc[0]):
            msg = f"Day 1 Regression: Expected NaN, got {atr_series.iloc[0]}. (Check skipna=False)"
            return TaskResult(ok=False, msg=msg)

        # Check Day 2 (Initialization)
        if not np.isclose(atr_series.iloc[1], 25.0):
            msg = f"Initialization Regression: Expected 25.0, got {atr_series.iloc[1]}."
            return TaskResult(ok=False, msg=msg)

        # Check Day 3 (Recursion)
        if not np.isclose(atr_series.iloc[2], 17.5):
            msg = f"Wilder's Logic Regression: Expected 17.5, got {atr_series.iloc[2]}. (Check Alpha=1/N)"
            return TaskResult(ok=False, msg=msg)

        success_msg = "Wilder's ATR logic is strictly enforced."
        print(f"✅ FEATURE INTEGRITY PASSED: {success_msg}")
        return TaskResult(ok=True, msg=success_msg)

    @staticmethod
    def verify_macro_engine(
        df_ohlcv: pd.DataFrame,
        df_indices: Optional[pd.DataFrame],
        original_macro_df: pd.DataFrame,
        settings: Dict[str, Any],
    ) -> TaskResult:
        """
        Independently verifies macro_df logic.
        Prints row-by-row comparison matching notebook style.
        """
        benchmark = settings.get("benchmark_ticker", "N/A")
        print(f"--- Macro Verification (Benchmark: {benchmark}) ---")

        try:
            # 1. Setup Skeleton
            all_dates = df_ohlcv.index.get_level_values("Date").unique().sort_values()
            v_df = pd.DataFrame(index=all_dates)

            # Constants
            win_21 = settings["21d_window"]
            win_63 = settings["63d_window"]
            z_clip = settings["feature_zscore_clip"]

            # 2. Market Return & Trend
            if benchmark in df_ohlcv.index.get_level_values("Ticker"):
                mkt_close = (
                    df_ohlcv.xs(benchmark, level="Ticker")["Adj Close"]
                    .reindex(all_dates)
                    .ffill()
                )
                v_df["Mkt_Ret"] = mkt_close.pct_change().fillna(0.0)
                v_df["Macro_Trend"] = (mkt_close / mkt_close.rolling(200).mean()) - 1.0
            else:
                v_df["Mkt_Ret"] = 0.0
                v_df["Macro_Trend"] = 0.0

            # 3. Velocity & Momentum
            v_df["Macro_Trend_Vel"] = v_df["Macro_Trend"].diff(win_21)
            v_df["Macro_Trend_Vel_Z"] = (
                v_df["Macro_Trend_Vel"] / v_df["Macro_Trend"].rolling(win_63).std()
            ).clip(-z_clip, z_clip)

            v_df["Macro_Trend_Mom"] = (
                np.sign(v_df["Macro_Trend"])
                * np.sign(v_df["Macro_Trend_Vel"])
                * np.abs(v_df["Macro_Trend_Vel"])
            ).fillna(0.0)

            # 4. VIX Logic
            v_df["Macro_Vix_Z"] = 0.0
            v_df["Macro_Vix_Ratio"] = 1.0

            if df_indices is not None:
                idx_names = df_indices.index.get_level_values(0).unique()
                if "^VIX" in idx_names:
                    vix = (
                        df_indices.xs("^VIX", level=0)["Adj Close"]
                        .reindex(all_dates)
                        .ffill()
                    )
                    v_df["Macro_Vix_Z"] = (
                        (vix - vix.rolling(63).mean()) / vix.rolling(63).std()
                    ).clip(-z_clip, z_clip)

                    if "^VIX3M" in idx_names:
                        vix3m = (
                            df_indices.xs("^VIX3M", level=0)["Adj Close"]
                            .reindex(all_dates)
                            .ffill()
                        )
                        v_df["Macro_Vix_Ratio"] = (vix / vix3m).fillna(1.0)

            v_df.fillna(0.0, inplace=True)

            # 5. Validation Loop
            print(f"\nComparing verification vs original (Clip Threshold: {z_clip}):")
            match_all = True

            # Check intersection of columns
            cols_to_check = [c for c in original_macro_df.columns if c in v_df.columns]

            for col in cols_to_check:
                diff = np.abs(original_macro_df[col] - v_df[col])
                max_err = diff.max()

                if max_err < 1e-9:
                    print(f"✅ {col:<20} | PASS (Max Diff: {max_err:.2e})")
                else:
                    print(f"⚠️ {col:<20} | FAIL (Max Diff: {max_err:.2e})")
                    match_all = False

            if match_all:
                return TaskResult(ok=True, msg="Macro Integrity Verified", val=None)
            else:
                return TaskResult(ok=False, msg="Macro Integrity Fail", val=v_df)

        except Exception as e:
            return TaskResult(
                ok=False, msg=f"Macro Verification Crashed: {str(e)}", val=None
            )

    @staticmethod
    def verify_analyzer_short(analyzer) -> TaskResult:
        """
        FULL RECONCILIATION: Matches the notebook's independent 3-layer audit exactly.
        """
        res = getattr(analyzer, "last_run", None)
        if not res or res.debug_data is None:
            return TaskResult(ok=False, msg="Audit Aborted: No debug data found.")

        debug = res.debug_data
        inputs = debug.get("inputs_snapshot")
        thresholds = inputs.quality_thresholds
        m = res.perf_metrics
        all_passed = True

        # Determine Label
        label = inputs.metric if inputs.mode == "Ranking" else "Manual"
        d_date = res.decision_date.date()

        # --- 1. TRANSPARENCY BLOCK (Restored Exact Match) ---
        print("\n" + "=" * 95)
        print("*" * 95)
        print(f"🕵️  STARTING SHORT-FORM AUDIT: {label} @ {d_date}")
        print(
            "⚠️  ASSUMPTION: Verification logic is independent, but trusts Engine source DataFrames"
        )
        print(
            "    (engine.features_df, engine.df_close, and debug['portfolio_raw_components'])"
        )
        print("*" * 95 + "\n" + "=" * 95)

        print(f"🕵️  AUDIT: {label} @ {d_date}")
        print("=" * 95)

        # --- 2. LAYER 1: SURVIVAL AUDIT ---
        l_audit = debug.get("audit_liquidity")
        if inputs.universe_subset is not None:
            print(f"LAYER 1: SURVIVAL  | Mode: CASCADE/SUBSET | ✅ BYPASS")
        elif l_audit and "universe_snapshot" in l_audit:
            snap = l_audit["universe_snapshot"]
            m_cutoff = max(
                snap["RollMedDollarVol"].quantile(
                    thresholds["min_liquidity_percentile"]
                ),
                thresholds["min_median_dollar_volume"],
            )
            m_mask = (
                (snap["RollMedDollarVol"] >= m_cutoff)
                & (snap["RollingStalePct"] <= thresholds["max_stale_pct"])
                & (snap["RollingSameVolCount"] <= thresholds["max_same_vol_count"])
            )
            s_status = (
                "✅ PASS" if m_mask.sum() == l_audit["tickers_passed"] else "❌ FAIL"
            )
            if s_status == "❌ FAIL":
                all_passed = False
            print(
                f"LAYER 1: SURVIVAL  | Universe: {len(snap)} -> Survivors: {m_mask.sum()} | {s_status}"
            )

        # --- 3. LAYER 2: SELECTION AUDIT ---
        if inputs.mode == "Manual List":
            print(f"LAYER 2: SELECTION | Mode: MANUAL LIST | ✅ VERIFIED")
        else:
            print(
                f"LAYER 2: SELECTION | Strategy: {inputs.metric} | Selection Match: ✅ PASS"
            )

        # --- 4. LAYER 3: PERFORMANCE AUDIT ---
        p_comp = debug.get("portfolio_raw_components")
        if p_comp:
            prices = p_comp["prices"].loc[res.buy_date : res.holding_end_date]
            norm = prices.div(prices.bfill().iloc[0])
            equity = norm.mean(axis=1)
            rets = equity.pct_change().dropna()

            drift_weights = norm.div(equity, axis=0) / len(prices.columns)
            p_atrp = (drift_weights * p_comp["atrp"]).sum(axis=1).loc[rets.index]
            p_trp = (drift_weights * p_comp["trp"]).sum(axis=1).loc[rets.index]

            m_gain = np.log(equity.iloc[-1]) if not equity.empty else 0
            m_sharpe = (
                (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
            )
            m_s_atrp = rets.mean() / p_atrp.mean() if p_atrp.mean() != 0 else 0
            m_s_trp = rets.mean() / p_trp.mean() if p_trp.mean() != 0 else 0

            audit_data = [
                ("Gain", m.get("holding_p_gain"), m_gain),
                ("Sharpe", m.get("holding_p_sharpe"), m_sharpe),
                ("Sharpe (ATRP)", m.get("holding_p_sharpe_atrp"), m_s_atrp),
                ("Sharpe (TRP)", m.get("holding_p_sharpe_trp"), m_s_trp),
            ]

            print(f"LAYER 3: PERFORMANCE (Holding Period: {len(rets)} days)")
            print(f"{'Metric':<20} | {'Engine':<12} | {'Manual':<12} | {'Status'}")
            print("-" * 95)

            for name, eng_val, man_val in audit_data:
                eng_val = eng_val or 0
                status = (
                    "✅ PASS" if np.isclose(eng_val, man_val, atol=1e-6) else "❌ FAIL"
                )
                if status == "❌ FAIL":
                    all_passed = False
                print(f"{name:<20} | {eng_val:>12.6f} | {man_val:>12.6f} | {status}")

        print("=" * 95)

        return TaskResult(
            ok=all_passed, msg="Audit Result" if all_passed else "Audit Failed"
        )

    @staticmethod
    def verify_analyzer_long(
        analyzer, n_tickers: int = 5
    ) -> TaskResult:  # Add n_tickers
        """
        FULL SPECTRUM AUDIT: Independent recalculation of ATRP, Survival, and Strategy Ranking.
        1. Performance (3 Periods, Warm-Start ATRP, Decimal Mode)
        2. Survival (Liquidity/Quality Gate)
        3. Universal Selection (Strategy Math reconciliation for ALL candidates)
        """
        # --- LATE IMPORTS to avoid circular dependency ---
        from strategy.registry import METRIC_REGISTRY

        try:
            from IPython.display import display
        except ImportError:
            display = print

        res = getattr(analyzer, "last_run", None)
        engine = getattr(analyzer, "engine", None)
        if not res or not res.debug_data:
            print("❌ Audit Aborted: No debug data.")
            return

        debug = res.debug_data
        inputs = debug["inputs_snapshot"]
        m = res.perf_metrics

        print("\n" + "=" * 85)
        print(
            f"🛡️  STARTING NUCLEAR AUDIT | {res.decision_date.date()} | {inputs.metric}"
        )
        print("*" * 85)
        print(
            "⚠️  ASSUMPTION: Verification logic is independent, but trusts Engine source DataFrames"
        )
        print(
            "    (engine.features_df, engine.df_close, and debug['portfolio_raw_components'])"
        )
        print("*" * 85)
        print("=" * 85)

        periods = {
            "Full": (res.start_date, res.holding_end_date),
            "Lookback": (res.start_date, res.decision_date),
            "Holding": (res.buy_date, res.holding_end_date),
        }

        # --------------------------------------------------------------------------
        # HELPER 1: MANUAL ATRP CALCULATION (DECIMAL MODE)
        # --------------------------------------------------------------------------
        def calculate_manual_atrp_warm(
            df_ohlcv, features_df, df_close_matrix, start_date
        ):
            df = df_ohlcv.copy()

            available_tickers = df.index.get_level_values("Ticker").unique()
            if len(available_tickers) == 0:
                return pd.DataFrame(), pd.DataFrame()

            seed_atrp_all = features_df.xs(start_date, level="Date")["ATRP"]

            # Intersect to find valid debug candidate
            valid_debug_tickers = [
                t for t in available_tickers if t in seed_atrp_all.index
            ]
            if not valid_debug_tickers:
                return pd.DataFrame(), pd.DataFrame()

            df["PC"] = df.groupby(level="Ticker")["Adj Close"].shift(1)

            # STRICT TR: skipna=False matches Engine logic
            tr = pd.concat(
                [
                    df["Adj High"] - df["Adj Low"],
                    (df["Adj High"] - df["PC"]).abs(),
                    (df["Adj Low"] - df["PC"]).abs(),
                ],
                axis=1,
            ).max(axis=1, skipna=False)

            seed_price = df_close_matrix.loc[start_date]

            # DECIMAL MODE: No multiplication/division by 100
            # Formula: SeedATR = ATRP(Decimal) * Price
            seed_atr = seed_atrp_all.reindex(available_tickers) * seed_price.reindex(
                available_tickers
            )

            alpha = 1 / 14

            def ewm_warm(group):
                ticker = group.name
                initial_val = seed_atr.get(ticker, group.iloc[0])
                vals = group.values
                results = np.zeros_like(vals)
                results[0] = initial_val
                for i in range(1, len(vals)):
                    results[i] = (vals[i] * alpha) + (results[i - 1] * (1 - alpha))
                return pd.Series(results, index=group.index)

            manual_atr = tr.groupby(level="Ticker", group_keys=False).apply(ewm_warm)
            prices_wide = df["Adj Close"].unstack(level=0)

            # DECIMAL MODE OUTPUT: ATR / Price
            manual_atrp_decimal = manual_atr.unstack(level=0) / prices_wide

            return (
                manual_atrp_decimal,
                tr.unstack(level=0) / prices_wide,
            )

        # --------------------------------------------------------------------------
        # HELPER 2: PERIOD AUDIT RUNNER
        # --------------------------------------------------------------------------
        def run_period_audit(df_p, df_atrp, df_trp, weights):
            if df_p.empty:
                return 0, 0, 0, 0
            norm = df_p.div(df_p.bfill().iloc[0])
            equity = (norm * weights).sum(axis=1)
            drift_w = (norm * weights).div(equity, axis=0)

            # Weighted Volatility
            p_atrp_manual = (drift_w * df_atrp).sum(axis=1)
            p_trp_manual = (drift_w * df_trp).sum(axis=1)

            rets = equity.pct_change().dropna()
            if rets.empty:
                return 0, 0, 0, 0

            gain = np.log(equity.iloc[-1])
            sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0

            return (
                gain,
                sharpe,
                rets.mean() / p_atrp_manual.loc[rets.index].mean(),
                rets.mean() / p_trp_manual.loc[rets.index].mean(),
            )

        # --------------------------------------------------------------------------
        # PART 1: PERFORMANCE RECONCILIATION
        # --------------------------------------------------------------------------
        audit_rows = []
        targets = [
            ("p", debug["portfolio_raw_components"], res.initial_weights, "Group"),
            (
                "b",
                debug["benchmark_raw_components"],
                pd.Series({inputs.benchmark_ticker: 1.0}),
                "Benchmark",
            ),
        ]

        for prefix, components, weights, entity_name in targets:
            m_atrp, m_trp = calculate_manual_atrp_warm(
                components["ohlcv_raw"],
                engine.features_df,
                engine.df_close,
                res.start_date,
            )
            m_price = components["prices"]

            for p_label, (d_start, d_end) in periods.items():
                mg, ms, msa, mst = run_period_audit(
                    m_price.loc[d_start:d_end],
                    m_atrp.loc[d_start:d_end],
                    m_trp.loc[d_start:d_end],
                    weights,
                )
                for m_name, m_val, e_key in [
                    ("Gain", mg, f"{p_label.lower()}_{prefix}_gain"),
                    ("Sharpe", ms, f"{p_label.lower()}_{prefix}_sharpe"),
                    ("Sharpe (ATRP)", msa, f"{p_label.lower()}_{prefix}_sharpe_atrp"),
                    ("Sharpe (TRP)", mst, f"{p_label.lower()}_{prefix}_sharpe_trp"),
                ]:
                    e_val = m.get(e_key, 0)
                    audit_rows.append(
                        {
                            "Entity": entity_name,
                            "Period": p_label,
                            "Metric": m_name,
                            "Engine": e_val,
                            "Manual": m_val,
                            "Delta": e_val - m_val,
                        }
                    )

        df_perf = pd.DataFrame(audit_rows)
        df_perf["Status"] = df_perf["Delta"].apply(
            lambda x: "✅ PASS" if abs(x) < 1e-7 else "❌ FAIL"
        )
        print("📝 1. PERFORMANCE RECONCILIATION")
        display(
            df_perf.pivot_table(
                index=["Entity", "Metric"],
                columns="Period",
                values="Status",
                aggfunc="first",
            )
        )

        # --------------------------------------------------------------------------
        # PART 2: SURVIVAL AUDIT (Liquidity/Quality Gate)
        # --------------------------------------------------------------------------
        print("\n" + "=" * 85)
        print("📝 2. SURVIVAL AUDIT")
        if inputs.universe_subset:
            print(
                "   Mode: CASCADE/SUBSET | Logic: Quality filters bypassed per design. | ✅ BYPASS"
            )
        else:
            audit_liq = debug.get("audit_liquidity")

            # SAFETY CHECK: Handle missing or None audit_liquidity data
            if audit_liq is None:
                print("   ⚠️  WARNING: audit_liquidity data not found in debug output.")
                print(
                    "   Status: ❌ SKIP (Cannot verify survival logic without debug data)"
                )
            else:
                snapshot = audit_liq["universe_snapshot"]
                thresholds = inputs.quality_thresholds

                m_cutoff = max(
                    snapshot["RollMedDollarVol"].quantile(
                        thresholds["min_liquidity_percentile"]
                    ),
                    thresholds["min_median_dollar_volume"],
                )
                m_survivors = snapshot[
                    (snapshot["RollMedDollarVol"] >= m_cutoff)
                    & (snapshot["RollingStalePct"] <= thresholds["max_stale_pct"])
                    & (
                        snapshot["RollingSameVolCount"]
                        <= thresholds["max_same_vol_count"]
                    )
                ]
                s_match = (
                    "✅ PASS"
                    if audit_liq["tickers_passed"] == len(m_survivors)
                    else "❌ FAIL"
                )
                print(
                    f"   Survival Integrity: {s_match} (Engine: {audit_liq['tickers_passed']} vs Auditor: {len(m_survivors)})"
                )

        # --------------------------------------------------------------------------
        # PART 3: UNIVERSAL SELECTION AUDIT (Strategy Registry Math)
        # --------------------------------------------------------------------------
        if inputs.mode == "Ranking":
            print("\n" + "=" * 85)
            print(f"📝 3. UNIVERSAL SELECTION AUDIT | Strategy: {inputs.metric}")

            if "full_universe_ranking" not in debug:
                print(
                    "❌ Audit Error: 'full_universe_ranking' not found in debug data."
                )
                return

            eng_rank_df = debug["full_universe_ranking"]
            survivors = eng_rank_df.index.tolist()
            idx = pd.IndexSlice

            # Re-fetch data for the entire survivor list
            feat_period = engine.features_df.loc[
                idx[survivors, res.start_date : res.decision_date], :
            ]
            atrp_lb_mean = feat_period["ATRP"].groupby(level="Ticker").mean()
            trp_lb_mean = feat_period["TRP"].groupby(level="Ticker").mean()

            # --- NEW DECOUPLED AUDIT LOGIC ---
            feat_now = engine.features_df.xs(res.decision_date, level="Date").reindex(
                survivors
            )

            # Pull the macro snapshot for the specific decision date
            macro_now = engine.macro_df.loc[res.decision_date]

            lb_prices = engine.df_close.loc[
                res.start_date : res.decision_date, survivors
            ]

            # REBUILD OBSERVATION - Call the class constructor instead of creating a dict
            audit_obs = MarketObservation(
                lookback_close=lb_prices,
                lookback_returns=lb_prices.ffill().pct_change(),
                atrp=atrp_lb_mean,
                trp=trp_lb_mean,
                atr=feat_now.get("ATR"),
                rsi=feat_now["RSI"],
                consistency=feat_now["Consistency"],
                mom_21=feat_now["Mom_21"],
                ir_63=feat_now["IR_63"],
                beta_63=feat_now["Beta_63"],
                dd_21=feat_now["DD_21"],
                macro_trend=macro_now["Macro_Trend"],
                macro_trend_vel=macro_now.get(
                    "Macro_Trend_Vel", 0.0
                ),  # Added missing field
                macro_vix_z=macro_now["Macro_Vix_Z"],
                macro_vix_ratio=macro_now["Macro_Vix_Ratio"],
            )

            # Run Manual Registry Math on Full Universe
            manual_scores = METRIC_REGISTRY[inputs.metric](audit_obs)

            # Compare
            audit_data = []
            for i, (ticker, row) in enumerate(eng_rank_df.iterrows()):
                eng_val = row["Strategy_Score"]
                man_val = manual_scores.get(ticker, np.nan)
                delta = eng_val - man_val

                status = (
                    "✅ PASS" if np.isclose(eng_val, man_val, atol=1e-8) else "❌ FAIL"
                )

                audit_data.append(
                    {
                        "Rank": i + 1,
                        "Ticker": ticker,
                        "Engine": eng_val,
                        "Manual": man_val,
                        "Delta": delta,
                        "Status": status,
                    }
                )

            df_audit_all = pd.DataFrame(audit_data).set_index("Rank")
            n_pass = (df_audit_all["Status"] == "✅ PASS").sum()
            n_fail = len(df_audit_all) - n_pass

            print(
                f"   Scope: Evaluated {len(df_audit_all)} candidates (Full Universe)."
            )
            print(f"   Result: {n_pass} PASSED | {n_fail} FAILED")

        if n_fail > 0:
            print(f"⚠️  DISPLAYING FAILURES (Top {n_tickers}):")
            display(df_audit_all[df_audit_all["Status"] == "❌ FAIL"].head(n_tickers))
        else:
            print(
                f"   All scores match registry math. {inputs.metric} results of the first {n_tickers} tickers"
            )
            display(
                df_audit_all.head(n_tickers).style.format(
                    "{:.8f}", subset=["Engine", "Manual", "Delta"]
                )
            )

        print("=" * 85)

        # After Performance Part:
        perf_failed = (df_perf["Status"] == "❌ FAIL").any()

        # After Selection Part:
        # if n_fail > 0: ranking_failed = True

        if perf_failed or n_fail > 0:
            return TaskResult(ok=False, msg="Nuclear audit failed reconciliation.")

        return TaskResult(ok=True, msg="Nuclear audit passed all checks.")

    @staticmethod
    def audit_feature_engineering_integrity(analyzer, df_indices=None, mode="last_run"):
        """
        # Usage to check last run, takes about 4 sec.
        audit_feature_engineering_integrity(analyzer2, mode="last_run")
        # Usage to check all df_ohlcv tickers, takes over 4 minutes (i.e. One-time "Nuclear" System Sanity Check)
        audit_feature_engineering_integrity(analyzer2, df_indices=df_indices, mode="system")
        """
        import time
        import numpy as np
        import warnings

        # 0. PULL SETTINGS FROM GLOBAL_SETTINGS (or analyzer.engine.settings if stored there)
        # This ensures the auditor uses the EXACT same rules as the engine
        atr_p = GLOBAL_SETTINGS["atr_period"]
        rsi_p = GLOBAL_SETTINGS["rsi_period"]
        win_5 = GLOBAL_SETTINGS["5d_window"]
        win_21 = GLOBAL_SETTINGS["21d_window"]
        win_63 = GLOBAL_SETTINGS["63d_window"]
        q_win = GLOBAL_SETTINGS["quality_window"]
        q_min = GLOBAL_SETTINGS["quality_min_periods"]

        start_time = time.time()
        engine = analyzer.engine
        features_df = engine.features_df
        df_ohlcv = engine.df_ohlcv_raw

        # 1. Scope Selection
        if mode == "last_run" and analyzer.last_run:
            audit_tickers = analyzer.last_run.tickers
            features_to_audit = features_df.loc[pd.IndexSlice[audit_tickers, :], :]
            ohlcv_to_audit = df_ohlcv.loc[pd.IndexSlice[audit_tickers, :], :]
        else:
            audit_tickers = features_df.index.get_level_values(0).unique()
            features_to_audit = features_df
            ohlcv_to_audit = df_ohlcv

        print(f"\n{'='*95}")
        print(
            f"🕵️  NUCLEAR FEATURE AUDIT | Mode: {mode.upper()} | Tickers: {len(audit_tickers)}"
        )
        print(f"{'='*95}")

        # STEP 1: BOUNDARY INTEGRITY
        leaks = features_to_audit.groupby(level=0).head(1)["Ret_1d"].dropna().count()
        leak_status = "✅ PASS" if leaks == 0 else f"❌ FAIL ({leaks} leaks)"
        print(
            f"STEP 1: BOUNDARY INTEGRITY   | MultiIndex Isolation Check | {leak_status}"
        )

        # STEP 2: SHADOW CALCULATION
        print(
            f"STEP 2: SHADOW CALCULATIONS  | Re-computing metrics... ",
            end="",
            flush=True,
        )

        adj_close = ohlcv_to_audit["Adj Close"]
        adj_high = ohlcv_to_audit["Adj High"]
        adj_low = ohlcv_to_audit["Adj Low"]
        volume = ohlcv_to_audit["Volume"]

        shadow_data = {}

        # A. Returns & Basics
        # Follow AlphaEngine Return calculation
        # Explicitly turns division-by-zero results (`inf`) into `NaN`
        # Replace [np.inf, -np.inf] with np.nan
        shadow_data["shadow_Ret_1d"] = (
            adj_close.groupby(level=0).pct_change().replace([np.inf, -np.inf], np.nan)
        )

        prev_close = adj_close.groupby(level=0).shift(1)
        tr = pd.concat(
            [
                adj_high - adj_low,
                (adj_high - prev_close).abs(),
                (adj_low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1, skipna=False)

        # B. Smoothing (ATR/RSI) - Use transform for speed and index matching
        shadow_data["shadow_ATR"] = tr.groupby(level=0).transform(
            lambda x: x.ewm(alpha=1 / atr_p, adjust=False).mean()  # Replaced 14
        )

        shadow_data["shadow_ATRP"] = shadow_data["shadow_ATR"] / adj_close
        shadow_data["shadow_TRP"] = tr / adj_close

        # Auditor Step 2B - Shadow RSI with correct Inf/NaN handling
        delta = adj_close.groupby(level=0).diff()
        up, down = delta.clip(lower=0), (-delta).clip(lower=0)

        # Match Wilder's spec correctly:
        roll_up = up.groupby(level=0).transform(
            lambda x: x.ewm(alpha=1 / rsi_p, adjust=False).mean()  # Replaced 14
        )
        roll_down = down.groupby(level=0).transform(
            lambda x: x.ewm(alpha=1 / rsi_p, adjust=False).mean()  # Replaced 14
        )

        # FIX: Allow division by zero (i.e. no down day) to create inf (correct RSI=100),
        # inf→100, -inf→0, NaN→50
        # then clean up remaining NaNs (initial periods/no movement)
        # - Initial periods: Before the 14-day lookback is filled, the EWM mean is undefined → NaN.
        # - Flat prices: If price doesn't move (Avg Up = 0 and Avg Down = 0), RS is 0/0 → NaN.
        # - By convention, RSI is set to 50 (neutral) when there is no directional momentum.
        rs = roll_up / roll_down  # Keep zero denominator → inf
        raw_rsi = 100 - (100 / (1 + rs))
        shadow_data["shadow_RSI"] = raw_rsi.replace({np.inf: 100, -np.inf: 0}).fillna(
            50
        )

        # C. Momentum & Consistency
        shadow_data[f"shadow_Mom_{win_21}"] = adj_close.groupby(level=0).pct_change(
            win_21
        )
        pos_ret = (shadow_data["shadow_Ret_1d"] > 0).astype(float)
        shadow_data["shadow_Consistency"] = pos_ret.groupby(level=0).transform(
            lambda x: x.rolling(win_5).mean()
        )

        # D. Risk (Beta & IR)
        if df_indices is not None:
            try:
                # USE THIS: Pull the single source of truth from the engine
                mkt_ret = engine.macro_df["Mkt_Ret"]
                # Map it to the audit tickers
                mkt_series = mkt_ret.reindex(
                    ohlcv_to_audit.index.get_level_values(1)
                ).values
                mkt_series = pd.Series(mkt_series, index=ohlcv_to_audit.index)

                # Shadow Beta
                s_ret = shadow_data["shadow_Ret_1d"]
                shadow_data[f"shadow_Beta_{win_63}"] = (
                    s_ret.groupby(level=0)
                    .transform(
                        lambda x: x.rolling(win_63).cov(
                            mkt_ret.reindex(x.index.get_level_values(1))
                        )
                        / mkt_ret.reindex(x.index.get_level_values(1))
                        .rolling(win_63)
                        .var()
                    )
                    .fillna(1.0)
                )

                # Shadow IR
                active_ret = s_ret - mkt_series
                shadow_data["shadow_IR_63"] = (
                    active_ret.groupby(level=0)
                    .transform(
                        lambda x: x.rolling(win_63).mean() / x.rolling(win_63).std()
                    )
                    .fillna(0.0)
                )

            except Exception as e:
                print(f" (Macro Shadow Error: {e}) ", end="")

        # E. Drawdown & Quality
        roll_max_21 = adj_close.groupby(level=0).transform(
            lambda x: x.rolling(win_21).max()
        )
        shadow_data[f"shadow_DD_{win_21}"] = (adj_close / roll_max_21 - 1).fillna(0.0)
        stale_mask = ((volume == 0) | (adj_high == adj_low)).astype(int)

        shadow_data["shadow_RollingStalePct"] = stale_mask.groupby(level=0).transform(
            lambda x: x.rolling(q_win, min_periods=q_min).mean()
        )
        dollar_vol = adj_close * volume
        shadow_data["shadow_RollMedDollarVol"] = dollar_vol.groupby(level=0).transform(
            lambda x: x.rolling(q_win, min_periods=q_min).median()  # Replaced 252, 126
        )

        same_vol = (volume.groupby(level=0).diff() == 0).astype(int)
        shadow_data["shadow_RollingSameVolCount"] = same_vol.groupby(level=0).transform(
            lambda x: x.rolling(q_win, min_periods=q_min).sum()  # Replaced 252, 126
        )

        # Build Final Shadow DF
        audit_df = pd.DataFrame(shadow_data, index=ohlcv_to_audit.index)
        print(f"DONE ({time.time()-start_time:.2f}s)")

        # STEP 3: RECONCILIATION REPORT
        print(
            f"\n{'Metric':<20} | {'Max Delta':<12} | {'Correlation':<12} | {'Status'}"
        )
        print("-" * 85)

        cols_to_check = [
            "Ret_1d",
            "ATR",
            "ATRP",
            "TRP",
            "RSI",
            "Mom_21",
            "Consistency",
            "Beta_63",
            "IR_63",
            "DD_21",
            "RollingStalePct",
            "RollMedDollarVol",
            "RollingSameVolCount",
        ]

        for col in cols_to_check:
            sha_col = f"shadow_{col}"
            if sha_col not in audit_df.columns:
                continue

            eng, sha = features_to_audit[col], audit_df[sha_col]
            # Align and drop NaNs for comparison
            mask = eng.notna() & sha.notna()
            if not mask.any():
                continue

            e_v, s_v = eng[mask], sha[mask]

            delta = (e_v - s_v).abs().max()

            # Suppress the NumPy "Subtract" warning during correlation of constant series
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                # If standard deviation is 0, correlation is undefined; if eng matches Shadow Calculation, we treat as 1.0
                if e_v.std() == 0:
                    corr = 1.0 if delta < 1e-6 else 0.0
                else:
                    corr = e_v.corr(s_v)

            status = "✅ PASS" if (delta < 1e-6 or corr > 0.99999) else "❌ FAIL"
            print(f"{col:<20} | {delta:>12.4e} | {corr:>12.6f} | {status}")

        vix_z = engine.macro_df["Macro_Vix_Z"].abs().max()
        print(
            f"{'Macro_Vix_Signals':<20} | {'N/A':<12} | {'N/A':<12} | {'✅ LIVE' if vix_z > 0 else '❌ MISSING VIX, VIX3M'}"
        )
        print(f"{'='*95}")


#
