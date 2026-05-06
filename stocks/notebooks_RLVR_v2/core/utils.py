import os
import pandas as pd
import numpy as np

from pathlib import Path
from dotenv import load_dotenv
from dataclasses import asdict, is_dataclass, fields
from typing import Any, List, Dict, Union
from core.paths import OUTPUT_DIR
from core.kernel import QuantUtils


class SystemUtils:
    @staticmethod
    def peek(idx: int, reg: List[Dict[str, Any]]) -> Any:
        """
        Displays metadata and RETURNS the object for further use.
        SAFEGUARD: Checks if reg is actually a list from the visualizer.
        """
        if not isinstance(reg, list):
            print(f"❌ Error: Pass the result map (list), not the analyzer object.")
            return None

        if idx < 0 or idx >= len(reg):
            print(f"❌ Index {idx} out of range (0 to {len(reg)-1}).")
            return None

        entry = reg[idx]

        print(f" {'='*60}")
        print(f" 📍 INDEX: [{idx}]")
        print(f" 🏷️  NAME:  {entry.get('name', 'N/A')}")
        print(f" 📂 PATH:  {entry.get('path', 'N/A')}")
        print(f" {'='*60}\n")

        try:
            from IPython.display import display

            display(entry["obj"])
        except ImportError:
            print(entry["obj"])

        return entry["obj"]

    @staticmethod
    def visualize_analyzer_structure(analyzer) -> List[Dict]:
        """
        High-level entry point for the Analyzer.
        Maps the internal data structure of the last simulation run.
        """
        # Check if last_run exists (WalkForwardAnalyzer specific logic)
        last_run = getattr(analyzer, "last_run", None)

        if not last_run:
            print("❌ Audit Aborted: No simulation data found in analyzer.last_run.")
            return []

        return SystemUtils.visualize_audit_structure(last_run)

    @staticmethod
    def visualize_audit_structure(obj) -> List[Dict]:
        """
        CORE ENGINE: Generates the Map and returns a Registry.
        """
        id_memory = {}
        registry = []
        output = [
            "====================================================================",
            "🔍 HIGH-TRANSPARENCY AUDIT MAP",
            "====================================================================",
        ]

        def get_icon(val):
            if isinstance(val, pd.DataFrame):
                return "🧮"
            if isinstance(val, pd.Series):
                return "📈"
            if isinstance(val, (list, tuple, dict)):
                return "📂"
            if isinstance(val, pd.Timestamp):
                return "📅"
            if is_dataclass(val):
                return "📦"
            return "🔢" if isinstance(val, (int, float)) else "📄"

        def process(item, name, level=0, path=""):
            indent = "  " * level
            item_id = id(item)
            current_path = f"{path} -> {name}" if path else name

            is_primitive = isinstance(item, (int, float, str, bool, type(None)))

            # Avoid infinite recursion and handle shared references
            if not is_primitive and item_id in id_memory:
                output.append(
                    f"{indent}          ╰── {name} --> [See ID {id_memory[item_id]}]"
                )
                return

            curr_idx = len(registry)
            registry.append({"name": name, "path": current_path, "obj": item})

            if not is_primitive:
                id_memory[item_id] = curr_idx

            # Generate Metadata String
            meta = f"{type(item).__name__}"
            if hasattr(item, "shape"):
                meta = f"shape={item.shape}"
            elif isinstance(item, (list, dict)):
                meta = f"len={len(item)}"

            output.append(f"[{curr_idx:>3}] {indent}{get_icon(item)} {name} ({meta})")

            # Recursion Logic
            if isinstance(item, dict):
                for k, v in item.items():
                    process(v, k, level + 1, current_path)
            elif isinstance(item, (list, tuple)):
                for i, v in enumerate(item):
                    process(v, f"index_{i}", level + 1, current_path)
            elif is_dataclass(item):
                for f in fields(item):
                    process(getattr(item, f.name), f.name, level + 1, current_path)

        process(obj, "audit_pack")
        print("\n".join(output))

        return registry

    @staticmethod
    def export_audit_to_excel(audit_pack, filename="Audit_Verification_Report.xlsx"):
        """
        Final Zero-Base Audit Export.
        Provides everything needed to reconstruct the Strategy results from raw candles.
        Usage: SystemUtils.export_audit_to_excel(audit_pack=analyzer.last_run, filename=f_name_excel)
        """
        if audit_pack is None:
            return print("❌ Error: Audit Pack is empty.")

        # Resolve full output path
        output_path = Path(OUTPUT_DIR) / filename

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Setup Core References
        debug = audit_pack.debug_data or {}
        inputs = debug.get("inputs_snapshot")
        p_raw = debug.get("portfolio_raw_components", {})
        b_raw = debug.get("benchmark_raw_components", {})

        dec_date = audit_pack.decision_date
        bench_ticker = inputs.benchmark_ticker if inputs else "Benchmark"
        all_tickers = audit_pack.tickers if audit_pack.tickers else []

        print(f"📂 [EXCEL AUDIT] Building full transparency report: {output_path}")

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:

            # --- SHEET 1: OVERVIEW (Settings & Results) ---
            meta = {**asdict(inputs)} if inputs else {}
            meta.update(audit_pack.perf_metrics or {})
            pd.DataFrame.from_dict(
                {k: str(v) for k, v in meta.items()}, orient="index", columns=["Value"]
            ).to_excel(writer, sheet_name="OVERVIEW")

            # --- SHEET 2: DAILY_AUDIT (The Timeline + Period Labels) ---
            daily = {
                "Port_Value": audit_pack.portfolio_series,
                "Port_ATRP": audit_pack.portfolio_atrp_series,
                "Port_TRP": audit_pack.portfolio_trp_series,
                "Bench_Value": audit_pack.benchmark_series,
                "Bench_ATRP": audit_pack.benchmark_atrp_series,
                "Bench_TRP": audit_pack.benchmark_trp_series,
            }
            if audit_pack.portfolio_series is not None:
                daily["Port_Ret"] = QuantUtils.compute_returns(
                    audit_pack.portfolio_series
                )
            if audit_pack.benchmark_series is not None:
                daily["Bench_Ret"] = QuantUtils.compute_returns(
                    audit_pack.benchmark_series
                )

            df_daily = pd.concat(
                {k: v for k, v in daily.items() if v is not None}, axis=1
            )

            # Add Period Label Column for Excel Range Selection
            df_daily["Period_Label"] = np.where(
                df_daily.index <= dec_date, "LOOKBACK", "HOLDING"
            )
            df_daily.to_excel(writer, sheet_name="DAILY_AUDIT")

            # --- SHEET 3: RAW_OHLCV_SAMPLES (Spot Check for ALL Tickers + Benchmark) ---
            ohlcv_list = []
            # Get Benchmark OHLCV
            if (b_ohlcv := b_raw.get("ohlcv_raw")) is not None:
                b_temp = b_ohlcv.copy()
                b_temp["Ticker"] = bench_ticker
                ohlcv_list.append(b_temp)
            # Get ALL Tickers OHLCV (not just top 3)
            if (p_ohlcv := p_raw.get("ohlcv_raw")) is not None:
                if isinstance(p_ohlcv.index, pd.MultiIndex):
                    sample_p = p_ohlcv.query("Ticker in @all_tickers")
                    ohlcv_list.append(sample_p.reset_index())
                else:
                    # Fallback: Filter by 'ticker' column if it exists
                    col_name = "ticker" if "ticker" in p_ohlcv.columns else "Ticker"
                    if col_name in p_ohlcv.columns:
                        ohlcv_list.append(p_ohlcv[p_ohlcv[col_name].isin(all_tickers)])

            if ohlcv_list:
                pd.concat(ohlcv_list).to_excel(
                    writer, sheet_name="RAW_OHLCV_SAMPLES", index=False
                )

            # --- SHEET 4, 5, 6: MERGED MATRICES (Price, ATRP, TRP) ---
            for sheet_name, key in [
                ("RAW_PRICES", "prices"),
                ("RAW_ATRP_DATA", "atrp"),
                ("RAW_TRP_DATA", "trp"),
            ]:
                p_df, b_df = p_raw.get(key), b_raw.get(key)
                if p_df is not None and b_df is not None:
                    pd.concat(
                        [p_df, b_df.rename(columns={b_df.columns[0]: bench_ticker})],
                        axis=1,
                    ).to_excel(writer, sheet_name=sheet_name)

            # --- SHEET 7: RAW_DRIFTED_WEIGHTS ---
            if (p_prices := p_raw.get("prices")) is not None:
                weights_ser = pd.Series(
                    audit_pack.initial_weights, index=audit_pack.tickers
                )
                norm_p = p_prices.div(p_prices.bfill().iloc[0])
                weighted = norm_p.mul(weights_ser, axis=1)
                drift_weights = weighted.div(weighted.sum(axis=1), axis=0)
                drift_weights.to_excel(writer, sheet_name="RAW_DRIFTED_WEIGHTS")

            # --- SHEET 8: SURVIVAL_AUDIT (Layer 1 Filter Verification) ---
            if liq_audit := debug.get("audit_liquidity", {}):
                if (snap := liq_audit.get("universe_snapshot")) is not None:
                    snap.to_excel(writer, sheet_name="SURVIVAL_AUDIT")

            # --- SHEET 9: FULL_RANKING ---
            if (df_rank := debug.get("full_universe_ranking")) is not None:
                df_rank.to_excel(writer, sheet_name="FULL_RANKING")

        print(f"✨ Audit Report Complete: {output_path}")
        return output_path

    @staticmethod
    def export_last_run_tickers_data_to_csv(
        analyzer, df_ohlcv, features_df, filename="all_tickers_stacked.csv"
    ):
        """
        Export the last run ticker data from a WalkForwardAnalyzer to a stacked CSV file.
        """

        # 1. Access the result object from the analyzer
        res = analyzer.last_run

        if res is None:
            raise ValueError(
                "❌ No results found in analyzer. Please click 'Run Simulation' first."
            )

        # 2. Extract attributes directly
        benchmark = res.debug_data["inputs_snapshot"].benchmark_ticker
        tickers = res.tickers + [benchmark]
        start_date = res.start_date
        end_date = res.holding_end_date

        # 3. Generate the combined dict
        combined = SystemUtils.create_combined_dict(
            df_ohlcv=df_ohlcv.copy(),
            features_df=features_df,
            tickers=tickers,
            date_start=start_date,
            date_end=end_date,
            verbose=True,
        )

        # 4. Save ticker data to CSV
        file_path = filename

        # Save first ticker with header
        first_ticker = tickers[0]
        df_first = combined[first_ticker][start_date:end_date].reset_index()
        df_first["Ticker"] = first_ticker

        df_first.to_csv(file_path, header=True, index=False, lineterminator="\n")
        print(f"✓ Saved {first_ticker} with header")

        # Append remaining tickers without header
        for ticker in tickers[1:]:
            df = combined[ticker][start_date:end_date].reset_index()
            df["Ticker"] = ticker

            df.to_csv(
                file_path, header=False, index=False, lineterminator="\n", mode="a"
            )
            print(f"✓ Appended {ticker}")

        print(f"\n✓ Saved all tickers to: {file_path}")

        return file_path

    @staticmethod
    def get_ticker_OHLCV(
        df_ohlcv: pd.DataFrame,
        tickers: Union[str, List[str]],
        date_start: str,
        date_end: str,
        return_format: str = "dataframe",
        verbose: bool = True,
    ) -> Union[pd.DataFrame, dict]:
        """
        Get OHLCV data for specified tickers within a date range.
        """

        # Input validation
        if not isinstance(df_ohlcv, pd.DataFrame):
            raise TypeError("df_ohlcv must be a pandas DataFrame")

        if not isinstance(df_ohlcv.index, pd.MultiIndex):
            raise ValueError("DataFrame must have MultiIndex of (ticker, date)")

        # Convert single ticker to list for consistent processing
        if isinstance(tickers, str):
            tickers = [tickers]
        elif not isinstance(tickers, list):
            raise TypeError("tickers must be a string or list of strings")

        # Convert dates to Timestamps
        try:
            start_date = pd.Timestamp(date_start)
            end_date = pd.Timestamp(date_end)
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use 'YYYY-MM-DD': {e}")

        # Filter the data using MultiIndex slicing
        try:
            filtered_data = df_ohlcv.loc[(tickers, slice(date_start, date_end)), :]
        except Exception as e:
            raise ValueError(f"Error filtering data: {e}")

        if filtered_data.empty:
            return filtered_data

        # Return in requested format
        if return_format == "dict":
            result = {}
            for ticker in tickers:
                try:
                    result[ticker] = filtered_data.xs(ticker, level=0).loc[
                        date_start:date_end
                    ]
                except KeyError:
                    result[ticker] = pd.DataFrame()
            return result

        elif return_format == "separate":
            result = []
            for ticker in tickers:
                try:
                    result.append(
                        filtered_data.xs(ticker, level=0).loc[date_start:date_end]
                    )
                except KeyError:
                    result.append(pd.DataFrame())
            return result

        elif return_format == "dataframe":
            return filtered_data

        return filtered_data

    @staticmethod
    def get_ticker_features(
        features_df: pd.DataFrame,
        tickers: Union[str, List[str]],
        date_start: str,
        date_end: str,
        return_format: str = "dataframe",
        verbose: bool = True,
    ) -> Union[pd.DataFrame, dict]:
        """
        Get features data for specified tickers within a date range.
        """
        if isinstance(tickers, str):
            tickers = [tickers]

        try:
            filtered_data = features_df.loc[(tickers, slice(date_start, date_end)), :]
        except Exception:
            return pd.DataFrame() if return_format == "dataframe" else {}

        if filtered_data.empty:
            return filtered_data

        if return_format == "dict":
            result = {}
            for ticker in tickers:
                try:
                    result[ticker] = filtered_data.xs(ticker, level=0).loc[
                        date_start:date_end
                    ]
                except KeyError:
                    result[ticker] = pd.DataFrame()
            return result

        return filtered_data

    @staticmethod
    def create_combined_dict(
        df_ohlcv: pd.DataFrame,
        features_df: pd.DataFrame,
        tickers: Union[str, List[str]],
        date_start: str,
        date_end: str,
        verbose: bool = True,
    ) -> dict:
        """
        Create a combined dictionary with both OHLCV and features data for each ticker.
        """
        if isinstance(tickers, str):
            tickers = [tickers]

        ohlcv_dict = SystemUtils.get_ticker_OHLCV(
            df_ohlcv, tickers, date_start, date_end, return_format="dict", verbose=verbose
        )

        features_dict = SystemUtils.get_ticker_features(
            features_df, tickers, date_start, date_end, return_format="dict", verbose=verbose
        )

        combined_dict = {}

        for ticker in tickers:
            if ticker in ohlcv_dict and ticker in features_dict:
                ohlcv_data = ohlcv_dict[ticker]
                features_data = features_dict[ticker]

                if not ohlcv_data.empty and not features_data.empty:
                    combined_df = pd.concat([ohlcv_data, features_data], axis=1)
                    combined_df.index.name = "Date"
                    combined_dict[ticker] = combined_df
                else:
                    combined_dict[ticker] = pd.DataFrame()
            else:
                combined_dict[ticker] = pd.DataFrame()

        return combined_dict

    @staticmethod
    def load_env_from_root(env_var_name):
        """
        Load specified environment variable from .env file in root directory
        """
        try:
            start_path = Path(__file__).resolve().parent
        except NameError:
            start_path = Path.cwd()

        current = start_path
        for _ in range(10):  # Limit search depth
            env_path = current / ".env" / "my_api_key.env"
            if env_path.exists():
                load_dotenv(env_path, override=True)
                value = os.getenv(env_var_name)
                if value is None:
                    raise KeyError(f"Variable '{env_var_name}' not found in {env_path}")
                return value

            parent = current.parent
            if parent == current:
                break
            current = parent

        raise FileNotFoundError(
            f"Could not find .env/my_api_key.env when searching for '{env_var_name}'"
        )
