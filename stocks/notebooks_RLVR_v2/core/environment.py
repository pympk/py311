import os
import glob
import multiprocessing
import gc
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from functools import partial
from typing import List

from core.kernel import QuantUtils
from core.config import GLOBAL_SETTINGS

class AlphaLogic:
    """STATELESS: The mathematical engine for rewards and ensemble generation."""

    @staticmethod
    def calculate_veritable_reward(
        reward_matrix: pd.DataFrame, date: pd.Timestamp, tickers: List[str]
    ) -> float:
        """The Log-Return truth engine. LN(1 + Arithmetic_Mean)."""
        if not tickers or date not in reward_matrix.index:
            return 0.0

        # Arithmetic Mean of the group
        arith_mean = reward_matrix.loc[date, tickers].mean()
        # Transform to Log for the Agent's additive math
        return float(np.log1p(arith_mean))

    @staticmethod
    def slugify_columns(columns: List[str]) -> List[str]:
        """Ensures names are machine-safe: 21d_Sharpe_(ATRP) -> 21d_Sharpe_ATRP"""
        return [
            c.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "")
            .replace(",", "")  # Added to handle "Alpha, 63d"
            .replace("__", "_")  # Clean up double underscores
            for c in columns
        ]


class SelectionLogic:
    """STATELESS: Decodes actions into ticker lists."""

    @staticmethod
    def apply_action(
        ensemble: pd.DataFrame, action: np.ndarray, rank_max_offset: int = 50
    ) -> List[str]:
        """Vectorized Matrix Multiplication + Sorting."""
        if ensemble.empty:
            return []

        # 1. Action Decoding
        n_features = ensemble.shape[1]
        weights = action[:n_features]

        # Map normalized [-1, 1] to discrete ranges
        offset = int(np.interp(action[-2], [-1, 1], [0, rank_max_offset]))
        width = int(np.interp(action[-1], [-1, 1], [1, 10]))  # Your Max-10 constraint

        # 2. Vectorized Scoring [Tickers x Features] @ [Features]
        scores = ensemble.values @ weights

        # 3. High-Speed Sort (Numpy is 10x faster than Pandas for this)
        top_indices = np.argsort(scores)[::-1][offset : offset + width]

        return ensemble.index[top_indices].tolist()


class ParallelFeatureBuilder:
    """
    ULTRA-PERFORMANCE BUILDER: Uses all CPU cores to bake the AlphaCache.
    Features: Parallelism, Atomic Checkpointing, Smart Recovery, and DEBUG MODE.
    """

    # Class-level flag to enable debug CSV output
    DEBUG_MODE = False
    DEBUG_DIR = "debug_zscore_check"

    @staticmethod
    def _worker_task(
        date, engine_params, lookbacks, debug_mode=False, debug_dir="debug_zscore_check"
    ):
        """The atomic unit of work performed by each CPU core."""
        try:
            master_engine = engine_params["engine"]

            # 1. Generate Raw Metrics
            raw_ensemble = master_engine.compute_alpha_ensemble(date, lookbacks)
            if raw_ensemble.empty:
                return None

            # --- FORENSIC CHECK 1: TICKER UNIQUENESS ---
            if raw_ensemble.index.duplicated().any():
                dupes = (
                    raw_ensemble.index[raw_ensemble.index.duplicated()]
                    .unique()
                    .tolist()
                )
                print(f"⚠️  DUPLICATE TICKERS detected on {date.date()}: {dupes}")
                # Force uniqueness
                raw_ensemble = raw_ensemble.loc[
                    ~raw_ensemble.index.duplicated(keep="first")
                ]

            # 2. Cleaning
            clean_ensemble = raw_ensemble.dropna()
            if clean_ensemble.empty:
                return None

            # ============ DEBUG EXPORT: BEFORE NORMALIZATION ============
            if debug_mode:
                if not os.path.exists(debug_dir):
                    os.makedirs(debug_dir)

                # Save raw values with metadata
                debug_raw = clean_ensemble.copy()
                debug_raw["__Date__"] = date
                debug_raw["__Stage__"] = "RAW"

                # Calculate but don't apply normalization stats
                means = clean_ensemble.mean()
                stds = clean_ensemble.std(ddof=1)
                counts = clean_ensemble.count()

                # Append stats as extra rows for Excel verification
                stats_df = pd.DataFrame(
                    {
                        col: [means[col], stds[col], counts[col]]
                        for col in clean_ensemble.columns
                    },
                    index=["__MEAN__", "__STD__", "__COUNT__"],
                )

                stats_df["__Date__"] = date
                stats_df["__Stage__"] = "STATS"
                stats_df["__Ticker__"] = ["__MEAN__", "__STD__", "__COUNT__"]

                # Combine raw data + stats
                debug_raw_indexed = debug_raw.reset_index()
                debug_raw_indexed.rename(columns={"index": "Ticker"}, inplace=True)

                stats_df = stats_df.reset_index().rename(columns={"index": "Ticker"})
                # Reorder columns to match
                cols = ["Ticker", "__Date__", "__Stage__"] + [
                    c
                    for c in debug_raw_indexed.columns
                    if c not in ["Ticker", "__Date__", "__Stage__"]
                ]
                debug_raw_indexed = debug_raw_indexed[cols]
                stats_df = stats_df[cols]

                combined_debug = pd.concat(
                    [debug_raw_indexed, stats_df], ignore_index=True
                )

                # Save to CSV
                debug_file = f"{debug_dir}/debug_{date.strftime('%Y%m%d')}.csv"
                combined_debug.to_csv(debug_file, index=False)
                print(f"💾 Debug CSV saved: {debug_file}")

            # 3. Normalization (ddof=1 for Excel Parity)
            normalized = (clean_ensemble - clean_ensemble.mean()) / clean_ensemble.std(
                ddof=1
            )

            # ============ DEBUG EXPORT: AFTER NORMALIZATION ============
            if debug_mode:
                debug_norm = normalized.copy()
                debug_norm["__Date__"] = date
                debug_norm["__Stage__"] = "ZSCORE"
                debug_norm = debug_norm.reset_index().rename(
                    columns={"index": "Ticker"}
                )

                # Reorder columns
                cols = ["Ticker", "__Date__", "__Stage__"] + [
                    c
                    for c in debug_norm.columns
                    if c not in ["Ticker", "__Date__", "__Stage__"]
                ]
                debug_norm = debug_norm[cols]

                # Append to same file (mode='a' for append)
                debug_file = f"{debug_dir}/debug_{date.strftime('%Y%m%d')}.csv"
                with open(debug_file, "a") as f:
                    f.write("\n")  # Separator
                debug_norm.to_csv(debug_file, mode="a", index=False, header=False)
                print(f"💾 Z-scores appended to: {debug_file}")

            # 4. Slugify names
            new_cols = AlphaLogic.slugify_columns(normalized.columns.tolist())

            # --- FORENSIC CHECK 2: COLUMN UNIQUENESS ---
            if len(new_cols) != len(set(new_cols)):
                from collections import Counter

                counts = Counter(new_cols)
                dupe_cols = [item for item, count in counts.items() if count > 1]
                print(f"⚠️  COLUMN NAME COLLISION on {date.date()}: {dupe_cols}")
                # We fix the names to be unique by appending the index
                new_cols = [
                    f"{c}_{i}" if counts[c] > 1 else c for i, c in enumerate(new_cols)
                ]

            normalized.columns = new_cols

            # 5. Prepare MultiIndex
            normalized.index.name = "Ticker"
            normalized["Date"] = date

            # Use 'append=True' to keep Ticker in index, then swap so Date is first
            res = normalized.set_index("Date", append=True).swaplevel(0, 1)

            # --- FORENSIC CHECK 3: FINAL INDEX INTEGRITY ---
            if not res.index.is_unique:
                print(
                    f"❌ FATAL: Index still not unique on {date.date()} after cleaning."
                )
                return None

            return res

        except Exception as e:
            print(f"❌ ERROR on {date.date()}: {str(e)}")
            return None

    @staticmethod
    def run_marathon(
        master_engine,
        lookbacks,
        start_date,
        checkpoint_dir="cache_checkpoints",
        batch_size=50,
        num_workers=None,
        debug_mode=False,
        debug_sample_dates=None,
    ):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        if num_workers is None:
            num_workers = max(1, multiprocessing.cpu_count() - 2)

        # 1. IDENTIFY GAPS (Smart Recovery)
        calendar = master_engine.trading_calendar
        all_dates = calendar[calendar >= pd.Timestamp(start_date)]
        processed_dates = ParallelFeatureBuilder._get_processed_dates(checkpoint_dir)

        # Filter for debug dates if specified
        if debug_sample_dates:
            debug_sample_dates = [pd.Timestamp(d) for d in debug_sample_dates]
            missing_dates = [d for d in debug_sample_dates if d not in processed_dates]
            print(f"🔍 DEBUG MODE: Processing only {len(missing_dates)} sample dates")
        else:
            missing_dates = [d for d in all_dates if d not in processed_dates]

        if not missing_dates:
            print("✅ All data baked and verified.")
            return

        print(
            f"🚀 Parallel Bake: {len(missing_dates)} days | Using {num_workers} of {multiprocessing.cpu_count()} cores."
        )

        if debug_mode:
            print(f"🐛 DEBUG CSV export enabled → {ParallelFeatureBuilder.DEBUG_DIR}/")

        # 2. CHUNK THE WORK
        date_chunks = [
            missing_dates[i : i + batch_size]
            for i in range(0, len(missing_dates), batch_size)
        ]

        engine_params = {"engine": master_engine}

        with tqdm(total=len(missing_dates), desc="Baking AlphaCache") as pbar:
            for chunk in date_chunks:
                with multiprocessing.Pool(processes=num_workers) as pool:
                    worker_func = partial(
                        ParallelFeatureBuilder._worker_task,
                        engine_params=engine_params,
                        lookbacks=lookbacks,
                        debug_mode=debug_mode,
                        debug_dir=ParallelFeatureBuilder.DEBUG_DIR,
                    )
                    results = pool.map(worker_func, chunk)

                # Filter out None/Errors and save batch
                valid_results = [r for r in results if isinstance(r, pd.DataFrame)]
                if valid_results:
                    batch_df = pd.concat(valid_results, sort=False)
                    batch_fn = (
                        f"{checkpoint_dir}/batch_{chunk[0].strftime('%Y%m%d')}.parquet"
                    )
                    batch_df.to_parquet(batch_fn, engine="pyarrow", compression="zstd")

                pbar.update(len(chunk))

        print(f"✨ Marathon complete. All batches saved to {checkpoint_dir}")

    @staticmethod
    def _get_processed_dates(checkpoint_dir: str) -> set:
        files = glob.glob(f"{checkpoint_dir}/*.parquet")
        processed = set()
        for f in files:
            df = pd.read_parquet(f, columns=[])
            processed.update(df.index.get_level_values("Date").unique())
        return processed


class FeatureCubeStitcher:
    """Combines thousands of fragments into one High-Speed Feature Cube."""

    @staticmethod
    def assemble(checkpoint_dir: str, final_output_fn: str):
        files = sorted(glob.glob(f"{checkpoint_dir}/*.parquet"))
        if not files:
            print("❌ No checkpoint files found.")
            return

        print(f"🧵 Stitching {len(files)} batches...")

        all_chunks = []
        for f in tqdm(files, desc="Merging"):
            chunk = pd.read_parquet(f)
            all_chunks.append(chunk[~chunk.index.duplicated(keep="first")])

        full_df = pd.concat(all_chunks).sort_index()
        del all_chunks
        gc.collect()

        full_df.to_parquet(final_output_fn, engine="pyarrow", compression="zstd")
        print(f"✅ Final Master Cube Saved! Shape: {full_df.shape}")
        return full_df


class DiscoveryEnv:
    """STATEFUL: The 'Arena' for the Agent."""

    def __init__(
        self,
        feature_cube: pd.DataFrame,
        reward_matrix: pd.DataFrame,
        calendar: pd.DatetimeIndex,
        holding_period: int = 5,
    ):
        self.cube = feature_cube
        self.reward_matrix = reward_matrix
        self.calendar = calendar
        self.holding_period = holding_period
        self.reset()

    def reset(self, start_date=None):
        if start_date:
            self.current_date_idx = self.calendar.get_loc(start_date)
        else:
            # Start far enough in so we have cache data
            self.current_date_idx = 252
        self.equity_curve = [1.0]
        return self._get_observation()

    def _get_observation(self):
        date = self.calendar[self.current_date_idx]
        try:
            ensemble = self.cube.xs(date, level="Date")
        except KeyError:
            ensemble = pd.DataFrame()

        return {"ensemble": ensemble, "date": date}

    def step(self, action: np.ndarray):
        date = self.calendar[self.current_date_idx]
        obs_dict = self._get_observation()
        ensemble = obs_dict["ensemble"]

        # 1. Delegate Ticker Selection (Stateless)
        selected_tickers = SelectionLogic.apply_action(ensemble, action)

        # 2. Delegate Reward Calculation (Stateless)
        reward = AlphaLogic.calculate_veritable_reward(
            self.reward_matrix, date, selected_tickers
        )

        # 3. Update internal state
        self.equity_curve.append(self.equity_curve[-1] * np.exp(reward))
        self.current_date_idx += self.holding_period

        done = self.current_date_idx >= (len(self.calendar) - self.holding_period - 1)

        info = {"date": date, "tickers": selected_tickers, "reward": reward}
        return self._get_observation(), reward, done, info
