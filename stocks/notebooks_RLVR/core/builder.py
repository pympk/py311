import glob
import pandas as pd
import os, glob, multiprocessing
import pandas as pd
import gc

from tqdm.auto import tqdm  # For a beautiful, professional progress bar
from core.logic import AlphaLogic


class ParallelFeatureBuilder:
    """
    ULTRA-PERFORMANCE BUILDER: Uses all CPU cores to bake the AlphaCache.
    Features: Parallelism, Atomic Checkpointing, and Smart Recovery.
    """

    @staticmethod
    def _worker_task(date, engine_params, lookbacks):
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

            # 3. Normalization (ddof=1 for Excel Parity)
            normalized = (clean_ensemble - clean_ensemble.mean()) / clean_ensemble.std(
                ddof=1
            )

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
        num_workers=None,  # <--- CRITICAL FIX: Add this to the signature
    ):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # --- THE TAME LOGIC ---
        # If not specified, we use ALL cores minus 2 (The 'User-Friendly' offset)
        if num_workers is None:
            num_workers = max(1, multiprocessing.cpu_count() - 2)

        # 1. IDENTIFY GAPS (Smart Recovery)
        calendar = master_engine.trading_calendar
        all_dates = calendar[calendar >= pd.Timestamp(start_date)]
        processed_dates = ParallelFeatureBuilder._get_processed_dates(checkpoint_dir)
        missing_dates = [d for d in all_dates if d not in processed_dates]

        if not missing_dates:
            print("✅ All data baked and verified.")
            return

        # Simplified heartbeat print
        print(
            f"🚀 Parallel Bake: {len(missing_dates)} days | Using {num_workers} of {multiprocessing.cpu_count()} cores."
        )
        print(
            f"🖥️  {multiprocessing.cpu_count() - num_workers} cores reserved for your PC usage."
        )

        # 2. CHUNK THE WORK
        date_chunks = [
            missing_dates[i : i + batch_size]
            for i in range(0, len(missing_dates), batch_size)
        ]

        engine_params = {"engine": master_engine}

        with tqdm(total=len(missing_dates), desc="Baking AlphaCache") as pbar:
            for chunk in date_chunks:
                # Use the 'Tame' worker count here
                with multiprocessing.Pool(processes=num_workers) as pool:
                    results = pool.starmap(
                        ParallelFeatureBuilder._worker_task,
                        [(d, engine_params, lookbacks) for d in chunk],
                    )

                # Filter out None/Errors and save batch
                valid_results = [r for r in results if isinstance(r, pd.DataFrame)]
                if valid_results:
                    # Logic Check: We use 'sort=False' for speed since we sort at the very end
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
            # We only read the index (Date) to keep recovery lightning fast
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
            # One last check for duplicates across batches
            all_chunks.append(chunk[~chunk.index.duplicated(keep="first")])

        full_df = pd.concat(all_chunks).sort_index()
        # Release intermediate memory
        del all_chunks
        gc.collect()

        # Save to Parquet
        full_df.to_parquet(final_output_fn, engine="pyarrow", compression="zstd")
        print(f"✅ Final Master Cube Saved! Shape: {full_df.shape}")
        return full_df


#
