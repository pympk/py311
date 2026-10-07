import os
import pandas as pd
import time

from pathlib import Path
from typing import List, cast

from core.settings import TradingConfig, CacheConfig
from data_pipeline.screener import UniverseScreener
from strategy.registry import get_strategy_registry


class AlphaCache:
    # Notice: It no longer takes an 'AlphaEngine'.
    # It only takes the Screener and Config!
    def __init__(
        self, screener: UniverseScreener, config: TradingConfig, lookbacks: List[int]
    ):
        self.screener = screener
        self.config = config
        self.lookbacks = lookbacks
        self.feature_cube = pd.DataFrame()

    def compute_alpha_ensemble(
        self, decision_date: pd.Timestamp, lookback_periods: List[int]
    ) -> pd.DataFrame:
        candidates = self.screener.filter_universe(
            date_ts=decision_date, thresholds=self.config.thresholds, audit_container={}
        )
        if not candidates:
            return pd.DataFrame()

        ensemble_parts = []
        registry = get_strategy_registry(self.config)

        for lb in lookback_periods:
            try:
                decision_idx = self.screener.trading_calendar.searchsorted(
                    decision_date
                )
                start_idx = int(decision_idx - lb)

                # Fix 1: Prevent silent negative wrap-around indexing
                if start_idx < 0:
                    raise IndexError(
                        f"Lookback period {lb} exceeds available history for decision date {decision_date.date()}"
                    )

                start_date = self.screener.trading_calendar[start_idx]

                obs = self.screener.build_observation(
                    decision_date=decision_date,
                    candidates=candidates,
                    start_date=start_date,
                )

                for name, blueprint in registry.items():
                    score_series = blueprint(obs).copy()
                    score_series.name = f"{lb}d_{name}"
                    ensemble_parts.append(score_series)

            except Exception as e:
                print(
                    f"[WARNING] Warning: Lookback {lb} failed for {decision_date.date()}: {e}"
                )
                continue

        if not ensemble_parts:
            return pd.DataFrame()
        return pd.concat(ensemble_parts, axis=1)

    def build(self, start_date: str = "2024-01-01"):
        all_dates = self.screener.trading_calendar
        target_dates = [d for d in all_dates if d >= pd.Timestamp(start_date)]
        cache_parts = []

        print(
            f"[INFO] Building AlphaCache for {len(target_dates)} days (Starting {start_date})..."
        )

        for i, date in enumerate(target_dates):
            ensemble = self.compute_alpha_ensemble(date, self.lookbacks)
            if ensemble.empty:
                continue

            # 1. Name the current index
            ensemble.index.name = "Ticker"

            # 2. Promote the Ticker index to a standard column
            ensemble = ensemble.reset_index()

            # 3. Add the Date column
            ensemble["Date"] = date

            # 4. Sink them both into a clean MultiIndex
            ensemble = ensemble.set_index(["Date", "Ticker"])

            cache_parts.append(ensemble)

            if i % 20 == 0:
                print(f"  Processed {i}/{len(target_dates)} days...")

        if not cache_parts:
            print(
                "[ERROR] Error: No features were generated. Check if start_date is too early for lookbacks."
            )
            return

        self.feature_cube = pd.concat(cache_parts).sort_index()
        print(f"[OK] AlphaCache built. Shape: {self.feature_cube.shape}")

    def get_vision(self, date: pd.Timestamp) -> pd.DataFrame:
        # Fix 2: Guard against uninitialized or empty cubes lacking a MultiIndex
        if self.feature_cube.empty or not isinstance(
            self.feature_cube.index, pd.MultiIndex
        ):
            return pd.DataFrame()

        try:
            # Cast the result so Pylance knows it's a DataFrame
            return cast(pd.DataFrame, self.feature_cube.xs(date, level="Date"))

        except KeyError:
            return pd.DataFrame()


class CheckpointAlphaCache(AlphaCache):
    """
    Subclass of AlphaCache that overrides the 'build' method
    to support robust progress checkpointing, automatic resumption,
    time-to-completion estimations, and proper end-date limitations.
    """

    def build(
        self,
        start_date: str = "2024-01-01",
        end_date: str = "DEFAULT",
        checkpoint_path: Path | None = None,
        save_every_n_days: int = 20,
    ):
        all_dates = self.screener.trading_calendar

        # Resolve end date with maximum flexibility:
        resolved_end_date = None
        if end_date == "DEFAULT":
            # Check environment variable first
            env_val = os.environ.get("CACHE_END_DATE")
            if env_val and env_val.strip().lower() not in ("none", "null", ""):
                resolved_end_date = env_val.strip()
            else:
                # Check CacheConfig fallback
                config_val = getattr(CacheConfig, "END_DATE", None)
                if config_val and str(config_val).strip().lower() not in (
                    "none",
                    "null",
                    "",
                ):
                    resolved_end_date = str(config_val).strip()
        elif end_date is not None:
            # User explicitly passed a specific end date string (e.g., "2026-06-04")
            resolved_end_date = end_date

        # If end_date is explicitly set to None, resolved_end_date remains None (no limit)

        # Filter the target trading dates
        target_dates = [d for d in all_dates if d >= pd.Timestamp(start_date)]
        if resolved_end_date:
            target_dates = [
                d for d in target_dates if d <= pd.Timestamp(resolved_end_date)
            ]

        if not target_dates:
            print(
                f"[ERROR] No target dates found between {start_date} and {resolved_end_date if resolved_end_date else 'End of Calendar'}."
            )
            return

        if checkpoint_path is None:
            print(
                "[INFO] No checkpoint path specified. Proceeding with standard build."
            )
            # Standard inline execution using the resolved date limits
            cache_parts = []
            for i, date in enumerate(target_dates):
                ensemble = self.compute_alpha_ensemble(date, self.lookbacks)
                if ensemble.empty:
                    continue
                ensemble.index.name = "Ticker"
                ensemble = ensemble.reset_index()
                ensemble["Date"] = date
                ensemble = ensemble.set_index(["Date", "Ticker"])
                cache_parts.append(ensemble)
                if i % 20 == 0:
                    print(f"  Processed {i}/{len(target_dates)} days...")
            if cache_parts:
                self.feature_cube = pd.concat(cache_parts).sort_index()
            return

        existing_df = pd.DataFrame()
        processed_dates = set()

        # Load existing progress file
        if checkpoint_path.exists():
            try:
                existing_df = pd.read_parquet(checkpoint_path)
                if not existing_df.empty and isinstance(
                    existing_df.index, pd.MultiIndex
                ):
                    processed_dates = set(
                        pd.Timestamp(d)
                        for d in existing_df.index.get_level_values("Date").unique()
                    )
                    print(
                        f"[RESUME] Found existing cache file with {len(processed_dates)} processed dates."
                    )
                    print(
                        f"  Existing index range: {existing_df.index.get_level_values('Date').min().date()} "
                        f"to {existing_df.index.get_level_values('Date').max().date()}"
                    )
                else:
                    print(
                        "[INFO] Existing cache file is empty or formatted incorrectly. Starting fresh."
                    )
            except Exception as e:
                print(
                    f"[WARNING] Failed to load existing cache file: {e}. Starting fresh."
                )

        # Isolate remaining dates to compute
        dates_to_process = [
            d for d in target_dates if pd.Timestamp(d) not in processed_dates
        ]

        if not dates_to_process:
            print(
                "[INFO] All target dates have already been processed! Skipping generation."
            )
            self.feature_cube = existing_df
            return

        print(f"[INFO] Building AlphaCache with automatic resumption support:")
        print(f"  - Start Date:               {start_date}")
        print(
            f"  - End Date Limit:           {resolved_end_date if resolved_end_date else 'None (Processing all available dates)'}"
        )
        print(f"  - Total target dates:       {len(target_dates)}")
        print(f"  - Already completed:        {len(processed_dates)}")
        print(f"  - Remaining to compute:     {len(dates_to_process)}")
        print(f"  - Checkpoint save interval: Every {save_every_n_days} processed days")
        print(f"  - Output location:          {checkpoint_path}")

        new_cache_parts = []
        loop_start_time = time.time()
        total_to_process = len(dates_to_process)

        # Sub-helper to execute atomic saves safely
        def save_checkpoint(df_accumulated, parts_to_concat):
            if parts_to_concat:
                new_chunk = pd.concat(parts_to_concat)
                if not df_accumulated.empty:
                    df_accumulated = pd.concat([df_accumulated, new_chunk])
                else:
                    df_accumulated = new_chunk
                df_accumulated = df_accumulated.sort_index()

                temp_path = checkpoint_path.with_suffix(".tmp")
                try:
                    df_accumulated.to_parquet(temp_path)
                    if temp_path.exists():
                        if checkpoint_path.exists():
                            checkpoint_path.unlink()
                        temp_path.rename(checkpoint_path)
                except Exception as ex:
                    print(f"[ERROR] Failed to save checkpoint file: {ex}")
            return df_accumulated

        for i, date in enumerate(dates_to_process):
            ensemble = self.compute_alpha_ensemble(date, self.lookbacks)
            if ensemble.empty:
                continue

            ensemble.index.name = "Ticker"
            ensemble = ensemble.reset_index()
            ensemble["Date"] = date
            ensemble = ensemble.set_index(["Date", "Ticker"])

            new_cache_parts.append(ensemble)

            # Performance math and estimations
            processed_so_far = i + 1
            elapsed = time.time() - loop_start_time
            sec_per_day = elapsed / processed_so_far
            remaining_days = total_to_process - processed_so_far
            est_sec_left = remaining_days * sec_per_day

            h = int(est_sec_left // 3600)
            m = int((est_sec_left % 3600) // 60)
            s = int(est_sec_left % 60)
            time_str = f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m}:{s:02d}"

            # Interval saving
            if i > 0 and i % save_every_n_days == 0:
                if new_cache_parts:
                    existing_df = save_checkpoint(existing_df, new_cache_parts)
                    self.feature_cube = existing_df
                    print(
                        f"  [CHECKPOINT] Successfully saved progress. "
                        f"Total days: {len(existing_df.index.get_level_values('Date').unique())} "
                        f"({processed_so_far}/{total_to_process} remaining processed | est. {time_str} till completion)"
                    )
                    new_cache_parts = []

            if i % 20 == 0 and i > 0:
                print(
                    f"  Processed {i}/{total_to_process} remaining days... (est. {time_str} till completion)"
                )

        # --- FINAL POST-LOOP FLUSH ---
        # Flush any remaining items left in new_cache_parts (corrects the loop leak bug)
        if new_cache_parts:
            existing_df = save_checkpoint(existing_df, new_cache_parts)
            print(
                f"  [FINAL SAVED] Flushed and saved remaining progress. "
                f"Total days: {len(existing_df.index.get_level_values('Date').unique())}"
            )
            new_cache_parts = []

        self.feature_cube = existing_df
        print(
            f"[OK] AlphaCache building phase completed. Final Shape: {self.feature_cube.shape}"
        )


#
