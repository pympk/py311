import pandas as pd
import numpy as np
import time

from typing import List, Any

from core.logic import AlphaLogic


# class FeatureCubeBuilder:
#     """STATELESS: The factory that 'bakes' the AlphaCache."""

#     @staticmethod
#     def build(
#         master_engine: Any, lookbacks: List[int], start_date: pd.Timestamp
#     ) -> pd.DataFrame:
#         calendar = master_engine.trading_calendar
#         target_dates = calendar[calendar >= start_date]

#         cache_parts = []
#         print(f"🏗️ Building Final AlphaCache for {len(target_dates)} days...")

#         for date in target_dates:
#             # 1. Get raw metrics from Engine
#             raw_ensemble = master_engine.compute_alpha_ensemble(date, lookbacks)

#             if raw_ensemble.empty:
#                 continue

#             # 2. THE STRICT SURVIVOR MASK (THE FIX)
#             # If a stock is missing ANY of the 33 metrics, it is purged.
#             # This ensures Mean and Std are calculated on a perfectly aligned universe.
#             clean_ensemble = raw_ensemble.dropna()

#             if clean_ensemble.empty:
#                 continue

#             # 3. SYNCHRONIZED Z-SCORING (Excel Parity)
#             # We use ddof=1 to match Excel's STDEV.S exactly.
#             normalized = (clean_ensemble - clean_ensemble.mean()) / clean_ensemble.std(
#                 ddof=1
#             )

#             # 4. SLUGIFY NAMES
#             normalized.columns = AlphaLogic.slugify_columns(normalized.columns)

#             # 5. Prepare for MultiIndex
#             normalized["Date"] = date
#             normalized = normalized.set_index(["Date", normalized.index])
#             cache_parts.append(normalized)

#         if not cache_parts:
#             return pd.DataFrame()

#         final_cube = pd.concat(cache_parts).sort_index()
#         print(f"✅ Feature Cube Built. Shape: {final_cube.shape}")
#         return final_cube


class FeatureCubeBuilder:
    """STATELESS: The factory that 'bakes' the AlphaCache with Progress Tracking."""

    @staticmethod
    def build(
        master_engine: Any, lookbacks: List[int], start_date: pd.Timestamp
    ) -> pd.DataFrame:
        calendar = master_engine.trading_calendar
        target_dates = calendar[calendar >= start_date]
        total_days = len(target_dates)

        cache_parts = []
        start_time = time.time()

        print(f"🏗️  Starting Final AlphaCache Build: {total_days} days.")
        print(f"📅  Window: {target_dates[0].date()} to {target_dates[-1].date()}")
        print("-" * 50)

        for i, date in enumerate(target_dates):
            # 1. Generate Raw Metrics
            raw_ensemble = master_engine.compute_alpha_ensemble(date, lookbacks)

            if raw_ensemble.empty:
                continue

            # 2. THE STRICT SURVIVOR MASK
            # Drop any ticker missing ANY metric to ensure perfect Z-score parity
            clean_ensemble = raw_ensemble.dropna()

            if clean_ensemble.empty:
                continue

            # 3. SYNCHRONIZED Z-SCORING (ddof=1 for Excel Parity)
            normalized = (clean_ensemble - clean_ensemble.mean()) / clean_ensemble.std(
                ddof=1
            )

            # 4. SLUGIFY NAMES
            normalized.columns = AlphaLogic.slugify_columns(normalized.columns)

            # 5. Prepare for MultiIndex
            normalized["Date"] = date
            normalized = normalized.set_index(["Date", normalized.index])
            cache_parts.append(normalized)

            # --- PROGRESS HEARTBEAT (Every 10 days) ---
            if i % 10 == 0 and i > 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                remaining_days = total_days - i
                est_remaining = (remaining_days * avg_time) / 60

                print(
                    f"🔄  [{i:03d}/{total_days}] | Date: {date.date()} | "
                    f"Survivors: {len(clean_ensemble):03d} | "
                    f"Est. Remaining: {est_remaining:.1f} mins"
                )

        if not cache_parts:
            print("❌  Build Failed: No valid features generated.")
            return pd.DataFrame()

        final_cube = pd.concat(cache_parts).sort_index()
        total_elapsed = (time.time() - start_time) / 60

        print("-" * 50)
        print(f"✅  Feature Cube Built in {total_elapsed:.1f} mins.")
        print(f"📊  Final Shape: {final_cube.shape}")
        return final_cube


#
