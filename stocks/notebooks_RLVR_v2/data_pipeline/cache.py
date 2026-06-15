import pandas as pd

from typing import List, cast

from core.settings import TradingConfig
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


#
