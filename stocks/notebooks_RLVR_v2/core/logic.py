import pandas as pd
import numpy as np

from typing import List
from core.settings import TradingConfig


class AlphaLogic:
    """STATELESS: The mathematical engine for rewards and ensemble generation."""

    @staticmethod
    def calculate_veritable_reward(
        reward_matrix: pd.DataFrame, date: pd.Timestamp, tickers: List[str]
    ) -> float:
        """The Log-Return truth engine. LN(1 + Arithmetic_Mean)."""
        if not tickers or date not in reward_matrix.index:
            return 0.0

        row = reward_matrix.loc[date]
        valid_returns = row.reindex(tickers).dropna()
        if valid_returns.empty:
            return 0.0

        # .to_numpy() guarantees a single scalar float from .mean()
        arith_mean = float(valid_returns.to_numpy().mean())

        return float(np.log1p(arith_mean))

    @staticmethod
    def slugify_columns(columns: List[str]) -> List[str]:
        """Ensures names are machine-safe: 21d_Sharpe_(ATRP) -> 21d_Sharpe_ATRP"""
        return [
            c.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "")
            .replace(",", "")
            .replace("__", "_")
            for c in columns
        ]


class SelectionLogic:
    """STATELESS: Decodes actions into ticker lists."""

    @staticmethod
    def apply_action(
        ensemble: pd.DataFrame,
        action: np.ndarray,
        # ---> FIXED: Pulling directly from TradingConfig class attributes
        rank_max_offset_percentile: float = TradingConfig.rank_max_offset_percentile,
        rank_max_width: int = TradingConfig.rank_max_width,
    ) -> tuple:
        """Vectorized Matrix Multiplication + Sorting."""
        if ensemble.empty:
            return [], [], 0, 0, 0.0, 0.0

        # ---> DYNAMIC FEATURE WEIGHT EXTRACTION <---
        # Automatically handles any feature count by slicing off the last 2 rank dimensions
        weights = action[:-2]

        # ---> NEW: DYNAMIC OFFSET <---
        universe_size = len(ensemble)
        # Offset is bounded by a percentage of today's available universe
        max_allowed_offset = int(universe_size * rank_max_offset_percentile)

        # Interpolate width from [0, rank_max_width]
        width = int(np.interp(action[-1], [-1, 1], [0, rank_max_width]))

        # Interpolate offset from [0, max_allowed_offset]
        offset = int(np.interp(action[-2], [-1, 1], [0, max_allowed_offset]))

        # ---> THE FIX: Clean the NaNs before math <---
        # In a Z-score space, 0.0 is exactly neutral (market average)
        clean_ensemble = ensemble.fillna(0.0)

        # Vectorized Scoring
        scores = pd.Series(clean_ensemble.values @ weights, index=clean_ensemble.index)
        sorted_tickers = scores.sort_values(ascending=False)

        # Extract metadata
        top_3 = sorted_tickers.index[:3].tolist()
        selected = sorted_tickers.index[offset : offset + width].tolist()

        return selected, top_3, offset, width, float(scores.max()), float(scores.min())
