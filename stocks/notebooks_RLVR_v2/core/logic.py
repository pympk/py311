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

        # Arithmetic Mean of the group
        # 1. Pull the row for the date (Returns a Series)
        row = reward_matrix.loc[date]

        # 2. Filter for specific tickers and calculate mean
        # .reindex is safer for type checkers than double-indexing
        arith_mean = row.reindex(tickers).mean()

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
        ensemble: pd.DataFrame,
        action: np.ndarray,
        rank_max_offset: int = TradingConfig.rank_max_offset,
        rank_max_width: int = TradingConfig.rank_max_width,
    ) -> tuple:
        """Vectorized Matrix Multiplication + Sorting."""
        if ensemble.empty:
            return [], [], 0, 0, 0.0, 0.0

        n_features = 11  # 13 dims total: first 11 are weights
        weights = action[:n_features]

        # Map normalized [-1, 1] to discrete ranges
        offset = int(np.interp(action[-2], [-1, 1], [0, rank_max_offset]))
        width = int(np.interp(action[-1], [-1, 1], [1, rank_max_width]))

        # Vectorized Scoring
        scores = pd.Series(ensemble.values @ weights, index=ensemble.index)
        sorted_tickers = scores.sort_values(ascending=False)

        # Extract metadata
        top_3 = sorted_tickers.index[:3].tolist()
        selected = sorted_tickers.index[offset : offset + width].tolist()

        return selected, top_3, offset, width, float(scores.max()), float(scores.min())
