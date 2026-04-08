import pandas as pd
import numpy as np
from typing import List


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
