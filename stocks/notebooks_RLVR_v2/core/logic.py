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
    """STATELESS: Decodes actions into ticker lists and allocation controls."""

    @staticmethod
    def apply_action(
        ensemble: pd.DataFrame,
        action: np.ndarray,
        rank_max_offset_percentile: float = 1.0,
        rank_max_width: int = 10,
        min_basket_width: int = 1,
        max_cash_pct: float = 0.0,
        min_active_tilt: float = 0.0,
    ) -> tuple:
        """
        Vectorized Matrix Multiplication + Sorting + Strict Dynamic Tri-Asset Allocation.
        Action vector strictly matches K (feature count) + 4 control dimensions:
          [w_1, ..., w_K, offset_ctrl, width_ctrl, equity_exposure_ctrl, active_tilt_ctrl]
        """
        if ensemble.empty:
            return [], [], 0, 0, 0.0, 0.0, 0.0, 0.0

        if action is None:
            raise ValueError("Action vector cannot be None.")

        num_features = ensemble.shape[1]
        expected_dim = num_features + 4
        if len(action) != expected_dim:
            raise ValueError(
                f"Strict Action Dimension Mismatch: Expected {expected_dim} dimensions "
                f"(K={num_features} alpha weights + 4 control dimensions), got {len(action)}."
            )

        # 1. Clean & Clip Action Space
        clipped_action = np.clip(np.nan_to_num(action, nan=0.0), -1.0, 1.0)

        # 2. Extract & L2-Normalize Alpha Feature Weights
        weights = clipped_action[:num_features]
        norm = np.linalg.norm(weights)
        if norm > 1e-6:
            weights = weights / norm

        # 3. Decode Dynamic Controls (Strict K+4 dimensions)
        universe_size = len(ensemble)
        max_allowed_offset = int(universe_size * rank_max_offset_percentile)

        offset = int(
            np.interp(clipped_action[-4], [-1.0, 1.0], [0, max_allowed_offset])
        )
        # Bounded basket width between [min_basket_width, rank_max_width]
        width = int(
            np.interp(
                clipped_action[-3],
                [-1.0, 1.0],
                [min_basket_width, rank_max_width],
            )
        )
        min_equity = max(0.0, min(1.0, 1.0 - max_cash_pct))
        equity_exposure = float(
            np.interp(clipped_action[-2], [-1.0, 1.0], [min_equity, 1.0])
        )
        # Bounded active tilt between [min_active_tilt, 1.0]
        active_tilt = float(
            np.interp(clipped_action[-1], [-1.0, 1.0], [min_active_tilt, 1.0])
        )

        # 4. Clean NaNs & Vectorized Scoring
        clean_ensemble = ensemble.fillna(0.0)
        scores = pd.Series(
            clean_ensemble.to_numpy() @ weights, index=clean_ensemble.index
        )
        sorted_tickers = scores.sort_values(ascending=False)

        top_3 = sorted_tickers.index[:3].tolist()
        selected = sorted_tickers.index[offset : offset + width].tolist()

        return (
            selected,
            top_3,
            offset,
            width,
            equity_exposure,
            active_tilt,
            float(scores.max()) if not scores.empty else 0.0,
            float(scores.min()) if not scores.empty else 0.0,
        )


#####################
