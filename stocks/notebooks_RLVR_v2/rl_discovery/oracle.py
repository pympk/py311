import pandas as pd
import numpy as np

from typing import List, cast, Optional

from core.contracts import DiscoveryResult, EngineInput
from core.settings import TradingConfig
from core.quant import QuantUtils
from data_pipeline.screener import UniverseScreener
from strategy.registry import get_strategy_registry


class RLOracle:
    """
    Provides state observations (alpha matrices, context vectors)
    and computes veritable rewards for the RL Agent.
    """

    def __init__(self, screener: UniverseScreener, config: TradingConfig):
        self.screener = screener
        self.config = config
        self.reward_matrix = pd.DataFrame()

    def precompute_reward_matrix(self, holding_period: int):
        close_df = self.screener.df_close
        self.reward_matrix = (
            close_df.shift(-(holding_period + 1)) / close_df.shift(-1)
        ) - 1.0
        self.reward_matrix = self.reward_matrix.fillna(0.0)

    def get_batch_reward(
        self, decision_date: pd.Timestamp, tickers: List[str]
    ) -> float:
        if decision_date not in self.reward_matrix.index:
            return 0.0
        arithmetic_group_return = self.reward_matrix.loc[decision_date][tickers].mean()
        return float(np.log1p(arithmetic_group_return))

    def compute_context_vector(self, decision_date: pd.Timestamp) -> pd.Series:
        macro_df = self.screener.macro_df
        if macro_df is None or decision_date not in macro_df.index:
            return pd.Series(
                {
                    "Context_Trend": 0.0,
                    "Context_Vel_Z": 0.0,
                    "Context_Vix_Z": 0.0,
                    "Context_Vix_Ratio": 1.0,
                }
            )

        # CAST HERE:
        macro_row = cast(pd.Series, macro_df.loc[decision_date])

        return pd.Series(
            {
                "Context_Trend": float(macro_row.get("Macro_Trend", 0.0)) * 10,
                "Context_Vel_Z": float(macro_row.get("Macro_Trend_Vel_Z", 0.0)),
                "Context_Vix_Z": float(macro_row.get("Macro_Vix_Z", 0.0)),
                "Context_Vix_Ratio": float(macro_row.get("Macro_Vix_Ratio", 1.0)) - 1.0,
            }
        )

    def compute_alpha_matrix(
        self, decision_date: pd.Timestamp, lookback_period: int
    ) -> pd.DataFrame:
        mock_input = EngineInput(
            mode="Discovery",
            decision_date=decision_date,
            lookback_period=lookback_period,
            holding_period=1,
            metric="All",
            benchmark_ticker=self.config.benchmark_ticker,
            quality_thresholds=self.config.thresholds,
        )

        try:
            safe_start, safe_decision, _, _ = self.screener.validate_timeline(
                mock_input
            )
        except ValueError as e:
            print(f"Timeline Error for {decision_date.date()}: {e}")
            return pd.DataFrame()

        candidates = self.screener.filter_universe(
            safe_decision, self.config.thresholds, audit_container={}
        )
        if not candidates:
            return pd.DataFrame()

        obs = self.screener.build_observation(safe_decision, candidates, safe_start)
        alpha_results = {}
        registry = get_strategy_registry(self.config)

        for name, blueprint in registry.items():
            try:
                suffix = " (Z)" if blueprint.scaling_type == "Z-Score" else " (S)"
                tagged_name = f"{name}{suffix}"
                scores = blueprint.get_agent_view(obs, config=self.config)

                if isinstance(scores, (pd.Series, pd.DataFrame)):
                    alpha_results[tagged_name] = scores
                else:
                    alpha_results[tagged_name] = pd.Series(scores, index=candidates)
            except Exception as e:
                print(f"Warning: Strategy '{name}' failed: {e}")
                alpha_results[f"{name} (Err)"] = pd.Series(np.nan, index=candidates)

        alpha_matrix = pd.DataFrame(alpha_results)
        alpha_matrix.index.name = "Ticker"
        return alpha_matrix

    def normalize_alpha_matrix(self, alpha_matrix: pd.DataFrame) -> pd.DataFrame:
        if alpha_matrix.empty:
            return alpha_matrix
        normalized = alpha_matrix.apply(QuantUtils.zscore)
        clip_val = self.config.feature_zscore_clip
        return normalized.clip(-clip_val, clip_val).fillna(0.0)

    def run_discovery_action(
        self,
        decision_date: pd.Timestamp,
        lookback_period: int,
        holding_period: int,
        weights: np.ndarray,
    ) -> Optional[DiscoveryResult]:
        raw_matrix = self.compute_alpha_matrix(decision_date, lookback_period)
        norm_matrix = self.normalize_alpha_matrix(raw_matrix)
        if norm_matrix.empty:
            return None

        discovery_scores = norm_matrix.values @ weights
        discovery_series = pd.Series(discovery_scores, index=norm_matrix.index)
        top_tickers = (
            discovery_series.sort_values(ascending=False).head(10).index.tolist()
        )
        veritable_reward = self.get_batch_reward(decision_date, top_tickers)

        registry = get_strategy_registry(self.config)
        return DiscoveryResult(
            action_weights=dict(zip(list(registry.keys()), weights)),
            selected_tickers=top_tickers,
            veritable_reward=veritable_reward,
            metric_values=discovery_series.loc[top_tickers],
            raw_alpha_matrix=raw_matrix,
        )
