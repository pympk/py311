import pytest
import torch
import numpy as np
import pandas as pd
from rl_discovery.agent import AbsoluteZeroAgent
from rl_discovery.validator import AgentEvaluator
from rl_discovery.adapter import RLVRGymEnv


class MockDiscoveryEnv:
    """Minimal mock environment for deterministic rollout testing."""

    def __init__(self):
        self.step_count = 0
        self.holding_period = 5

    def reset(self):
        self.step_count = 0
        return {
            "date": pd.Timestamp("2024-01-01"),
            "ensemble": pd.DataFrame(np.random.randn(2, 11)),
        }

    def step(self, action):
        self.step_count += 1
        done = self.step_count >= 10
        # Give a small positive reward
        reward = 0.01
        info = {"date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=self.step_count)}
        obs = {"date": info["date"], "ensemble": pd.DataFrame(np.random.randn(2, 11))}
        return obs, reward, done, info


def test_evaluator_deterministic_execution():
    """
    [GUARD] Verifies the evaluator runs without gradients and outputs clean quant metrics.
    """
    agent = AbsoluteZeroAgent()
    mock_macro = pd.DataFrame(
        np.random.randn(20, 11), index=pd.date_range("2024-01-01", periods=20)
    )
    env = RLVRGymEnv(MockDiscoveryEnv(), mock_macro)

    # Run evaluation
    results = AgentEvaluator.evaluate(agent, env)

    # Assertions
    assert "total_return" in results, "Missing total_return metric"
    assert "sharpe_ratio" in results, "Missing sharpe_ratio metric"
    assert "equity_curve" in results, "Missing equity_curve"

    # The equity curve should have N+1 points (including the 1.0 start)
    assert (
        len(results["equity_curve"]) == results["steps"] + 1
    ), "Equity curve length mismatch"

    # Since reward is consistently 0.01, standard deviation is 0.
    # Our safeguard should trap division by zero and return Sharpe of 0.0
    assert results["sharpe_ratio"] == 0.0, "Sharpe zero-division safeguard failed"

    # Verify agent was reset to training mode
    assert (
        agent.training is True
    ), "Agent was not returned to training mode after evaluation"
