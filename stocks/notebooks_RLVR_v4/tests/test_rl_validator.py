import numpy as np
import pandas as pd
from rl_discovery.agent import AbsoluteZeroAgent
from rl_discovery.validator import AgentEvaluator
from rl_discovery.adapter import RLVRGymEnv


class MockDiscoveryEnv:
    """Minimal mock environment implementing the Gymnasium 5-tuple step protocol."""

    def __init__(self):
        self.step_count = 0
        self.holding_period = 5
        # Dynamic Space Detection requires a .cube attribute with 12 features
        self.cube = pd.DataFrame(np.zeros((1, 12)))

    def reset(self, seed=None, **kwargs):
        self.step_count = 0
        return {
            "date": pd.Timestamp("2024-01-01"),
            "ensemble": pd.DataFrame(np.random.randn(2, 12)),
            "macro_row": pd.Series(np.zeros(11)),
            "bm_row": pd.Series(np.zeros(12)),
        }

    def step(self, action):
        self.step_count += 1
        terminated = False
        truncated = self.step_count >= 10
        reward = 0.01

        info = {
            "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=self.step_count),
            "net_daily_simple_ret": 0.01,
            "bm_daily_simple_ret": 0.0,
            "alpha_daily_simple_ret": 0.01,
            "penalized_alpha_daily_simple_ret": 0.01,
        }

        obs = {
            "date": info["date"],
            "ensemble": pd.DataFrame(np.random.randn(2, 12)),
            "macro_row": pd.Series(np.zeros(11)),
            "bm_row": pd.Series(np.zeros(12)),
        }
        return obs, reward, terminated, truncated, info


def test_evaluator_deterministic_execution():
    """
    [GUARD] Verifies the evaluator runs without gradients and outputs clean quant metrics.
    """
    mock_macro = pd.DataFrame(
        np.random.randn(20, 11), index=pd.date_range("2024-01-01", periods=20)
    )
    env = RLVRGymEnv(MockDiscoveryEnv(), mock_macro)

    # Extract shapes and narrow types
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape

    assert obs_shape is not None, "Observation space shape is None"
    assert action_shape is not None, "Action space shape is None"

    agent = AbsoluteZeroAgent(obs_dim=obs_shape[0], action_dim=action_shape[0])

    # Run deterministic evaluation
    results = AgentEvaluator.evaluate(agent, env)

    # Assertions
    assert "total_return" in results, "Missing total_return metric"
    assert "sharpe_ratio" in results, "Missing sharpe_ratio metric"
    assert "information_ratio" in results, "Missing information_ratio metric"
    assert "equity_curve" in results, "Missing equity_curve"

    # Equity curve must have N + 1 points (including 1.0 initial seed)
    assert (
        len(results["equity_curve"]) == results["steps"] + 1
    ), "Equity curve length mismatch"

    # Flat return sequence (std = 0) -> Sharpe zero-division safeguard returns 0.0
    assert results["sharpe_ratio"] == 0.0, "Sharpe zero-division safeguard failed"

    # Verify agent was reset to training mode
    assert (
        agent.training is True
    ), "Agent was not returned to training mode after evaluation"
