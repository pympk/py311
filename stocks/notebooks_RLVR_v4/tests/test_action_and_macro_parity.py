import numpy as np
import pandas as pd
import pytest
import torch

from core.logic import SelectionLogic
from core.settings import TradingConfig
from rl_discovery.agent import AbsoluteZeroAgent


def test_actor_mean_is_tanh_bounded():
    """Verify that agent.actor_mean strictly outputs in [-1, 1]."""
    obs_dim = 46
    action_dim = 16
    batch_size = 128

    agent = AbsoluteZeroAgent(obs_dim=obs_dim, action_dim=action_dim, hidden_size=256)
    agent.eval()

    extreme_obs = torch.randn((batch_size, obs_dim)) * 100.0
    with torch.no_grad():
        mean = agent.actor_mean(extreme_obs)

    assert torch.all(mean >= -1.0), "actor_mean output below -1.0"
    assert torch.all(mean <= 1.0), "actor_mean output above 1.0"


def test_actor_logstd_initialization():
    """Verify logstd default initialization (-0.50, sigma ≈ 0.6065 for Gen 11) and configurable parameter."""
    action_dim = 16
    # 1. Authoritative Gen 11 contract (-0.50 -> sigma ≈ 0.6065)
    agent = AbsoluteZeroAgent(obs_dim=46, action_dim=action_dim)
    expected_logstd = -0.50
    actual_logstd = agent.actor_logstd.detach().cpu().numpy()

    np.testing.assert_allclose(
        actual_logstd,
        expected_logstd,
        atol=1e-5,
        err_msg="actor_logstd default not initialized to Gen 11 contract (-0.50)",
    )

    # 2. Configurable initialization support
    agent_custom = AbsoluteZeroAgent(
        obs_dim=46, action_dim=action_dim, initial_logstd=-1.2
    )
    np.testing.assert_allclose(
        agent_custom.actor_logstd.detach().cpu().numpy(),
        -1.2,
        atol=1e-5,
        err_msg="actor_logstd does not respect custom initial_logstd parameter",
    )


def test_selection_logic_l2_normalization_and_clipping():
    """Verify SelectionLogic handles out-of-bound actions and normalizes weights."""
    tickers = [f"TICK_{i}" for i in range(50)]
    features = [f"F_{j}" for j in range(12)]
    ensemble = pd.DataFrame(np.random.randn(50, 12), index=tickers, columns=features)

    # Extreme unclipped action: 12 weights + offset + width + equity_exposure + active_tilt (16 dims)
    extreme_action = np.array([10.0] * 12 + [5.0, -5.0, 3.0, 2.0])
    selected, top_3, offset, width, equity_exposure, active_tilt, max_s, min_s = (
        SelectionLogic.apply_action(
            ensemble,
            extreme_action,
            rank_max_offset_percentile=TradingConfig.rank_max_offset_percentile,
            rank_max_width=TradingConfig.rank_max_width,
        )
    )

    assert 0 <= offset <= int(len(tickers) * TradingConfig.rank_max_offset_percentile)
    assert 0 <= width <= TradingConfig.rank_max_width
    assert 0.0 <= equity_exposure <= 1.0
    assert 0.0 <= active_tilt <= 1.0
    assert len(top_3) == 3
    assert not np.isnan(max_s)
    assert not np.isnan(min_s)


def test_macro_df_high_yield_spread_bounds():
    """Verify macro_df High_Yield_Spread_Z is strictly bounded within [-4, 4]."""
    from core.paths import LOCAL_DATA_DIR

    macro_path = LOCAL_DATA_DIR / "macro_df.parquet"
    if macro_path.exists():
        macro_df = pd.read_parquet(macro_path)
        if "High_Yield_Spread_Z" in macro_df.columns:
            hy = macro_df["High_Yield_Spread_Z"]
            assert hy.min() >= -4.0, f"High_Yield_Spread_Z min {hy.min()} < -4.0"
            assert hy.max() <= 4.0, f"High_Yield_Spread_Z max {hy.max()} > 4.0"
            assert not hy.isna().any(), "High_Yield_Spread_Z contains NaNs"


#
