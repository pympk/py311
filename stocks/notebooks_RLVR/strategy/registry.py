from typing import Dict

from core.quant import QuantUtils
from core.contracts import MetricBlueprint


# METRIC_REGISTRY_OLD = {
#     "Price Gain": lambda obs: QuantUtils.calculate_gain(obs.lookback_close),
#     "Sharpe": lambda obs: QuantUtils.calculate_sharpe(obs.lookback_returns),
#     "Sharpe (ATRP)": lambda obs: QuantUtils.calculate_sharpe_vol(
#         obs.lookback_returns, obs.atrp
#     ),
#     "Sharpe (TRP)": lambda obs: QuantUtils.calculate_sharpe_vol(
#         obs.lookback_returns, obs.trp
#     ),
#     "Momentum (21d)": lambda obs: obs.mom_21,
#     "Info Ratio (Stdev_Alpha, 63d)": lambda obs: obs.ir_63,
#     "Consistency (WinRate 5d)": lambda obs: obs.consistency,
#     "Oversold (-RSI)": lambda obs: -obs.rsi,
#     "Dip Buyer (Drawdown -dd_21)": lambda obs: -obs.dd_21,
#     "Low Volatility (-ATRP)": lambda obs: -obs.atrp,
#     # Example using the NEW macro data we added to the dataclass!
#     "VIX Filtered Momentum": lambda obs: (
#         obs.mom_21 if obs.macro_vix_ratio < 1.0 else obs.mom_21 * 0
#     ),
# }


# METRIC_REGISTRY_ALL = {
#     "Log Price Gain": lambda obs: QuantUtils.calculate_gain(obs.lookback_close),
#     "Sharpe": lambda obs: QuantUtils.calculate_sharpe(obs.lookback_returns),
#     "Sharpe (ATRP)": lambda obs: QuantUtils.calculate_sharpe_vol(
#         obs.lookback_returns, obs.atrp
#     ),
#     "Sharpe (TRP)": lambda obs: QuantUtils.calculate_sharpe_vol(
#         obs.lookback_returns, obs.trp
#     ),
#     "Momentum (21d)": lambda obs: obs.mom_21,
#     "Info Ratio (Stdev_Alpha, 63d)": lambda obs: obs.ir_63,
#     "Consistency (WinRate 5d)": lambda obs: obs.consistency,
#     "Oversold (-RSI)": lambda obs: -obs.rsi,
#     "Dip Buyer (Drawdown -dd_21)": lambda obs: -obs.dd_21,
#     "Low Volatility (-ATRP)": lambda obs: -obs.atrp,
#     # Example using the NEW macro data we added to the dataclass!
#     "VIX Filtered Momentum": lambda obs: (
#         obs.mom_21 if obs.macro_vix_ratio < 1.0 else obs.mom_21 * 0
#     ),
#     # # 1. PURE RETURN (The "Gas Pedal")
#     # # 0% complexity; directly maps to the RL agent's Log Return reward.
#     # "Log Price Gain": lambda obs: QuantUtils.calculate_gain(obs.lookback_close),
#     # # 2. RISK-ADJUSTED TREND (The "Efficient" Pillar)
#     # # Replaces all other Sharpe variants. Provides the best balance of return vs volatility.
#     # "Sharpe (TRP)": lambda obs: QuantUtils.calculate_sharpe_vol(
#     #     obs.lookback_returns, obs.trp
#     # ),
#     # # 3. QUALITY & STEADINESS (The "Consistency" Pillar)
#     # # The most unique metric in the registry (~89% unique info).
#     # # Measures the 'smoothness' of the equity curve, which is critical for Log Returns.
#     # "Consistency (WinRate 5d)": lambda obs: obs.consistency,
#     # # 4. RECOVERY ALPHA (The "Dip Buyer" Pillar)
#     # # Finds structural value after a 21-day drawdown.
#     # # Helps the agent pivot during market corrections.
#     # "Dip Buyer (Drawdown -dd_21)": lambda obs: -obs.dd_21,
#     # # 5. VOLATILITY PROTECTION (The "Quiet" Pillar)
#     # # Completely orthogonal (~91% unique).
#     # # Essential for the agent to reduce 'Variance Drag' in the portfolio.
#     # "Low Volatility (-ATRP)": lambda obs: -obs.atrp,
#     # # 6. REGIME-AWARE TREND (The "Macro" Pillar)
#     # # Pure momentum but includes the built-in VIX 'Off-Switch' for safety.
#     # "VIX Filtered Momentum": lambda obs: (
#     #     obs.mom_21 if obs.macro_vix_ratio < 1.0 else obs.mom_21 * 0
#     # ),
#     ############################
#     # 7. REGIME DETECTION (The "Texture" Pillar)
#     # High (+) = Momentum is 'sticky'. Low (-) = Momentum is 'shaky/reverting'.
#     "Return Autocorr (15d)": lambda obs: obs.autocorr_15,
#     # 8. MEAN REVERSION / EXTREMES (The "Boundary" Pillar)
#     # High = Overbought/Breakout; Low = Oversold/Value.
#     "Range Position (20d)": lambda obs: obs.range_pos_20,
#     # 9. VOLUME CONFIRMATION (The "Fuel" Pillar)
#     # Measures the gap between price direction and volume flow.
#     # We Z-score both within the registry to ensure they are on the same scale.
#     "OBV Divergence (5d)": lambda obs: (
#         (obs.slope_v_5 - obs.slope_v_5.mean()) / obs.slope_v_5.std()
#     )
#     - ((obs.slope_p_5 - obs.slope_p_5.mean()) / obs.slope_p_5.std()),
#     # 10. MOMENTUM PHYSICS (The "Exhaustion" Pillar)
#     # Measures the acceleration of the 5-day price trend.
#     # High (+) = Accelerating; Low (-) = Decelerating/Topping out.
#     "Convexity": lambda obs: obs.convexity,
#     ############################
# }


# METRIC_REGISTRY = {
#     "Log Price Gain": lambda obs: QuantUtils.calculate_gain(obs.lookback_close),
#     "Sharpe (TRP)": lambda obs: QuantUtils.calculate_sharpe_vol(
#         obs.lookback_returns, obs.trp
#     ),
#     "Info Ratio (Stdev_Alpha, 63d)": lambda obs: obs.ir_63,
#     "Consistency (WinRate 5d)": lambda obs: obs.consistency,
#     "Oversold (-RSI)": lambda obs: -obs.rsi,
#     # 7. REGIME DETECTION (The "Texture" Pillar)
#     # High (+) = Momentum is 'sticky'. Low (-) = Momentum is 'shaky/reverting'.
#     "Return Autocorr (15d)": lambda obs: obs.autocorr_15,
#     # 8. MEAN REVERSION / EXTREMES (The "Boundary" Pillar)
#     # High = Overbought/Breakout; Low = Oversold/Value.
#     "Range Position (20d)": lambda obs: obs.range_pos_20,
#     # 9. VOLUME CONFIRMATION (The "Fuel" Pillar)
#     # Measures the gap between price direction and volume flow.
#     # We Z-score both within the registry to ensure they are on the same scale.
#     "OBV Divergence (5d)": lambda obs: (
#         (obs.slope_v_5 - obs.slope_v_5.mean()) / obs.slope_v_5.std()
#     )
#     - ((obs.slope_p_5 - obs.slope_p_5.mean()) / obs.slope_p_5.std()),
#     # 10. MOMENTUM PHYSICS (The "Exhaustion" Pillar)
#     # Measures the acceleration of the 5-day price trend.
#     # High (+) = Accelerating; Low (-) = Decelerating/Topping out.
#     "Convexity": lambda obs: obs.convexity,
#     ############################
# }


# METRIC_REGISTRY = {
#     "Log Price Gain": lambda obs: QuantUtils.calculate_gain(obs.lookback_close),
#     "Sharpe (TRP)": lambda obs: QuantUtils.calculate_sharpe_vol(
#         obs.lookback_returns, obs.trp
#     ),
#     "Momentum (21d)": lambda obs: obs.mom_21,
#     "Info Ratio (Stdev_Alpha, 63d)": lambda obs: obs.ir_63,
#     "Oversold (-RSI)": lambda obs: -obs.rsi,
#     "Dip Buyer (Drawdown -dd_21)": lambda obs: -obs.dd_21,
#     "Low Volatility (-ATRP)": lambda obs: -obs.atrp,
#     # 7. REGIME DETECTION (The "Texture" Pillar)
#     # High (+) = Momentum is 'sticky'. Low (-) = Momentum is 'shaky/reverting'.
#     "Return Autocorr (15d)": lambda obs: obs.autocorr_15,
#     # 8. MEAN REVERSION / EXTREMES (The "Boundary" Pillar)
#     # High = Overbought/Breakout; Low = Oversold/Value.
#     "Range Position (20d)": lambda obs: obs.range_pos_20,
#     # 9. VOLUME CONFIRMATION (The "Fuel" Pillar)
#     # Measures the gap between price direction and volume flow.
#     # We Z-score both within the registry to ensure they are on the same scale.
#     "OBV Divergence (5d)": lambda obs: (
#         (obs.slope_v_5 - obs.slope_v_5.mean()) / obs.slope_v_5.std()
#     )
#     - ((obs.slope_p_5 - obs.slope_p_5.mean()) / obs.slope_p_5.std()),
#     # 10. MOMENTUM PHYSICS (The "Exhaustion" Pillar)
#     # Measures the acceleration of the 5-day price trend.
#     # High (+) = Accelerating; Low (-) = Decelerating/Topping out.
#     "Convexity": lambda obs: obs.convexity,
# }


STRATEGY_REGISTRY: Dict[str, MetricBlueprint] = {
    # --- 1. RETURNS PILLAR ---
    "Log Price Gain": MetricBlueprint(
        name="Log Price Gain",
        category="Returns",
        regime="Trend",
        description="Pure logarithmic return over the lookback period.",
        agent_hint="The raw gas pedal. High values = strong past performance. Check for momentum persistence.",
        formula=lambda obs: QuantUtils.calculate_gain(obs.lookback_close),
    ),
    "Sharpe (TRP)": MetricBlueprint(
        name="Sharpe (TRP)",
        category="Risk-Adjusted",
        regime="Efficiency",
        description="Sharpe ratio of the Total Return Premium (TRP).",
        agent_hint="Risk-adjusted reward. High values indicate stable growth; low values suggest erratic price action.",
        formula=lambda obs: QuantUtils.calculate_sharpe_vol(
            obs.lookback_returns, obs.trp
        ),
    ),
    # --- 2. MOMENTUM PILLAR ---
    "Momentum (21d)": MetricBlueprint(
        name="Momentum (21d)",
        category="Momentum",
        regime="Trend",
        description="Standard 21-day price momentum.",
        agent_hint="The 'velocity' of price. Useful for capturing medium-term trend extensions.",
        formula=lambda obs: obs.mom_21,
    ),
    "Info Ratio (63d)": MetricBlueprint(
        name="Info Ratio (Stdev_Alpha, 63d)",
        category="Risk-Adjusted",
        regime="Trend Quality",
        description="Information ratio over a 63-day window.",
        agent_hint="Measures the consistency of the alpha. Look for values > 0 to confirm high-quality trends.",
        formula=lambda obs: obs.ir_63,
    ),
    # --- 3. MEAN REVERSION PILLAR ---
    "Oversold (-RSI)": MetricBlueprint(
        name="Oversold (-RSI)",
        category="Mean Reversion",
        regime="Boundary",
        description="Inverse Relative Strength Index.",
        agent_hint="High values = Oversold. Low values = Overbought. Useful for timing entries in ranging markets.",
        formula=lambda obs: -obs.rsi,
    ),
    "Dip Buyer (-dd_21)": MetricBlueprint(
        name="Dip Buyer (Drawdown -dd_21)",
        category="Mean Reversion",
        regime="Boundary",
        description="Inverse 21-day drawdown.",
        agent_hint="High values = Significant price drop from peak. Use to identify 'Buy the Dip' opportunities.",
        formula=lambda obs: -obs.dd_21,
    ),
    # --- 4. VOLATILITY PILLAR ---
    "Low Volatility (-ATRP)": MetricBlueprint(
        name="Low Volatility (-ATRP)",
        category="Volatility",
        regime="Risk Management",
        description="Inverse Average True Range Percentage.",
        agent_hint="High values = Compressed volatility (Quiet market). Low values = Explosive volatility.",
        formula=lambda obs: -obs.atrp,
    ),
    # --- 5. REGIME/TEXTURE PILLAR ---
    "Return Autocorr (15d)": MetricBlueprint(
        name="Return Autocorr (15d)",
        category="Microstructure",
        regime="Regime Detection",
        description="Measures price memory (Persistence vs. Reversion).",
        agent_hint="High (+) = Trend is sticky. High (-) = Mean reverting 'choppy' texture.",
        formula=lambda obs: obs.autocorr_15,
    ),
    "Range Position (20d)": MetricBlueprint(
        name="Range Position (20d)",
        category="Mean Reversion",
        regime="Boundary",
        description="Where the price sits relative to its 20-day High/Low.",
        agent_hint="1.0 = at 20d High (Breakout zone); 0.0 = at 20d Low (Support zone).",
        formula=lambda obs: obs.range_pos_20,
    ),
    # --- 6. FUEL/VOLUME PILLAR ---
    "OBV Divergence (5d)": MetricBlueprint(
        name="OBV Divergence (5d)",
        category="Volume/Fuel",
        regime="Confirmation",
        description="Measures Z-scored gap between volume flow and price direction.",
        agent_hint="Positive values indicate accumulation (Volume leading price); negative indicate distribution.",
        formula=lambda obs: (
            (obs.slope_v_5 - obs.slope_v_5.mean()) / obs.slope_v_5.std().replace(0, 1)
        )
        - ((obs.slope_p_5 - obs.slope_p_5.mean()) / obs.slope_p_5.std().replace(0, 1)),
    ),
    # --- 7. PHYSICS PILLAR ---
    "Convexity": MetricBlueprint(
        name="Convexity",
        category="Physics",
        regime="Momentum Quality",
        description="The 2nd derivative of price momentum (Acceleration).",
        agent_hint="Trend is healthy and accelerating when > 0. Trend is exhausting when < 0.",
        formula=lambda obs: obs.convexity,
    ),
}
