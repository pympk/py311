from core.quant import QuantUtils


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


# METRIC_REGISTRY = {
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


METRIC_REGISTRY = {
    "Log Price Gain": lambda obs: QuantUtils.calculate_gain(obs.lookback_close),
    "Sharpe (TRP)": lambda obs: QuantUtils.calculate_sharpe_vol(
        obs.lookback_returns, obs.trp
    ),
    "Info Ratio (Stdev_Alpha, 63d)": lambda obs: obs.ir_63,
    "Consistency (WinRate 5d)": lambda obs: obs.consistency,
    "Oversold (-RSI)": lambda obs: -obs.rsi,
    # 7. REGIME DETECTION (The "Texture" Pillar)
    # High (+) = Momentum is 'sticky'. Low (-) = Momentum is 'shaky/reverting'.
    "Return Autocorr (15d)": lambda obs: obs.autocorr_15,
    # 8. MEAN REVERSION / EXTREMES (The "Boundary" Pillar)
    # High = Overbought/Breakout; Low = Oversold/Value.
    "Range Position (20d)": lambda obs: obs.range_pos_20,
    # 9. VOLUME CONFIRMATION (The "Fuel" Pillar)
    # Measures the gap between price direction and volume flow.
    # We Z-score both within the registry to ensure they are on the same scale.
    "OBV Divergence (5d)": lambda obs: (
        (obs.slope_v_5 - obs.slope_v_5.mean()) / obs.slope_v_5.std()
    )
    - ((obs.slope_p_5 - obs.slope_p_5.mean()) / obs.slope_p_5.std()),
    # 10. MOMENTUM PHYSICS (The "Exhaustion" Pillar)
    # Measures the acceleration of the 5-day price trend.
    # High (+) = Accelerating; Low (-) = Decelerating/Topping out.
    "Convexity": lambda obs: obs.convexity,
    ############################
}
