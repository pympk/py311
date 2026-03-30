from core.quant import QuantUtils


# METRIC_REGISTRY = {
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


METRIC_REGISTRY = {
    # 1. RISK-ADJUSTED TREND (The "Smart" Momentum)
    # Selected over standard Sharpe because ATR-adjustment is more regime-robust.
    "Sharpe (ATRP)": lambda obs: QuantUtils.calculate_sharpe_vol(
        obs.lookback_returns, obs.atrp
    ),
    # 2. QUALITY & STEADINESS (The "Consistency" Pillar)
    # Only ~18% overlap with Sharpe; captures stocks with high win-rates, not just big moves.
    "Consistency (WinRate 5d)": lambda obs: obs.consistency,
    # 3. MEAN REVERSION (The "Anti-Momentum" Pillar)
    # 0% overlap with Sharpe; finds oversold stocks that momentum misses.
    "Oversold (-RSI)": lambda obs: -obs.rsi,
    # 4. RECOVERY ALPHA (The "Dip Buyer" Pillar)
    # Captures stocks recovering from local drawdowns; distinct from RSI.
    "Dip Buyer (Drawdown -dd_21)": lambda obs: -obs.dd_21,
    # 5. VOLATILITY PROTECTION (The "Quiet" Pillar)
    # Picks the "low-noise" stocks; completely orthogonal to pure trend.
    "Low Volatility (-ATRP)": lambda obs: -obs.atrp,
    # 6. PURE TREND (The "Macro-Aware" Pillar)
    # Replaces "Momentum (21d)" by adding the VIX safety switch logic.
    "VIX Filtered Momentum": lambda obs: (
        obs.mom_21 if obs.macro_vix_ratio < 1.0 else obs.mom_21 * 0
    ),
}
