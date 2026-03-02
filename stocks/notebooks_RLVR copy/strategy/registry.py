import numpy as np

from core.quant import QuantUtils


METRIC_REGISTRY = {
    # Dot notation: obs.lookback_close instead of obs.get("lookback_close")
    "Price Gain": lambda obs: QuantUtils.calculate_gain(obs.lookback_close),
    "Sharpe": lambda obs: QuantUtils.calculate_sharpe(obs.lookback_returns),
    "Sharpe (ATRP)": lambda obs: QuantUtils.calculate_sharpe_vol(
        obs.lookback_returns, obs.atrp
    ),
    "Sharpe (TRP)": lambda obs: QuantUtils.calculate_sharpe_vol(
        obs.lookback_returns, obs.trp
    ),
    "Momentum (21d)": lambda obs: obs.mom_21,
    "Info Ratio (Stdev_Alpha, 63d)": lambda obs: obs.ir_63,
    "Consistency (WinRate 5d)": lambda obs: obs.consistency,
    "Oversold (-RSI)": lambda obs: -obs.rsi,
    "Dip Buyer (Drawdown -dd_21)": lambda obs: -obs.dd_21,
    "Low Volatility (-ATRP)": lambda obs: -obs.atrp,
    # Example using the NEW macro data we added to the dataclass!
    "VIX Filtered Momentum": lambda obs: (
        obs.mom_21 if obs.macro_vix_ratio < 1.0 else obs.mom_21 * 0
    ),
}
