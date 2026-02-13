import numpy as np

from core.quant import QuantUtils


METRIC_REGISTRY = {
    "Price Gain": lambda obs: QuantUtils.calculate_gain(
        obs.get("lookback_close", np.nan)
    ),
    "Sharpe": lambda obs: QuantUtils.calculate_sharpe(
        obs.get("lookback_returns", np.nan)
    ),
    "Sharpe (ATRP)": lambda obs: QuantUtils.calculate_sharpe_vol(
        obs.get("lookback_returns", np.nan), obs.get("atrp", np.nan)
    ),
    "Sharpe (TRP)": lambda obs: QuantUtils.calculate_sharpe_vol(
        obs.get("lookback_returns", np.nan), obs.get("trp", np.nan)
    ),
    "Momentum (21d)": lambda obs: obs.get("mom_21", np.nan),
    "Info Ratio (Stdev_Alpha, 63d)": lambda obs: obs.get("ir_63", np.nan),
    "Consistency (WinRate 5d)": lambda obs: obs.get("consistency", np.nan),
    "Oversold (-RSI)": lambda obs: -obs.get("rsi", np.nan),
    "Dip Buyer (Drawdown -dd_21)": lambda obs: -obs.get("dd_21", np.nan),
    "Low Volatility (-ATRP)": lambda obs: -obs.get("atrp", np.nan),
}
