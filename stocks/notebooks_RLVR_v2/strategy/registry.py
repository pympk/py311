from typing import Dict

from core.quant import QuantUtils
from core.contracts import MetricBlueprint
from core.settings import TradingConfig


def get_strategy_registry(config: TradingConfig) -> Dict[str, MetricBlueprint]:
    S_PARAMS = config.strategy_params

    return {
        "Log Price Gain": MetricBlueprint(
            name="Log Price Gain",
            category="Returns",
            regime="Trend",
            description="Natural log return of lookback window.",
            agent_hint="Primary momentum filter. Use Z-scores to identify 'Normal' vs 'Extreme' growth.",
            intervention_trigger=f"LONG if Value > {S_PARAMS.standard_confidence}std & Autocorr > 0.15; FLAT if Value < -{S_PARAMS.standard_confidence}std or Convexity < 0",
            scaling_type="Z-Score",
            formula=lambda obs: QuantUtils.calculate_gain(obs.lookback_close),
        ),
        "Sharpe (TRP)": MetricBlueprint(
            name="Sharpe (TRP)",
            category="Risk-Adjusted",
            regime="Efficiency",
            description="Risk-adjusted efficiency of the Total Return Premium.",
            agent_hint="The 'Quality' dial. High values suggest stable, institutional-led trends.",
            intervention_trigger="SIZE = clip(Sharpe, 0, 3) / 2.0. If Sharpe < 0.5, reduce position by 50%.",
            formula=lambda obs: QuantUtils.calc_sharpe_cross_section(
                obs.lookback_returns, obs.trp  # Returns (DF) + TRP (Series)
            ),
        ),
        "Momentum (21d)": MetricBlueprint(
            name="Momentum (21d)",
            category="Momentum",
            regime="Trend",
            description="Standard 1-month momentum factor.",
            agent_hint=f"Use to rank assets. Avoid buying when Momentum is over-extended (>{S_PARAMS.extreme_confidence}std).",
            intervention_trigger=f"CONFIRM LONG if 21d > 63d Mean; AVOID if Value > {S_PARAMS.extreme_confidence}std (Parabolic Risk).",
            formula=lambda obs: obs.mom_21,
        ),
        "Info Ratio (63d)": MetricBlueprint(
            name="Info Ratio (63d)",
            category="Alpha",
            regime="Trend Quality",
            description="Alpha consistency over a quarterly window.",
            agent_hint="The 'Gatekeeper'. If IR is low, the trend is likely noise/random walk.",
            intervention_trigger="GATING: Only allow 'Trend' Pillar weight > 0.2 if Info Ratio > 0.5.",
            formula=lambda obs: obs.ir_63,
        ),
        "Oversold (-RSI)": MetricBlueprint(
            name="Oversold (-RSI)",
            category="Mean Reversion",
            regime="Contrarian",
            description="Inverse RSI(14). Scaled between -1.0 and 1.0.",
            agent_hint="Higher is more oversold. Look for the 'Hook' (Convexity > 0) to time entry.",
            intervention_trigger=f"BUY if Value > {100-S_PARAMS.rsi_oversold} AND Convexity > 0.2; SELL/FLAT if Value < {100-S_PARAMS.rsi_overbought}.",
            scaling_type="RSI",
            formula=lambda obs: -obs.rsi,
        ),
        "Dip Buyer (-dd_21)": MetricBlueprint(
            name="Dip Buyer (-dd_21)",
            category="Mean Reversion",
            regime="Contrarian",
            description="Inverse 21-day drawdown. High = Deep pullback.",
            agent_hint="Best used when the structural trend is still positive (Autocorr > 0.15).",
            intervention_trigger=f"BUY DIP if Value > {S_PARAMS.strong_confidence}std AND Autocorr_15 > 0.2 (Structural Trend).",
            formula=lambda obs: -obs.dd_21,
        ),
        "Range Position (20d)": MetricBlueprint(
            name="Range Position (20d)",
            category="Mean Reversion",
            regime="Boundary",
            description="Where price sits in 20-day High/Low range (0.0 to 1.0).",
            agent_hint="The 'Decision Fork'. Breakout at 0.8+, Support at 0.2-.",
            intervention_trigger=f"Value > {S_PARAMS.range_high}: LONG only if OBV > {S_PARAMS.standard_confidence}std; Value < {S_PARAMS.range_low}: LONG only if OBV < -{S_PARAMS.standard_confidence}std.",
            formula=lambda obs: obs.range_pos_20,
        ),
        "Return Autocorr (15d)": MetricBlueprint(
            name="Return Autocorr (15d)",
            category="Regime",
            regime="Market State",
            description="Measures price memory (Persistence vs. Mean Reversion).",
            agent_hint="THE MASTER SWITCH. Determines which other features to trust.",
            intervention_trigger="Bias 'Trend' if > 0.15; Bias 'Reversion' if < -0.15; Else prioritize 'Cash'.",
            scaling_type="None",
            formula=lambda obs: obs.autocorr_15,
        ),
        "Low Volatility (-ATRP)": MetricBlueprint(
            name="Low Volatility (-ATRP)",
            category="Volatility",
            regime="Risk Filter",
            description="Inverse ATR Percentage. High = Quiet market.",
            agent_hint="Standardized volatility. 0 = Market Average.",
            intervention_trigger=f"RISK OFF if Value < -2.0std; BREAKOUT WATCH if Value > {S_PARAMS.strong_confidence}std.",
            scaling_type="Z-Score",
            formula=lambda obs: -obs.atrp,
        ),
        "Slope_P_5_Z": MetricBlueprint(
            name="Slope_P_5_Z",
            category="Price/Velocity",
            regime="Confirmation",
            description="Temporally Z-scored price slope over 5 days.",
            agent_hint="Detects velocity of price trend.",
            intervention_trigger=f"CONFIRM TREND if Value > {S_PARAMS.standard_confidence}std; FLAT/REVERSAL if Value < -{S_PARAMS.standard_confidence}std.",
            scaling_type="Z-Score",
            formula=lambda obs: obs.slope_p_5_z,
        ),
        "Slope_V_5_Z": MetricBlueprint(
            name="Slope_V_5_Z",
            category="Volume/Fuel",
            regime="Confirmation",
            description="Temporally Z-scored volume slope over 5 days.",
            agent_hint="Detects volume flow velocity.",
            intervention_trigger=f"ACCUMULATION if Value > {S_PARAMS.standard_confidence}std; DISTRIBUTION if Value < -{S_PARAMS.standard_confidence}std.",
            scaling_type="Z-Score",
            formula=lambda obs: obs.slope_v_5_z,
        ),
        "Convexity": MetricBlueprint(
            name="Convexity",
            category="Physics",
            regime="Acceleration",
            description="Second derivative of price. Curvature of the trend.",
            agent_hint="The 'Golden Exit'. Trend is healthy when > 0, exhausting when < 0.",
            intervention_trigger=f"EXIT LONG if Value < {S_PARAMS.convexity_exit} (Deceleration). FRONT-RUN THE REVERSAL.",
            scaling_type="Z-Score",
            formula=lambda obs: obs.convexity,
        ),
    }
