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


STRATEGY_REGISTRY: Dict[str, MetricBlueprint] = {
    # """
    # AI Studio
    # **Verdict:**
    # The `intervention_trigger` **helps discovery** by defining the "Standard Path." Novelty is then defined as **"Optimized Deviation from the Standard Path."** Without the trigger, the agent has no "standard" to improve upon.
    # **Recommendation:** Keep the triggers. Use them as **Features (State)** and **Priors (Early Reward)**, but **never as Hard Action Masks.**
    # """
    # --- PILLAR 1: THE TREND ENGINE (DIRECTION) ---
    "Log Price Gain": MetricBlueprint(
        name="Log Price Gain",
        category="Returns",
        regime="Trend",
        description="Natural log return of lookback window.",
        agent_hint="Primary momentum filter. Use Z-scores to identify 'Normal' vs 'Extreme' growth.",
        intervention_trigger="LONG if Value > 1.0std & Autocorr > 0.15; FLAT if Value < -1.0std or Convexity < 0",
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
        formula=lambda obs: QuantUtils.calculate_sharpe_vol(
            obs.lookback_returns, obs.trp
        ),
    ),
    "Momentum (21d)": MetricBlueprint(
        name="Momentum (21d)",
        category="Momentum",
        regime="Trend",
        description="Standard 1-month momentum factor.",
        agent_hint="Use to rank assets. Avoid buying when Momentum is over-extended (>2.5std).",
        intervention_trigger="CONFIRM LONG if 21d > 63d Mean; AVOID if Value > 2.5std (Parabolic Risk).",
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
    # --- PILLAR 2: MEAN REVERSION (THE RUBBER BAND) ---
    "Oversold (-RSI)": MetricBlueprint(
        name="Oversold (-RSI)",
        category="Mean Reversion",
        regime="Contrarian",
        description="Inverse RSI(14). Transforms 0-100 into a 'Pressure' gauge.",
        agent_hint="Higher is more oversold. Look for the 'Hook' (Convexity > 0) to time entry.",
        intervention_trigger="BUY if Value > 70 AND Convexity > 0.2; SELL/FLAT if Value < 30.",
        scaling_type="RSI",
        formula=lambda obs: -obs.rsi,
    ),
    "Dip Buyer (-dd_21)": MetricBlueprint(
        name="Dip Buyer (-dd_21)",
        category="Mean Reversion",
        regime="Contrarian",
        description="Inverse 21-day drawdown. High = Deep pullback.",
        agent_hint="Best used when the structural trend is still positive (Autocorr > 0.15).",
        intervention_trigger="BUY DIP if Value > 1.5std AND Autocorr_15 > 0.2 (Structural Trend).",
        formula=lambda obs: -obs.dd_21,
    ),
    "Range Position (20d)": MetricBlueprint(
        name="Range Position (20d)",
        category="Mean Reversion",
        regime="Boundary",
        description="Where price sits in 20-day High/Low range (0.0 to 1.0).",
        agent_hint="The 'Decision Fork'. Breakout at 0.8+, Support at 0.2-.",
        intervention_trigger="Value > 0.8: LONG only if OBV > 1.0std; Value < 0.2: LONG only if OBV < -1.0std.",
        formula=lambda obs: obs.range_pos_20,
    ),
    # --- PILLAR 3: REGIME DETECTION (THE MASTER SWITCH) ---
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
    # --- PILLAR 4: VOLATILITY (RISK MANAGER) ---
    "Low Volatility (-ATRP)": MetricBlueprint(
        name="Low Volatility (-ATRP)",
        category="Volatility",
        regime="Risk Filter",
        description="Inverse ATR Percentage. High = Quiet market.",
        agent_hint="Volatility compression often precedes explosive moves. Watch for the breakout.",
        intervention_trigger="RISK OFF if Value < -2.0std (Spike); BREAKOUT WATCH if Value > 1.5std (Compression).",
        formula=lambda obs: -obs.atrp,
    ),
    # --- PILLAR 5: VOLUME FLOW (THE LIE DETECTOR) ---
    "OBV Divergence (5d)": MetricBlueprint(
        name="OBV Divergence (5d)",
        category="Volume/Fuel",
        regime="Confirmation",
        description="Z-scored gap between volume flow and price trend.",
        agent_hint="Detects smart money accumulation/distribution.",
        intervention_trigger="INVALIDATE Longs if Price Trend (+) but OBV < -1.0std.",
        scaling_type="Z-Score",
        formula=lambda obs: (
            # Corrected: Calculate std first, then handle the zero case
            (obs.slope_v_5 - obs.slope_v_5.mean())
            / (obs.slope_v_5.std() if obs.slope_v_5.std() != 0 else 1.0)
        )
        - (
            (obs.slope_p_5 - obs.slope_p_5.mean())
            / (obs.slope_p_5.std() if obs.slope_p_5.std() != 0 else 1.0)
        ),
    ),
    # --- PILLAR 6: PHYSICS (EXHAUSTION DETECTOR) ---
    "Convexity": MetricBlueprint(
        name="Convexity",
        category="Physics",
        regime="Acceleration",
        description="Second derivative of price. Curvature of the trend.",
        agent_hint="The 'Golden Exit'. Trend is healthy when > 0, exhausting when < 0.",
        intervention_trigger="EXIT LONG if Value < -0.7 (Deceleration). FRONT-RUN THE REVERSAL.",
        scaling_type="Z-Score",
        formula=lambda obs: obs.convexity,
    ),
}
