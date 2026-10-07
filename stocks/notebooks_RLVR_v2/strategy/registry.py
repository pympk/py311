from typing import Dict

from core.contracts import MetricBlueprint
from core.settings import TradingConfig


def get_strategy_registry(config: TradingConfig) -> Dict[str, MetricBlueprint]:
    S_PARAMS = config.strategy_params

    return {
        # #1: Secular Trend Anchor
        "Momentum (12-1m)": MetricBlueprint(
            name="Momentum (12-1m)",
            category="Momentum",
            regime="Structural Trend",
            description="Fama-French 12-1 momentum (t-252 to t-21 return, stripping 1m reversal).",
            agent_hint="Long-term institutional drift anchor with ~120d half-life. Resists short-term noise.",
            intervention_trigger=f"LONG if Value > {S_PARAMS.standard_confidence}std; AVOID if Value < -{S_PARAMS.standard_confidence}std",
            scaling_type="Z-Score",
            formula=lambda obs: obs.mom_252_21,
        ),
        # #2: Idiosyncratic Alpha (Replaced Sharpe TRP)
        "Residual Momentum (126d)": MetricBlueprint(
            name="Residual Momentum (126d)",
            category="Idiosyncratic Alpha",
            regime="Beta-Decontaminated Alpha",
            description="Beta-decontaminated semi-annual cumulative return standardized by idiosyncratic residual volatility.",
            agent_hint="Pure idiosyncratic alpha. Strips market exposure to isolate genuine stock-specific compounders.",
            intervention_trigger=f"LONG if Value > {S_PARAMS.standard_confidence}std; AVOID if Value < -{S_PARAMS.standard_confidence}std",
            scaling_type="Z-Score",
            formula=lambda obs: obs.res_mom_126,
        ),
        # #3: Overhead Supply Clearance (Replaced Mom 21d)
        "Range Position (52w High)": MetricBlueprint(
            name="Range Position (52w High)",
            category="Anchoring Quality",
            regime="Breakout Quality",
            description="Proximity to 52-week high (P_t / max(P_252)). Bounded in (0, 1].",
            agent_hint="Clearance of overhead resistance. Stocks trading near 52w highs suffer zero trapped overhead supply.",
            intervention_trigger="LONG if Value > 0.90; AVOID if Value < 0.70",
            scaling_type="MinMax",
            formula=lambda obs: obs.range_pos_52w,
        ),
        # #4: Alpha Consistency
        "Info Ratio (63d)": MetricBlueprint(
            name="Info Ratio (63d)",
            category="Alpha",
            regime="Trend Quality",
            description="Alpha consistency over a quarterly window.",
            agent_hint="The 'Gatekeeper'. If IR is low, the trend is likely noise/random walk.",
            intervention_trigger="GATING: Only allow 'Trend' Pillar weight > 0.2 if Info Ratio > 0.5.",
            formula=lambda obs: obs.ir_63,
        ),
        # #5: Mean Reversion Contrarian Pullback
        "Oversold (-RSI)": MetricBlueprint(
            name="Oversold (-RSI)",
            category="Mean Reversion",
            regime="Contrarian",
            description="Inverse RSI(14). Scaled between -1.0 and 1.0.",
            agent_hint="Higher is more oversold.",
            intervention_trigger=f"BUY if Value > {100-S_PARAMS.rsi_oversold}; SELL/FLAT if Value < {100-S_PARAMS.rsi_overbought}.",
            scaling_type="RSI",
            formula=lambda obs: -obs.rsi,
        ),
        # #6: Tactical Drawdown Depth
        "Dip Buyer (-dd_21)": MetricBlueprint(
            name="Dip Buyer (-dd_21)",
            category="Mean Reversion",
            regime="Contrarian",
            description="Inverse 21-day drawdown. High = Deep pullback.",
            agent_hint="Best used when the structural trend is still positive.",
            intervention_trigger=f"BUY DIP if Value > {S_PARAMS.strong_confidence}std.",
            formula=lambda obs: -obs.dd_21,
        ),
        # #7: Structural Linearity
        "Trend Quality (63d)": MetricBlueprint(
            name="Trend Quality (63d)",
            category="Trend",
            regime="Structural Quality",
            description="Smoothness/Linearity of price trend via log-price time correlation over 63 days.",
            agent_hint="Separates steady institutional compounders from erratic price spikes. Bounded [-1, 1].",
            intervention_trigger="CONFIRM LONG if Value > 0.60; REJECT if Value < 0.0 (Choppy/Declining).",
            scaling_type="Z-Score",
            formula=lambda obs: obs.trend_r2_63,
        ),
        # #8: Systematic Crash Filter (Replaced SemiDev 63d)
        "Downside Beta (-Beta_Down_63)": MetricBlueprint(
            name="Downside Beta (-Beta_Down_63)",
            category="Asymmetric Tail Defense",
            regime="Left-Tail Filter",
            description="Inverse Downside Beta over 63 days relative to benchmark on market down-days (r_bm < 0).",
            agent_hint="Pure crash-risk dampener. Penalizes stocks that amplify market downturns while preserving upside participation.",
            intervention_trigger="PREFER if Value > -0.80; AVOID if Value < -1.20",
            scaling_type="Z-Score",
            formula=lambda obs: -obs.beta_down_63,
        ),
        # #9: Absolute Noise Dampener
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
        # #10: Fractal Efficiency (Replaced Mom 63d)
        "Efficiency Ratio (ER_63)": MetricBlueprint(
            name="Efficiency Ratio (ER_63)",
            category="Fractal Efficiency",
            regime="Signal-to-Noise",
            description="Kaufman Efficiency Ratio over 63 days (|P_t - P_{t-63}| / sum(|Delta P|)). Bounded in [0, 1].",
            agent_hint="Trend efficiency sensor. 1.0 = smooth monotonic trend, 0.0 = pure noise/whipsaw.",
            intervention_trigger="CONFIRM LONG if Value > 0.35; REJECT if Value < 0.15",
            scaling_type="MinMax",
            formula=lambda obs: obs.er_63,
        ),
        # #11: Intermediate Trend Anchor
        "Momentum (126d)": MetricBlueprint(
            name="Momentum (126d)",
            category="Momentum",
            regime="Trend",
            description="Half-year intermediate price momentum (6-month academic gold standard).",
            agent_hint="Structural trend anchor. Immune to T+1 execution lag and short-term noise.",
            intervention_trigger=f"CONFIRM LONG if Value > {S_PARAMS.standard_confidence}std; REDUCE if Value < -{S_PARAMS.standard_confidence}std.",
            scaling_type="Z-Score",
            formula=lambda obs: obs.mom_126,
        ),
        # #12: Idiosyncratic Quality Shelter
        "Residual Low-Vol (63d)": MetricBlueprint(
            name="Residual Low-Vol (63d)",
            category="Risk-Adjusted",
            regime="Risk Filter",
            description="Inverse idiosyncratic volatility relative to benchmark over 63 days.",
            agent_hint="Idiosyncratic quality shelter. High values signify quiet, low-tail-risk stocks that compound steadily.",
            intervention_trigger=f"PREFER if Value > {S_PARAMS.standard_confidence}std; PENALIZE if Value < -{S_PARAMS.strong_confidence}std.",
            scaling_type="Z-Score",
            formula=lambda obs: -obs.ivol_63,
        ),
    }
