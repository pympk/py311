from typing import Dict

from core.quant import QuantUtils
from core.contracts import MetricBlueprint
from core.settings import TradingConfig


def get_strategy_registry(config: TradingConfig) -> Dict[str, MetricBlueprint]:
    S_PARAMS = config.strategy_params

    return {
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
        "Sharpe (TRP)": MetricBlueprint(
            name="Sharpe (TRP)",
            category="Risk-Adjusted",
            regime="Efficiency",
            description="Risk-adjusted efficiency of the Total Return Premium.",
            agent_hint="The 'Quality' dial. High values suggest stable, institutional-led trends.",
            intervention_trigger="SIZE = clip(Sharpe, 0, 3) / 2.0. If Sharpe < 0.5, reduce position by 50%.",
            formula=lambda obs: QuantUtils.calc_sharpe_cross_section(
                obs.lookback_returns, obs.trp
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
            agent_hint="Higher is more oversold.",
            intervention_trigger=f"BUY if Value > {100-S_PARAMS.rsi_oversold}; SELL/FLAT if Value < {100-S_PARAMS.rsi_overbought}.",
            scaling_type="RSI",
            formula=lambda obs: -obs.rsi,
        ),
        "Dip Buyer (-dd_21)": MetricBlueprint(
            name="Dip Buyer (-dd_21)",
            category="Mean Reversion",
            regime="Contrarian",
            description="Inverse 21-day drawdown. High = Deep pullback.",
            agent_hint="Best used when the structural trend is still positive.",
            intervention_trigger=f"BUY DIP if Value > {S_PARAMS.strong_confidence}std.",
            formula=lambda obs: -obs.dd_21,
        ),
        "Trend Quality (63d)": MetricBlueprint(
            name="Trend Quality (63d)",
            category="Trend",
            regime="Structural Quality",
            description="Smoothness/Linearity of price trend via log-price time correlation over 63 days.",
            agent_hint="Separates steady institutional compounders from erratic price spikes. Bounded [-1, 1].",
            intervention_trigger=f"CONFIRM LONG if Value > 0.60; REJECT if Value < 0.0 (Choppy/Declining).",
            scaling_type="Z-Score",
            formula=lambda obs: obs.trend_r2_63,
        ),
        "Downside Risk (-SemiDev_63)": MetricBlueprint(
            name="Downside Risk (-SemiDev_63)",
            category="Risk-Adjusted",
            regime="Left-Tail Filter",
            description="Inverse Downside Semi-Deviation over 63 days relative to 0.0.",
            agent_hint="Pure crash-risk dampener. Penalizes left-tail dispersion without clipping right-tail gains.",
            intervention_trigger=f"PREFER if Value > {S_PARAMS.standard_confidence}std; CUT if Value < -{S_PARAMS.strong_confidence}std.",
            scaling_type="Z-Score",
            formula=lambda obs: -obs.semidev_63,
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
        "Momentum (63d)": MetricBlueprint(
            name="Momentum (63d)",
            category="Momentum",
            regime="Trend",
            description="Quarterly intermediate price momentum (3-month cumulative return).",
            agent_hint="Primary intermediate trend filter. Persistent cross-sectional leader with ~35d half-life.",
            intervention_trigger=f"LONG if Value > {S_PARAMS.standard_confidence}std; AVOID if Value < -{S_PARAMS.standard_confidence}std.",
            scaling_type="Z-Score",
            formula=lambda obs: obs.mom_63,
        ),
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
