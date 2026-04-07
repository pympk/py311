import pandas as pd

import ipywidgets as widgets
import plotly.graph_objects as go


from plotly.subplots import make_subplots
from IPython.display import display
from typing import Optional, Protocol
from dataclasses import dataclass
from enum import IntEnum

# Internal Imports
from core.settings import GLOBAL_SETTINGS
from core.contracts import EngineInput, EngineOutput, FilterPack
from strategy.registry import STRATEGY_REGISTRY
from core.result import HeadlessReporter

# ============================================================================
# 1. TRACE REGISTRY (Eliminates Magic Numbers)
# ============================================================================


class TraceId(IntEnum):
    """Single source of truth for trace indices. Add new traces here."""

    # Row 1: Price Action (0-50)
    TICKERS_START = 0
    TICKERS_END = 50  # Exclusive
    BENCHMARK = 50
    PORTFOLIO = 51

    # Row 2-4: Macro Indicators
    TREND = 52
    TREND_VELOCITY = 53
    VIX_ZSCORE = 54
    HY_SPREAD_Z = 55  # New
    YIELD_CURVE_Z = 56  # New
    TOTAL_TRACES = 57  # Updated


# ============================================================================
# 2. PROTOCOLS (Define Contracts)
# ============================================================================


class ChartRenderer(Protocol):
    """Anything that can update a Plotly figure."""

    def update(
        self, figure: go.FigureWidget, data: "EngineOutput", inputs: "EngineInput"
    ) -> None: ...


class ReportRenderer(Protocol):
    """Anything that generates text/HTML output."""

    def render(self, data: "EngineOutput", inputs: "EngineInput") -> widgets.Widget: ...


# ============================================================================
# 3. MACRO VISUALIZER (Extracted Complex Logic)
# ============================================================================


@dataclass
class VolatilityRegime:
    label: str
    color: str
    threshold_low: Optional[float] = None
    threshold_high: Optional[float] = None

    def contains(self, ratio: float) -> bool:
        low_ok = self.threshold_low is None or ratio >= self.threshold_low
        high_ok = self.threshold_high is None or ratio < self.threshold_high
        return low_ok and high_ok


class MacroVisualizer:
    """Handles regime detection, shading, and macro subplot updates."""

    REGIMES = [
        VolatilityRegime("STABILITY", "rgba(0, 255, 0, 0.08)", None, 0.9),
        VolatilityRegime("TRANSITION", "rgba(128, 128, 128, 0.05)", 0.9, 1.0),
        VolatilityRegime("SYSTEMIC SHOCK", "rgba(255, 0, 0, 0.15)", 1.0, None),
    ]

    REFERENCE_LINES = {
        "trend": [(0, "dot", "gray")],
        "velocity": [(0, "dot", "gray"), (2, "dash", "red"), (-2, "dash", "green")],
        "vix": [(2, "dash", "red"), (-1.5, "dash", "green")],
    }

    def extract_macro_slice(self, res: "EngineOutput") -> Optional[pd.DataFrame]:
        """Safely extract macro data aligned to plot dates."""
        if not hasattr(res, "macro_df") or res.macro_df is None:
            return None
        if res.normalized_plot_data.empty:
            return None

        plot_dates = res.normalized_plot_data.index
        macro_index = pd.to_datetime(res.macro_df.index)

        mask = (macro_index >= plot_dates.min()) & (macro_index <= plot_dates.max())
        slice_df = res.macro_df.loc[mask]

        return slice_df if not slice_df.empty else None

    def detect_regime(self, ratio: float) -> VolatilityRegime:
        for regime in self.REGIMES:
            if regime.contains(ratio):
                return regime
        return self.REGIMES[1]  # Fallback to TRANSITION

    def create_shading_shapes(self, macro_slice: pd.DataFrame) -> list[dict]:
        """Generate rectangle shapes for volatility regimes."""
        if "Macro_Vix_Ratio" not in macro_slice.columns:
            return []

        shapes = []
        ratio = macro_slice["Macro_Vix_Ratio"]

        for regime in self.REGIMES:
            mask = pd.Series([regime.contains(r) for r in ratio], index=ratio.index)
            if not mask.any():
                continue

            # Find contiguous regions
            diff = mask.astype(int).diff().fillna(0)
            starts = macro_slice.index[diff == 1]
            ends = macro_slice.index[diff == -1]

            if mask.iloc[0]:
                starts = starts.insert(0, macro_slice.index[0])
            if mask.iloc[-1]:
                ends = ends.append(pd.Index([macro_slice.index[-1]]))

            for s, e in zip(starts, ends):
                shapes.append(
                    dict(
                        type="rect",
                        x0=s,
                        x1=e,
                        y0=-3,
                        y1=5,
                        xref="x",
                        yref="y4",
                        fillcolor=regime.color,
                        line_width=0,
                        layer="below",
                    )
                )

        return shapes

    def update(
        self, fig: go.FigureWidget, res: "EngineOutput", inputs: "EngineInput"
    ) -> Optional[list[dict]]:
        macro_slice = self.extract_macro_slice(res)

        if macro_slice is None:
            for tid in [
                TraceId.TREND,
                TraceId.TREND_VELOCITY,
                TraceId.VIX_ZSCORE,
                TraceId.HY_SPREAD_Z,
                TraceId.YIELD_CURVE_Z,
            ]:
                fig.data[tid].visible = False
            return None

        # 1. Update Existing Traces
        fig.data[TraceId.TREND].update(
            x=macro_slice.index, y=macro_slice["Macro_Trend"], visible=True
        )
        fig.data[TraceId.TREND_VELOCITY].update(
            x=macro_slice.index, y=macro_slice["Macro_Trend_Vel_Z"], visible=True
        )
        fig.data[TraceId.VIX_ZSCORE].update(
            x=macro_slice.index, y=macro_slice["Macro_Vix_Z"], visible=True
        )
        fig.data[TraceId.HY_SPREAD_Z].update(
            x=macro_slice.index, y=macro_slice["High_Yield_Spread_Z"], visible=True
        )
        fig.data[TraceId.YIELD_CURVE_Z].update(
            x=macro_slice.index, y=macro_slice["Yield_Curve_10Y2Y_Z"], visible=True
        )

        # 2. Extract Ratios for the Title
        # Get ratio at decision date (using .asof for safety)
        decision_ratio = res.macro_df["Macro_Vix_Ratio"].asof(res.decision_date)
        # Get ratio at the end of the holding period (last point in slice)
        end_ratio = macro_slice["Macro_Vix_Ratio"].iloc[-1]

        # 3. Detect current regime based on end ratio
        regime = self.detect_regime(end_ratio)

        # 4. Update the Title with the new formatting
        self._update_volatility_title(fig, regime, decision_ratio, end_ratio)

        return self.create_shading_shapes(macro_slice)

    def _update_volatility_title(
        self,
        fig: go.FigureWidget,
        regime: VolatilityRegime,
        dec_ratio: float,
        end_ratio: float,
    ) -> None:
        """
        Updated title formatting to match standard Plotly subplot titles.
        Uses <sup> to mimic the styling of 'Market Momentum' and 'Market Regime' panes.
        """
        # Primary Title line (Matches font weight/size of other panes)
        title = (
            f"Volatility Regime: {regime.label} | "
            f"VIX Ratio: {dec_ratio:.2f} (Dec) ➔ {end_ratio:.2f} (End)"
        )

        # Subtitle line (Matches the styling of row 2 and 3 titles)
        subtitle = "Line: Intensity (Z-Score) | Background: Structure (Ratio < 1.0 = Healthy, > 1.0 = Crisis)"

        for ann in fig.layout.annotations:
            # Find the specific annotation for this subplot
            if "Volatility Regime" in ann.text or "VIX Ratio" in ann.text:
                # <sup> is the standard way Plotly handles the smaller subtitle text
                # removing <b> and custom <span> ensures it inherits global font settings
                ann.text = f"{title}<br><sup>{subtitle}</sup>"


# ============================================================================
# 4. CHART CONTROLLER (Owns Figure Lifecycle)
# ============================================================================


class ChartController:
    """Manages Plotly figure creation and updates. Knows about traces, but only via TraceId."""

    def __init__(self):
        self.fig = self._create_figure()
        self.macro_viz = MacroVisualizer()
        self._price_updater = PricePanelUpdater()

    def _create_figure(self) -> go.FigureWidget:
        fig = go.FigureWidget(
            make_subplots(
                rows=4,
                cols=1,
                row_heights=[0.6, 0.15, 0.12, 0.13],
                shared_xaxes=True,
                vertical_spacing=0.08,
                subplot_titles=(
                    "Event-Driven Walk-Forward Analysis",
                    "Market Regime (200d MA Deviation)<br><sup>Percentage deviation of benchmark price from its 200-day moving average</sup>",
                    "Market Momentum (21d Z-Score)<br><sup>Standardized 21-day change in Market Regime, using 63-day rolling volatility</sup>",
                    "Volatility Regime (VIX Z-Score)<br><sup>Standardized VIX index relative to its recent 63-day behavior</sup>",
                ),
                specs=[[{"secondary_y": False}]] * 4,
            )
        )

        fig = go.FigureWidget(
            make_subplots(
                rows=5,  # Updated
                cols=1,
                row_heights=[0.55, 0.12, 0.11, 0.11, 0.11],  # Rebalanced
                shared_xaxes=True,
                vertical_spacing=0.05,  # Tightened slightly
                subplot_titles=(
                    "Event-Driven Walk-Forward Analysis",
                    "Market Regime (200d MA Deviation)",
                    "Market Momentum (21d Z-Score)",
                    "Volatility Regime (VIX Z-Score)",
                    "Credit & Rates (Z-Score)",  # New Title
                ),
                specs=[[{"secondary_y": False}]] * 5,
            )
        )

        fig.update_layout(
            template="plotly_white",
            height=800,
            margin=dict(l=40, r=40, t=60, b=40),
            hovermode="x unified",
            showlegend=True,
        )

        self._init_traces(fig)
        self._init_reference_lines(fig)
        return fig

    def _init_traces(self, fig: go.FigureWidget) -> None:
        # Row 1: 50 ticker traces
        for _ in range(TraceId.TICKERS_END):
            fig.add_trace(
                go.Scatter(visible=False, line=dict(width=1.5), showlegend=True),
                row=1,
                col=1,
            )

        # Benchmark
        fig.add_trace(
            go.Scatter(
                name="Benchmark",
                line=dict(color="black", width=2.5, dash="dash"),
                visible=False,
            ),
            row=1,
            col=1,
        )

        # Portfolio
        fig.add_trace(
            go.Scatter(
                name="Group Portfolio",
                line=dict(color="green", width=3),
                visible=False,
            ),
            row=1,
            col=1,
        )

        # Macro traces (Row 2-4)
        fig.add_trace(
            go.Scatter(
                name="Trend",
                line=dict(color="#2E8B57", width=2),
                fillcolor="rgba(46, 139, 87, 0.15)",
                fill="tozeroy",
                visible=False,
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                name="Trend Vel (21d)",
                line=dict(color="#FF6B35", width=2),
                fillcolor="rgba(255, 107, 53, 0.15)",
                fill="tozeroy",
                visible=False,
            ),
            row=3,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                name="VIX-Z",
                line=dict(color="#800080", width=2),
                fillcolor="rgba(128, 0, 128, 0.15)",
                fill="tozeroy",
                visible=False,
            ),
            row=4,
            col=1,
        )

        # Row 5: Credit & Rates
        fig.add_trace(
            go.Scatter(
                name="HY Spread Z", line=dict(color="#FF4500", width=1.5), visible=False
            ),
            row=5,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                name="Yield Curve Z",
                line=dict(color="#1E90FF", width=1.5),
                visible=False,
            ),
            row=5,
            col=1,
        )

        # Axis labels
        fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
        fig.update_yaxes(title_text="Trend", tickformat=".0%", row=2, col=1)
        fig.update_yaxes(title_text="Trend Vel (Z)", tickformat=".1f", row=3, col=1)
        fig.update_yaxes(title_text="VIX (Z)", row=4, col=1)

        # Axis labels update
        fig.update_yaxes(title_text="Credit/Rates (Z)", row=5, col=1)

        # Hide x-axis for rows 1-4, show for row 5
        for r in [1, 2, 3, 4]:
            fig.update_xaxes(showticklabels=False, row=r, col=1)
        fig.update_xaxes(showticklabels=True, row=5, col=1)

    def _init_reference_lines(self, fig: go.FigureWidget) -> None:
        # Static horizontal lines defined declaratively
        lines = [
            (0, "dot", "gray", 2),  # y2
            (0, "dot", "gray", 3),  # y3
            (2, "dash", "red", 3),  # y3
            (-2, "dash", "green", 3),  # y3
            (2, "dash", "red", 4),  # y4
            (-1.5, "dash", "green", 4),  # y4
            (0, "dot", "gray", 5),  # Add zero-line to new row 5
        ]

        for y, dash, color, yaxis in lines:
            fig.add_hline(y=y, line_dash=dash, line_color=color, row=yaxis, col=1)

    def update(self, res: "EngineOutput", inputs: "EngineInput") -> None:
        with self.fig.batch_update():
            # Update price panel (delegated)
            self._price_updater.update(self.fig, res, inputs)

            # Update macro panel (delegated)
            regime_shapes = self.macro_viz.update(self.fig, res, inputs)

            # Event lines (decision, entry)
            event_shapes = self._create_event_shapes(res)

            # Combine all visual elements
            all_shapes = event_shapes
            if regime_shapes:
                all_shapes = all_shapes + regime_shapes
            self.fig.layout.shapes = all_shapes

    def _create_event_shapes(self, res: "EngineOutput") -> list[dict]:
        return [
            dict(
                type="line",
                x0=res.decision_date,
                x1=res.decision_date,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line=dict(color="red", width=2, dash="dash"),
            ),
            dict(
                type="line",
                x0=res.buy_date,
                x1=res.buy_date,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line=dict(color="blue", width=2, dash="dot"),
            ),
        ]


class PricePanelUpdater:
    """Handles the 50 tickers + benchmark + portfolio updates."""

    MAX_TICKERS = TraceId.TICKERS_END - TraceId.TICKERS_START

    def update(
        self, fig: go.FigureWidget, res: "EngineOutput", inputs: "EngineInput"
    ) -> None:
        cols = res.normalized_plot_data.columns.tolist()

        # Update visible tickers
        for i in range(self.MAX_TICKERS):
            trace_idx = TraceId.TICKERS_START + i
            if i < len(cols):
                fig.data[trace_idx].update(
                    x=res.normalized_plot_data.index,
                    y=res.normalized_plot_data[cols[i]],
                    name=cols[i],
                    visible=True,
                )
            else:
                fig.data[trace_idx].visible = False

        # Benchmark
        if not res.benchmark_series.empty:
            fig.data[TraceId.BENCHMARK].update(
                x=res.benchmark_series.index,
                y=res.benchmark_series.values,
                name=f"Benchmark ({inputs.benchmark_ticker})",
                visible=True,
            )
        else:
            fig.data[TraceId.BENCHMARK].visible = False

        # Portfolio
        if not res.portfolio_series.empty:
            fig.data[TraceId.PORTFOLIO].update(
                x=res.portfolio_series.index,
                y=res.portfolio_series.values,
                name="Group Portfolio",
                visible=True,
            )
        else:
            fig.data[TraceId.PORTFOLIO].visible = False


# ============================================================================
# 5. UI COMPONENT (Pure Widget Layout)
# ============================================================================


class WalkForwardUI:
    """Knows nothing about data. Just widgets and layout."""

    def __init__(self, initial_date: pd.Timestamp, settings: dict):
        self.settings = settings
        self._build_widgets(initial_date)
        self._wire_events()

    def _build_widgets(self, initial_date: pd.Timestamp) -> None:
        # Timeline
        self.w_lookback = widgets.IntText(
            value=189,
            description="Lookback (Days):",
            layout=widgets.Layout(width="200px"),
            style={"description_width": "initial"},
        )
        self.w_decision_date = widgets.DatePicker(
            value=initial_date,
            description="Decision Date:",
            layout=widgets.Layout(width="auto"),
            style={"description_width": "initial"},
        )
        self.w_holding = widgets.IntText(
            value=5,
            description="Holding (Days):",
            layout=widgets.Layout(width="200px"),
            style={"description_width": "initial"},
        )

        # Strategy
        self.w_mode = widgets.RadioButtons(
            options=["Ranking", "Manual List"],
            value="Ranking",
            description="Mode:",
            layout=widgets.Layout(width="max-content", margin="0px 20px 0px 0px"),
            style={"description_width": "initial"},
        )
        self.w_strategy = widgets.Dropdown(
            options=list(STRATEGY_REGISTRY.keys()),
            value="Sharpe (TRP)",
            description="Strategy:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="220px"),
        )
        self.w_benchmark = widgets.Text(
            value=self.settings["benchmark_ticker"],
            description="Benchmark:",
            placeholder="Enter Ticker",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="150px"),
        )

        # Ranking controls
        self.w_rank_start = widgets.IntText(
            value=1,
            description="Rank Start:",
            layout=widgets.Layout(width="150px"),
            style={"description_width": "initial"},
        )
        self.w_rank_end = widgets.IntText(
            value=100,
            description="Rank End:",
            layout=widgets.Layout(width="150px"),
            style={"description_width": "initial"},
        )
        self.w_rank_range = widgets.HBox([self.w_rank_start, self.w_rank_end])

        self.w_manual_list = widgets.Textarea(
            placeholder="AAPL, TSLA...",
            description="Manual Tickers:",
            layout=widgets.Layout(width="400px", height="80px"),
            style={"description_width": "initial"},
        )
        self.w_manual_list.layout.display = "none"

        # Actions
        self.w_run_btn = widgets.Button(
            description="Run Simulation", button_style="primary"
        )
        self.output_area = widgets.Output()

    def _wire_events(self) -> None:
        self.w_mode.observe(self._on_mode_change, names="value")

    def _on_mode_change(self, change) -> None:
        is_ranking = change["new"] == "Ranking"
        self.w_rank_range.layout.display = "flex" if is_ranking else "none"
        self.w_manual_list.layout.display = "none" if is_ranking else "flex"

    def get_input_values(self) -> dict:
        """Extract current widget values as plain data."""
        return {
            "lookback": self.w_lookback.value,
            "decision_date": pd.to_datetime(self.w_decision_date.value),
            "holding": self.w_holding.value,
            "mode": self.w_mode.value,
            "strategy": self.w_strategy.value,
            "benchmark": self.w_benchmark.value.strip().upper(),
            "rank_start": self.w_rank_start.value,
            "rank_end": self.w_rank_end.value,
            "manual_tickers": [
                t.strip().upper()
                for t in self.w_manual_list.value.split(",")
                if t.strip()
            ],
        }

    def layout(self, figure_widget: go.FigureWidget) -> widgets.VBox:
        """Assemble the final UI."""
        timeline_box = widgets.HBox(
            [self.w_lookback, self.w_decision_date, self.w_holding],
            layout=widgets.Layout(
                justify_content="space-between",
                border="1px solid #ddd",
                padding="10px",
                margin="5px 0px 15px 0px",
            ),
        )

        strategy_container = widgets.HBox(
            [self.w_strategy, self.w_benchmark],
            layout=widgets.Layout(margin="0px 0px 0px 10px"),
        )

        settings_row = widgets.HBox(
            [self.w_mode, strategy_container],
            layout=widgets.Layout(align_items="flex-start"),
        )

        return widgets.VBox(
            [
                widgets.HTML(
                    "<b>1. Timeline Configuration:</b> (Past <--- Decision ---> Future)"
                ),
                timeline_box,
                widgets.HTML("<b>2. Strategy Settings:</b>"),
                settings_row,
                self.w_rank_range,
                self.w_manual_list,
                widgets.HTML("<hr>"),
                self.w_run_btn,
                self.output_area,
                figure_widget,
            ]
        )

    def set_loading(self, loading: bool) -> None:
        self.w_run_btn.disabled = loading
        self.w_run_btn.description = "Calculating..." if loading else "Run Simulation"

    def show_error(self, msg: str) -> None:
        with self.output_area:
            print(f"⚠️ {msg}")

    def clear_output(self) -> None:
        self.output_area.clear_output(wait=True)


# ============================================================================
# 6. REPORT GENERATOR (Extracted Display Logic)
# ============================================================================


class ReportGenerator:
    """Generates the metrics table and audit logs - using proven old code."""

    def generate(
        self,
        res: "EngineOutput",
        inputs: "EngineInput",
        universe_subset: Optional[list],
    ) -> widgets.Widget:
        output = widgets.Output()
        output.layout = widgets.Layout(margin="10px 0px 20px 0px")

        with output:
            # Header (Success Message) - from old code
            mode_str = (
                f"CASCADE (Subset of {len(universe_subset)})"
                if universe_subset
                else "DISCOVERY (Full Market)"
            )
            display(
                widgets.HTML(
                    f"<div style='font-family:sans-serif; font-size:12px; margin-bottom:10px'><b style='color:green'>✅ Success</b> | Mode: {mode_str}</div>"
                )
            )

            # Audit Logic - from old code
            if (
                inputs.mode == "Ranking"
                and res.debug_data
                and "audit_liquidity" in res.debug_data
            ):
                audit = res.debug_data["audit_liquidity"]
                print("-" * 70)
                if audit.get("forced_list"):
                    print(f"🔍 STAGE 2 AUDIT: Cascade Mode Active")
                    print(
                        f"   Pool Size: {audit.get('tickers_passed')} survivors (Forced List)"
                    )
                else:
                    print(f"🔍 STAGE 1 AUDIT (Decision: {res.decision_date.date()})")
                    print(f"   Pool Size: {audit.get('tickers_passed')} survivors")
                print("-" * 70)

            # Timeline - from old code
            print(
                f"Timeline: [{res.start_date.date()}] -> Decision: {res.decision_date.date()} -> Entry: {res.buy_date.date()} -> End: {res.holding_end_date.date()}"
            )

            # Tickers - from old code
            print(f"Selected Tickers ({len(res.tickers)}):")
            if res.tickers:
                for i in range(0, len(res.tickers), 10):
                    print(", ".join(res.tickers[i : i + 10]) + ",")
            else:
                print("None")
            print("")

            # --- DATA PREP (Metrics Table) - from old code ---
            m = res.perf_metrics
            rows = []
            for label, key in [
                ("Gain", "gain"),
                ("Sharpe", "sharpe"),
                ("Sharpe (ATRP)", "sharpe_atrp"),
                ("Sharpe (TRP)", "sharpe_trp"),
            ]:
                p_row = {
                    "Metric": f"Group {label}",
                    "Full": m.get(f"full_p_{key}"),
                    "Lookback": m.get(f"lookback_p_{key}"),
                    "Holding": m.get(f"holding_p_{key}"),
                }
                b_row = {
                    "Metric": f"Benchmark {label}",
                    "Full": m.get(f"full_b_{key}"),
                    "Lookback": m.get(f"lookback_b_{key}"),
                    "Holding": m.get(f"holding_b_{key}"),
                }
                d_row = {"Metric": f"== {label} Delta"}
                for col in ["Full", "Lookback", "Holding"]:
                    d_row[col] = (p_row[col] or 0) - (b_row[col] or 0)
                rows.extend([p_row, b_row, d_row])

            df_report = pd.DataFrame(rows).set_index("Metric")

            # --- STYLE - from old code exactly ---
            styler = df_report.style.format("{:+.4f}", na_rep="N/A")

            def row_logic(row):
                if "Delta" in row.name:
                    return [
                        "background-color: #f9f9f9; font-weight: 600; border-top: 1px solid #ddd"
                    ] * len(row)
                if "Group" in row.name:
                    return ["color: #2c5e8f; background-color: #fcfdfe"] * len(row)
                return ["color: #555"] * len(row)

            styler.apply(row_logic, axis=1)
            styler.set_table_styles(
                [
                    {
                        "selector": "",
                        "props": [
                            ("font-family", "inherit"),
                            ("font-size", "12px"),
                            ("border-collapse", "collapse"),
                            ("width", "auto"),
                        ],
                    },
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "white"),
                            ("color", "#222"),
                            ("font-weight", "600"),
                            ("padding", "6px 12px"),
                            ("border-bottom", "2px solid #444"),
                            ("text-align", "center"),
                        ],
                    },
                    {
                        "selector": "th.row_heading",
                        "props": [
                            ("text-align", "left"),
                            ("padding-right", "30px"),
                            ("border-bottom", "1px solid #eee"),
                        ],
                    },
                    {
                        "selector": "td",
                        "props": [
                            ("padding", "4px 12px"),
                            ("border-bottom", "1px solid #eee"),
                        ],
                    },
                ]
            )
            styler.index.name = None
            display(styler)

        return output


# ============================================================================
# 7. MAIN ORCHESTRATOR (Slimmed Down)
# ============================================================================


class WalkForwardAnalyzer:
    """
    Thin orchestrator. No longer knows how to build charts, create widgets,
    or format tables. Just wires components together and handles the run loop.
    """

    def __init__(
        self, engine, universe_subset=None, filter_pack=None, default_settings=None
    ):
        self.engine = engine
        self.universe_subset = universe_subset
        self.filter_pack = filter_pack or FilterPack()
        self.settings = default_settings or GLOBAL_SETTINGS

        # Initialize components
        initial_date = self.filter_pack.decision_date or pd.to_datetime("2026-12-10")

        self.ui = WalkForwardUI(initial_date, self.settings)
        self.chart = ChartController()
        self.reporter = ReportGenerator()
        self.last_run: Optional["EngineOutput"] = None

        # Wire up the run button
        self.ui.w_run_btn.on_click(self._on_run)

    def _on_run(self, _):
        self.ui.set_loading(True)
        self.ui.clear_output()

        try:
            inputs = self._create_engine_input()
            result = self.engine.run(inputs)
            self.last_run = result

            if result.error_msg:
                self.ui.show_error(result.error_msg)
                return

            self._update_filter_pack(result)
            self.chart.update(result, inputs)

            # Render report to output area
            with self.ui.output_area:
                display(self.reporter.generate(result, inputs, self.universe_subset))

        except Exception as e:
            with self.ui.output_area:
                import traceback

                print(f"🚨 Error: {e}")
                traceback.print_exc()
        finally:
            self.ui.set_loading(False)

    def _create_engine_input(self) -> "EngineInput":
        vals = self.ui.get_input_values()
        return EngineInput(
            mode=vals["mode"],
            decision_date=vals["decision_date"],
            lookback_period=vals["lookback"],
            holding_period=vals["holding"],
            metric=vals["strategy"],
            benchmark_ticker=vals["benchmark"],
            rank_start=vals["rank_start"],
            rank_end=vals["rank_end"],
            manual_tickers=vals["manual_tickers"],
            universe_subset=self.universe_subset,
            debug=True,
        )

    def _update_filter_pack(self, res: "EngineOutput") -> None:
        """Still mutates filter_pack (legacy requirement), but isolated."""
        self.filter_pack.decision_date = res.decision_date
        self.filter_pack.selected_tickers = res.tickers

        if res.debug_data and "audit_liquidity" in res.debug_data:
            audit = res.debug_data["audit_liquidity"]
            if "universe_snapshot" in audit and isinstance(
                audit["universe_snapshot"], pd.DataFrame
            ):
                snap = audit["universe_snapshot"]
                self.filter_pack.eligible_pool = snap[
                    snap["Passed_Final"]
                ].index.tolist()

    def show(self):
        """Returns the composed UI."""
        container = self.ui.layout(self.chart.fig)
        self._on_run(None)  # Auto-run
        return container


# Factory function (kept for API compatibility)
def create_walk_forward_analyzer(engine, universe_subset=None, filter_pack=None):
    """Factory function to match the requested (analyzer, pack) return signature."""
    pack = filter_pack or FilterPack()
    analyzer = WalkForwardAnalyzer(
        engine, universe_subset=universe_subset, filter_pack=pack
    )
    return analyzer, pack


def run_headless_simulation(engine, inputs: EngineInput) -> pd.DataFrame:
    """
    Orchestrator: Prints the Audit Timeline and Returns the Metrics DataFrame.
    """
    result = engine.run(inputs)
    if result.error_msg:
        print(f"🚨 Engine Error: {result.error_msg}")
        return pd.DataFrame()

    # 1. Extract and Print Metadata (Matches the Screenshot Audit)
    meta = HeadlessReporter.get_metadata(result)

    print("-" * 70)
    print(
        f"Timeline: [{meta['start']}] -> Decision: {meta['decision']} -> "
        f"Entry: {meta['entry']} -> End: {meta['end']}"
    )

    # Chunk tickers into groups of 10
    tickers = meta["tickers"]
    rows = [", ".join(tickers[i : i + 10]) for i in range(0, len(tickers), 10)]
    print(f"Selected Tickers ({meta['ticker_count']}):\n" + "\n".join(rows))
    print("-" * 70)

    # 2. Return Table
    return HeadlessReporter.get_metrics_table(result)


#
