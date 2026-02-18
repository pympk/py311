import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display
from typing import Optional, List

# Internal Imports
from core.settings import GLOBAL_SETTINGS
from core.contracts import EngineInput, EngineOutput, FilterPack
from strategy.registry import METRIC_REGISTRY


class WalkForwardAnalyzer:
    def __init__(
        self, engine, universe_subset=None, filter_pack=None, default_settings=None
    ):
        self.engine = engine
        self.universe_subset = universe_subset
        self.filter_pack = filter_pack or FilterPack()
        self.settings = default_settings or GLOBAL_SETTINGS
        self.last_run: Optional[EngineOutput] = None

        # Sync Date with FilterPack if we are in Stage 2
        self.initial_date = self.filter_pack.decision_date or pd.to_datetime(
            "2025-12-10"
        )

        self._init_widgets()
        self._init_figure()
        self.output_area = widgets.Output()

    def _init_widgets(self):
        # --- 1. Timeline Inputs ---
        self.w_lookback = widgets.IntText(
            value=10,
            description="Lookback (Days):",
            layout=widgets.Layout(width="200px"),
            style={"description_width": "initial"},
        )
        self.w_decision_date = widgets.DatePicker(
            value=self.initial_date,
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

        # --- 2. Strategy & Benchmark ---
        self.w_mode = widgets.RadioButtons(
            options=["Ranking", "Manual List"],
            value="Ranking",
            description="Mode:",
            layout=widgets.Layout(width="max-content", margin="0px 20px 0px 0px"),
            style={"description_width": "initial"},
        )

        common_style = {"description_width": "initial"}

        self.w_strategy = widgets.Dropdown(
            options=list(METRIC_REGISTRY.keys()),
            value="Sharpe (ATRP)",
            description="Strategy:",
            style=common_style,
            layout=widgets.Layout(width="220px"),
        )

        self.w_benchmark = widgets.Text(
            value=self.settings["benchmark_ticker"],
            description="Benchmark:",
            placeholder="Enter Ticker",
            style=common_style,
            layout=widgets.Layout(width="150px"),
        )

        # --- 3. Ranking Controls ---
        self.w_rank_start = widgets.IntText(
            value=1,
            description="Rank Start:",
            layout=widgets.Layout(width="150px"),
            style={"description_width": "initial"},
        )
        self.w_rank_end = widgets.IntText(
            value=10,
            description="Rank End:",
            layout=widgets.Layout(width="150px"),
            style={"description_width": "initial"},
        )
        # Grouping them here for logic, but we'll group them visually in show()
        self.w_rank_range = widgets.HBox([self.w_rank_start, self.w_rank_end])

        self.w_manual_list = widgets.Textarea(
            placeholder="AAPL, TSLA...",
            description="Manual Tickers:",
            layout=widgets.Layout(width="400px", height="80px"),
            style={"description_width": "initial"},
        )
        self.w_manual_list.layout.display = "none"

        # --- 4. Run Button ---
        self.w_run_btn = widgets.Button(
            description="Run Simulation",
            button_style="primary",
        )

        # Observers
        self.w_mode.observe(self._on_mode_change, names="value")
        self.w_run_btn.on_click(self._on_run_clicked)

    def _init_figure(self):
        """Initialize 3-panel figure using original layout parameters"""
        self.fig = go.FigureWidget(
            make_subplots(
                rows=4,
                cols=1,
                row_heights=[0.6, 0.15, 0.12, 0.13],
                shared_xaxes=True,
                vertical_spacing=0.08,
                subplot_titles=(
                    # Row 1
                    "Event-Driven Walk-Forward Analysis",
                    # Row 2: CORRECTED
                    "Market Regime (200d MA Deviation)<br><sup>Percentage deviation of benchmark price from its 200-day moving average</sup>",
                    # Row 3: CORRECTED — Technical accuracy + clarity
                    "Market Momentum (21d Z-Score)<br><sup>Standardized 21-day change in Market Regime, using 63-day rolling volatility</sup>",
                    # Row 4: CORRECTED — Grammar
                    "Volatility Regime (VIX Z-Score)<br><sup>Standardized VIX index relative to its recent 63-day behavior</sup>",
                ),
                specs=[
                    [{"secondary_y": False}],
                    [{"secondary_y": False}],
                    [{"secondary_y": False}],
                    [{"secondary_y": False}],
                ],
            )
        )

        # EXACT old layout configuration
        self.fig.update_layout(
            template="plotly_white",
            height=800,  # Slightly increased for 3 panels but maintains proportions
            margin=dict(l=40, r=40, t=60, b=40),  # EXACTLY like old code
            hovermode="x unified",
            showlegend=True,
            # NO explicit legend positioning - uses Plotly defaults like old code
            # This places legend immediately adjacent to plot area on the right
        )

        # --- Row 1: Price Action (Traces 0-51) ---
        # 1. Create 50 empty traces for Tickers (Legend enabled by default like old code)
        for _ in range(50):
            self.fig.add_trace(
                go.Scatter(
                    visible=False,
                    line=dict(width=1.5),
                    showlegend=True,  # Explicitly True (old code default behavior)
                ),
                row=1,
                col=1,
            )

        # 2. Trace 50: Benchmark (Black Dash)
        self.fig.add_trace(
            go.Scatter(
                name="Benchmark",
                line=dict(color="black", width=2.5, dash="dash"),
                visible=False,
            ),
            row=1,
            col=1,
        )

        # 3. Trace 51: Group Portfolio (Green)
        self.fig.add_trace(
            go.Scatter(
                name="Group Portfolio", line=dict(color="green", width=3), visible=False
            ),
            row=1,
            col=1,
        )

        # --- Row 2: Macro Trend [52] ---
        self.fig.add_trace(
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

        # Zero line for Trend (static)
        self.fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

        # --- Row 3: Trend Velocity [53] ---  <-- ADD THIS BLOCK
        self.fig.add_trace(
            go.Scatter(
                name="Trend Vel (21d)",
                line=dict(color="#FF6B35", width=2),
                fillcolor="rgba(255, 107, 53, 0.15)",  # Orange tint
                fill="tozeroy",  # Fill to zero
                visible=False,
            ),
            row=3,  # New row 3
            col=1,
        )
        # Zero line for velocity (static, row 3)
        self.fig.add_hline(y=0, line_dash="dot", line_color="gray", row=3, col=1)
        self.fig.add_hline(
            y=2,
            line_dash="dash",
            line_color="red",
            annotation_text="Accel",
            row=3,
            col=1,
        )  # Fear threshold
        self.fig.add_hline(
            y=-2,
            line_dash="dash",
            line_color="green",
            annotation_text="Decel",
            row=3,
            col=1,
        )  # Capitulation threshold

        # --- Row 4: VIX Z-Score [54] ---
        self.fig.add_trace(
            go.Scatter(
                name="VIX-Z",
                line=dict(color="#800080", width=2),
                fillcolor="rgba(128, 0, 128, 0.15)",  # Purple tint
                fill="tozeroy",  # Fill to zero
                visible=False,
            ),
            row=4,  # <-- ADD THIS
            col=1,  # <-- ADD THIS
        )

        # Reference lines for VIX
        self.fig.add_hline(
            y=2,
            line_dash="dash",
            line_color="red",
            row=4,
            col=1,
            annotation_text="Fear",
        )
        self.fig.add_hline(
            y=-1.5,
            line_dash="dash",
            line_color="green",
            row=4,
            col=1,
            annotation_text="Calm",
        )

        # Axis labels
        self.fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
        self.fig.update_yaxes(title_text="Trend", tickformat=".0%", row=2, col=1)
        self.fig.update_yaxes(
            title_text="Trend Vel (Z)", tickformat=".1f", row=3, col=1
        )
        self.fig.update_yaxes(title_text="VIX (Z)", row=4, col=1)

        # Hide x-axis labels for top rows
        self.fig.update_xaxes(showticklabels=False, row=1, col=1)
        self.fig.update_xaxes(showticklabels=False, row=2, col=1)
        self.fig.update_xaxes(
            showticklabels=False, row=3, col=1
        )  # CHANGED (was row 3, still hide for middle)

    def _on_mode_change(self, change):
        is_ranking = change["new"] == "Ranking"
        self.w_rank_range.layout.display = "flex" if is_ranking else "none"
        self.w_manual_list.layout.display = "none" if is_ranking else "flex"

    def _on_run_clicked(self, b):
        # 1. UI Feedback
        self.w_run_btn.disabled = True
        self.w_run_btn.description = "Calculating..."

        # 2. CLEAR THE OUTPUT (Use wait=True to prevent flickering/doubling)
        self.output_area.clear_output(wait=True)

        with self.output_area:
            try:
                # 3. Capture Inputs
                cur_decision_date = pd.to_datetime(self.w_decision_date.value)
                manual_list = [
                    t.strip().upper()
                    for t in self.w_manual_list.value.split(",")
                    if t.strip()
                ]

                inputs = EngineInput(
                    mode=self.w_mode.value,
                    start_date=cur_decision_date,
                    lookback_period=self.w_lookback.value,
                    holding_period=self.w_holding.value,
                    metric=self.w_strategy.value,
                    benchmark_ticker=self.w_benchmark.value.strip().upper(),
                    rank_start=self.w_rank_start.value,
                    rank_end=self.w_rank_end.value,
                    manual_tickers=manual_list,
                    universe_subset=self.universe_subset,
                    debug=True,
                )

                # 4. Engine Run
                res = self.engine.run(inputs)
                self.last_run = res

                if res.error_msg:
                    print(f"⚠️ {res.error_msg}")
                    return

                # 3. Update FilterPack (The "Save" Step)
                self.filter_pack.decision_date = res.decision_date
                self.filter_pack.selected_tickers = res.tickers

                # Extract eligible pool (Survivors) from audit data if available
                if res.debug_data and "audit_liquidity" in res.debug_data:
                    audit = res.debug_data["audit_liquidity"]
                    if "universe_snapshot" in audit and isinstance(
                        audit["universe_snapshot"], pd.DataFrame
                    ):
                        snap = audit["universe_snapshot"]
                        self.filter_pack.eligible_pool = snap[
                            snap["Passed_Final"]
                        ].index.tolist()

                # 4. Render Visuals
                self._update_plots(res, inputs)
                self._display_audit_and_metrics(res, inputs)

            except Exception as e:
                import traceback

                print(f"🚨 Error: {str(e)}")
                traceback.print_exc()
            finally:
                self.w_run_btn.disabled = False
                self.w_run_btn.description = "Run Simulation"

    def _update_plots(self, res: EngineOutput, inputs: EngineInput):
        with self.fig.batch_update():
            # --- A. Row 1: Ticker & Portfolio Traces ---
            cols = res.normalized_plot_data.columns.tolist()
            for i in range(50):
                if i < len(cols):
                    self.fig.data[i].update(
                        x=res.normalized_plot_data.index,
                        y=res.normalized_plot_data[cols[i]],
                        name=cols[i],
                        visible=True,
                    )
                else:
                    self.fig.data[i].visible = False

            # Benchmark (Trace 50)
            if not res.benchmark_series.empty:
                self.fig.data[50].update(
                    x=res.benchmark_series.index,
                    y=res.benchmark_series.values,
                    name=f"Benchmark ({inputs.benchmark_ticker})",
                    visible=True,
                )
            else:
                self.fig.data[50].visible = False

            # Portfolio (Trace 51)
            if not res.portfolio_series.empty:
                self.fig.data[51].update(
                    x=res.portfolio_series.index,
                    y=res.portfolio_series.values,
                    name="Group Portfolio",
                    visible=True,
                )
            else:
                self.fig.data[51].visible = False

            # --- B. Rows 2 & 3: Macro Data ---
            # Check if macro_df exists and slice to current window
            if (
                hasattr(res, "macro_df")
                and res.macro_df is not None
                and not res.normalized_plot_data.empty
            ):
                plot_dates = res.normalized_plot_data.index
                # Ensure macro_df has DatetimeIndex
                if not isinstance(res.macro_df.index, pd.DatetimeIndex):
                    macro_index = pd.to_datetime(res.macro_df.index)
                else:
                    macro_index = res.macro_df.index

                # Slice macro data to match plot window for performance and visual alignment
                mask = (macro_index >= plot_dates.min()) & (
                    macro_index <= plot_dates.max()
                )
                macro_slice = res.macro_df.loc[mask]

                if not macro_slice.empty:
                    # Update Trend (Trace 52)
                    self.fig.data[52].update(
                        x=macro_slice.index, y=macro_slice["Macro_Trend"], visible=True
                    )

                    # Update Trend Velocity (Trace 53) --- ADD THIS BLOCK
                    if "Macro_Trend_Vel_Z" in macro_slice.columns:
                        self.fig.data[53].update(
                            x=macro_slice.index,
                            y=macro_slice[
                                "Macro_Trend_Vel_Z"
                            ],  # Changed from Macro_Trend_Vel
                            visible=True,
                        )

                    # Update VIX-Z (Trace 54)
                    if "Macro_Vix_Z" in macro_slice.columns:
                        self.fig.data[54].update(
                            x=macro_slice.index,
                            y=macro_slice["Macro_Vix_Z"],
                            visible=True,
                        )

            else:
                # Hide macro traces if no data
                self.fig.data[52].visible = False
                self.fig.data[53].visible = False
                self.fig.data[54].visible = False  # <-- ADD THIS LINE

            # --- C. Vertical Event Lines (Span all rows) ---
            # Using yref="paper" spans 0-1 across all subplots
            shapes = [
                dict(
                    type="line",
                    x0=res.decision_date,
                    y0=0,
                    x1=res.decision_date,
                    y1=1,
                    xref="x",
                    yref="paper",  # Critical: spans all subplots
                    line=dict(color="red", width=2, dash="dash"),
                    name="Decision Date",
                ),
                dict(
                    type="line",
                    x0=res.buy_date,
                    y0=0,
                    x1=res.buy_date,
                    y1=1,
                    xref="x",
                    yref="paper",
                    line=dict(color="blue", width=2, dash="dot"),
                    name="Execution Date",
                ),
            ]

            # Vertical event lines spanning all rows (yref='paper')
            event_shapes = [
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

            # Static horizontal reference lines (manualdicts to avoid add_hline bugs)
            # yref mapping: row2='y2', row3='y3', row4='y4'
            static_shapes = [
                # Row 2: Trend zero line
                dict(
                    type="line",
                    y0=0,
                    y1=0,
                    x0=0,
                    x1=1,
                    xref="paper",
                    yref="y2",
                    line=dict(color="gray", width=1, dash="dot"),
                ),
                # Row 3: Velocity Accel (+2)
                dict(
                    type="line",
                    y0=2,
                    y1=2,
                    x0=0,
                    x1=1,
                    xref="paper",
                    yref="y3",
                    line=dict(color="red", width=1, dash="dash"),
                ),
                # Row 3: Velocity Decel (-2)
                dict(
                    type="line",
                    y0=-2,
                    y1=-2,
                    x0=0,
                    x1=1,
                    xref="paper",
                    yref="y3",
                    line=dict(color="green", width=1, dash="dash"),
                ),
                # Row 3: Velocity Zero
                dict(
                    type="line",
                    y0=0,
                    y1=0,
                    x0=0,
                    x1=1,
                    xref="paper",
                    yref="y3",
                    line=dict(color="gray", width=1, dash="dot"),
                ),
                # Row 4: VIX Fear (+2)
                dict(
                    type="line",
                    y0=2,
                    y1=2,
                    x0=0,
                    x1=1,
                    xref="paper",
                    yref="y4",
                    line=dict(color="red", width=1, dash="dash"),
                ),
                # Row 4: VIX Calm (-1.5)
                dict(
                    type="line",
                    y0=-1.5,
                    y1=-1.5,
                    x0=0,
                    x1=1,
                    xref="paper",
                    yref="y4",
                    line=dict(color="green", width=1, dash="dash"),
                ),
            ]

            self.fig.layout.shapes = event_shapes + static_shapes

    def _display_audit_and_metrics(self, res: EngineOutput, inputs: EngineInput):
        self.output_area.layout = widgets.Layout(margin="10px 0px 20px 0px")

        # Header (Success Message)
        mode_str = (
            f"CASCADE (Subset of {len(self.universe_subset)})"
            if self.universe_subset
            else "DISCOVERY (Full Market)"
        )
        display(
            widgets.HTML(
                f"<div style='font-family:sans-serif; font-size:12px; margin-bottom:10px'><b style='color:green'>✅ Success</b> | Mode: {mode_str}</div>"
            )
        )

        # Audit Logic
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

        # Timeline
        print(
            f"Timeline: [{res.start_date.date()}] -> Decision: {res.decision_date.date()} -> Entry: {res.buy_date.date()} -> End: {res.holding_end_date.date()}"
        )

        # --- FIX: WRAP TICKERS TO 10 PER LINE ---
        print(f"Selected Tickers ({len(res.tickers)}):")
        if res.tickers:
            for i in range(0, len(res.tickers), 10):
                print(", ".join(res.tickers[i : i + 10]) + ",")
        else:
            print("None")
        print("")
        # ----------------------------------------

        # --- DATA PREP (Metrics Table) ---
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

        # --- STYLE ---
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

    def show(self):
        # 1. Timeline Box (Bordered)
        timeline_box = widgets.HBox(
            [self.w_lookback, self.w_decision_date, self.w_holding],
            layout=widgets.Layout(
                justify_content="space-between",
                border="1px solid #ddd",
                padding="10px",
                margin="5px 0px 15px 0px",
            ),
        )

        # 2. Strategy & Benchmark container
        strategy_container = widgets.HBox(
            [self.w_strategy, self.w_benchmark],
            layout=widgets.Layout(margin="0px 0px 0px 10px"),
        )

        # 3. Settings Row
        settings_row = widgets.HBox(
            [self.w_mode, strategy_container],
            layout=widgets.Layout(align_items="flex-start"),
        )

        # 4. Construct UI
        ui = widgets.VBox(
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
                self.fig,  # The FigureWidget with subplots
            ]
        )

        # display(ui)
        # Auto-run on display
        self._on_run_clicked(None)
        return ui  # <--- Changed from display(ui)


def create_walk_forward_analyzer(engine, universe_subset=None, filter_pack=None):
    """Factory function to match the requested (analyzer, pack) return signature."""
    pack = filter_pack or FilterPack()
    analyzer = WalkForwardAnalyzer(
        engine, universe_subset=universe_subset, filter_pack=pack
    )
    return analyzer, pack
