# src/plotting_utils.py

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import seaborn as sns

# =============================================================================
# === FORMATTERS
# =============================================================================

def millions_to_trillions_formatter(x, pos):
    """Formats a number from millions into trillions for a plot axis."""
    return f'{x / 1e6:.1f}T'

# =============================================================================
# === INDIVIDUAL SUBPLOT HELPERS
# =============================================================================

def _plot_rsi_sentiment(ax, df):
    ax.plot(df.index, df['RSI_mean'], label='Mean RSI', marker='o', markersize=3)
    ax.fill_between(df.index, df['RSI_25%'], df['RSI_75%'], color='blue', alpha=0.2, label='RSI 25%-75% Range')
    ax.axhline(50, color='grey', linestyle='--', linewidth=0.8, label='Neutral (50)')
    ax.axhline(70, color='red', linestyle=':', linewidth=0.8, label='Overbought (70)')
    ax.axhline(30, color='green', linestyle=':', linewidth=0.8, label='Oversold (30)')
    ax.set_title('Market Sentiment (RSI)')
    ax.set_ylabel('RSI Value'); ax.legend()

def _plot_short_term_perf(ax, df):
    ax.plot(df.index, df['Perf 3D %_mean'], label='Mean 3-Day Perf %', marker='o', markersize=3)
    ax.plot(df.index, df['Perf Week %_mean'], label='Mean Week Perf %', marker='s', markersize=3)
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    ax.set_title('Average Short-Term Performance')
    ax.set_ylabel('Performance (%)'); ax.legend()

def _plot_dist_from_sma(ax, df):
    ax.plot(df.index, df['SMA20 %_mean'], label='Mean % vs SMA20', marker='o', markersize=3)
    ax.plot(df.index, df['SMA50 %_mean'], label='Mean % vs SMA50', marker='s', markersize=3)
    ax.plot(df.index, df['SMA200 %_mean'], label='Mean % vs SMA200', marker='^', markersize=3)
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    ax.set_title('Average Distance from Moving Averages')
    ax.set_ylabel('Percent (%)'); ax.legend()

def _plot_dist_from_high_low(ax, df):
    ax.plot(df.index, df['50D High %_mean'], label='Mean % from 50D High', marker='o', markersize=3)
    ax.plot(df.index, df['50D Low %_mean'], label='Mean % above 50D Low', marker='s', markersize=3)
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    ax.set_title('Average Distance from 50-Day High/Low')
    ax.set_ylabel('Percent (%)'); ax.legend()

def _plot_volume_and_volatility(ax, df):
    color = 'tab:red'
    ax.set_ylabel('Relative Volume (Ratio)', color=color)
    ax.plot(df.index, df['Rel Volume_mean'], label='Mean Rel Volume', marker='o', markersize=3, color=color)
    ax.tick_params(axis='y', labelcolor=color)
    ax.axhline(1, color='grey', linestyle='--', linewidth=0.8, label='Avg Volume (1.0)')
    ax.legend(loc='upper left')
    ax2 = ax.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Weekly Volatility (%)', color=color)
    ax2.plot(df.index, df['Volatility W %_mean'], label='Mean Weekly Volatility %', marker='s', markersize=3, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')
    ax.set_title('Relative Volume & Volatility')

def _plot_risk_adjusted_returns(ax, df):
    ax.plot(df.index, df['Sharpe 3d_mean'], label='Mean Sharpe (3d)', marker='o', markersize=3, color='teal')
    ax.plot(df.index, df['Sortino 3d_mean'], label='Mean Sortino (3d)', marker='s', markersize=3, color='navy')
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    ax.set_title('Average Risk-Adjusted Returns (3-Day)')
    ax.set_ylabel('Ratio Value'); ax.legend()

# =============================================================================
# === HIGH-LEVEL PLOTTING FUNCTIONS (to be called from notebooks)
# =============================================================================

def plot_market_sentiment_dashboard(df_history: pd.DataFrame):
    """Generates and displays a 3x2 dashboard of market sentiment indicators."""
    if df_history is None or df_history.empty:
        print("Cannot generate dashboard: Input DataFrame is empty or None.")
        return
        
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 14, 'figure.titlesize': 18})

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20, 18), sharex=True)
    axes = axes.flatten()

    # Call each internal plotting helper
    _plot_rsi_sentiment(axes[0], df_history)
    _plot_short_term_perf(axes[1], df_history)
    _plot_dist_from_sma(axes[2], df_history)
    _plot_dist_from_high_low(axes[3], df_history)
    _plot_volume_and_volatility(axes[4], df_history)
    _plot_risk_adjusted_returns(axes[5], df_history)

    # Apply final formatting to all subplots
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    fig.autofmt_xdate(rotation=30, ha='right')
    start_date = df_history.index.min().strftime('%Y-%m-%d')
    end_date = df_history.index.max().strftime('%Y-%m-%d')
    fig.suptitle(f'Market Internals Analysis ({start_date} to {end_date})', y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def plot_single_timeseries(
    series: pd.Series, 
    title: str, 
    ylabel: str, 
    y_formatter: callable = None
    ):
    """
    Creates a standardized, presentation-quality plot for a single time series.
    
    Args:
        series (pd.Series): The time series data to plot (must have a DatetimeIndex).
        title (str): The main title for the plot.
        ylabel (str): The label for the y-axis.
        y_formatter (callable, optional): A matplotlib formatter function for the y-axis.
    """
    if series is None or series.empty:
        print("Cannot generate plot: Input Series is empty or None.")
        return

    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 7))
    ax = sns.lineplot(data=series)
    
    if y_formatter:
        ax.yaxis.set_major_formatter(FuncFormatter(y_formatter))
        
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()

import plotly.express as px
import pandas as pd
import math

def plot_rank_with_criteria(df_rank_history, ticker_list, title_suffix="", filter_criteria=None,
                            width=1100, height=700):
    """
    Plots rank history with period markers, interactive buttons, and size controls.
    - Adds vertical lines and shaded regions for lookback and recent periods.
    - 'Clear All' sets traces to 'legendonly', hiding them and graying out the legend item.
    - 'Reset View' makes all traces fully visible.
    
    Args:
        df_rank_history (pd.DataFrame): The full rank history DataFrame.
        ticker_list (list): A list of ticker symbols to plot.
        title_suffix (str, optional): Text to append to the main plot title.
        filter_criteria (dict, optional): A dictionary of filter parameters to display.
        width (int, optional): The width of the figure in pixels.
        height (int, optional): The height of the figure in pixels.
    """
    
    if not ticker_list:
        print("Ticker list is empty. Nothing to plot.")
        return

    # Prepare data for plotting
    plot_df = df_rank_history.loc[ticker_list].T
    plot_df.index = pd.to_datetime(plot_df.index)

    custom = px.colors.qualitative.Plotly.copy()
    # Replace the 7th color '#B6E880' with '#1F77B4' with a darker blue
    custom[7] = '#1F77B4'        

    fig = px.line(
        plot_df, 
        x=plot_df.index, 
        y=plot_df.columns,
        # Line color sequence 
        color_discrete_sequence=custom,
        title=f"Rank History: {title_suffix}",
        labels={'value': 'Rank', 'x': 'Date', 'variable': 'Ticker'}
    )

    # Force x-axis title as "Date"
    fig.update_xaxes(title_text="Date", title_standoff=25)

    # Y-Axis configuration (unchanged and correct)
    fig.update_yaxes(
        autorange="reversed", 
        dtick=100,
        showgrid=True,
        gridcolor='LightGrey'
    )
    
    # --- CORRECTED X-Axis configuration ---
    fig.update_xaxes(
        type='date',  # Ensures the axis is treated as a continuous date axis
        
        # --- Major Ticks and Grid (for labels) ---
        showgrid=True,
        gridcolor='LightGrey',
        dtick="D7",  #<-- MAJOR CHANGE: Place labels and major grid lines only every 7 days
        tickformat="%b %d",  # Format as "May 04". Now readable with weekly spacing.

        # --- Minor Ticks and Grid (for dense grid lines WITHOUT labels) ---
        minor=dict(
            showgrid=True, 
            gridcolor="rgba(235, 235, 235, 0.5)", # A fainter color for minor lines
            dtick="D1" #<-- Place an unlabeled minor grid line every 1 day
        )
    )

    # --- Vertical lines and shaded regions section (unchanged) ---
    if filter_criteria and 'recent_days' in filter_criteria and 'lookback_days' in filter_criteria:
        recent_days = filter_criteria.get('recent_days', 0)
        lookback_days = filter_criteria.get('lookback_days', 0)
        all_dates = pd.to_datetime(df_rank_history.columns)
        if len(all_dates) >= (lookback_days + recent_days):
            last_date = all_dates[-1]
            recent_period_start_date = all_dates[-recent_days]
            lookback_period_end_date = all_dates[-(recent_days + 1)]
            lookback_period_start_date = all_dates[-(recent_days + lookback_days)]
            fig.add_vrect(x0=recent_period_start_date, x1=last_date, fillcolor="LightSkyBlue", opacity=0.2, layer="below", line_width=0, annotation_text="Recent", annotation_position="top left")
            fig.add_vrect(x0=lookback_period_start_date, x1=lookback_period_end_date, fillcolor="LightGreen", opacity=0.2, layer="below", line_width=0, annotation_text="Lookback", annotation_position="top left")
            fig.add_vline(x=recent_period_start_date, line_width=2, line_dash="dash", line_color="grey")

    # --- Interactive buttons section (unchanged) ---
    num_traces = len(fig.data)
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons", direction="left", x=0.01, xanchor="left",
                y=1.1, yanchor="top", showactive=False,
                buttons=list([
                    dict(label="Reset View", method="restyle", args=[{"visible": [True] * num_traces}]),
                    dict(label="Clear All", method="restyle", args=[{"visible": ['legendonly'] * num_traces}]),
                ]),
            )
        ]
    )
    
    # --- Filter criteria annotation section (unchanged) ---
    criteria_text = ""
    num_rows = 0
    if filter_criteria:
        active_criteria = {k: v for k, v in filter_criteria.items() if v is not None}
        col_width = 38
        criteria_lines = []
        items = list(active_criteria.items())
        for i in range(0, len(items), 3):
            chunk = items[i:i+3]
            line_parts = [f"{f'  • {k}: {v}':<{col_width}}" for k, v in chunk]
            criteria_lines.append("".join(line_parts))
        num_rows = len(criteria_lines)
        criteria_text = "<b>Filter Criteria:</b><br>" + "<br>".join(criteria_lines)

    if criteria_text:
        FONT_SIZE = 14                       # whatever you want
        LINE_HEIGHT = FONT_SIZE * 1.4        # ~40 % leading is comfortable

        criteria_text = "<b>Filter Criteria:</b><br>" + "<br>".join(criteria_lines)

        fig.add_annotation(
            showarrow=False,
            text=criteria_text,
            xref="paper", yref="paper",
            x=0, y=-0.25,            
            xanchor="left", yanchor="top",
            align="left",
            font=dict(family="Courier New, monospace", size=FONT_SIZE)
        )


    # compute y-range manually
    y_vals = plot_df.values.ravel()          # all y values
    y_range = (y_vals.min(), y_vals.max())
    
    # ------------------------------------------------------------------
    # horizontal red boundary lines – use the data range we just computed
    # ------------------------------------------------------------------
    y_min, y_max = y_range       # comes from the manual computation above

    # xref and yref = "paper", uses plot area as a reference. 
    # x0, y0 = 0, 0 is lower left corner
    # x1, y1 = 1, 1 is upper right corner
    fig.add_shape(type="line",
                  xref="paper", yref="paper",
                  x0=0, x1=1,
                  y0=1, y1=1,  
                  line=dict(color="red", width=2))  # top of plot area

    fig.add_shape(type="line",
                  xref="paper", yref="paper",
                  x0=0, x1=1,
                  y0=0, y1=0,  
                  line=dict(color="blue", width=2))  # bottom of plot area

    # # optional thin line just below the filter-text annotation
    # fig.add_shape(type="line",
    #               xref="paper", yref="paper",
    #               x0=0, x1=1,
    #               y0=-0.05, y1=-0.05,                 
    #               line=dict(color="green", width=2))  # line 5% below the plot area


    # --- Layout update section (unchanged) ---
    bottom_margin = 140 + num_rows * LINE_HEIGHT
    fig.update_layout(width=width, height=height, margin=dict(b=bottom_margin))

    # Set the x-axis range to a suitable range
    fig.update_xaxes(range=[plot_df.index.min(), plot_df.index.max()])

    fig.show()
    return fig



