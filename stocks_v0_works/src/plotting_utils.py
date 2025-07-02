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