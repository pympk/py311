import pandas as pd

from collections import Counter
from core.quant import QuantUtils
from typing import Union, Tuple, List


def _prepare_initial_weights(tickers: List[str]) -> pd.Series:
    """Helper to convert ticker list to weight map."""
    ticker_counts = Counter(tickers)
    total = len(tickers)
    return pd.Series({t: c / total for t, c in ticker_counts.items()})


def calculate_buy_and_hold_performance(
    df_close_wide: pd.DataFrame,
    df_atrp_wide: pd.DataFrame,
    df_trp_wide: pd.DataFrame,
    tickers: List[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
):
    if not tickers:
        return pd.Series(), pd.Series(), pd.Series()

    # Internal helper call
    initial_weights = _prepare_initial_weights(tickers)

    # SLICE Logic (Belongs here, not in quant.py)
    ticker_list = initial_weights.index.tolist()
    p_slice = df_close_wide.reindex(columns=ticker_list).loc[start_date:end_date]
    a_slice = df_atrp_wide.reindex(columns=ticker_list).loc[start_date:end_date]
    t_slice = df_trp_wide.reindex(columns=ticker_list).loc[start_date:end_date]

    # KERNEL Call - Dispatching to pure math
    return QuantUtils.compute_portfolio_stats(
        p_slice, a_slice, t_slice, initial_weights
    )


class PerformanceCalculator:

    @staticmethod
    def calculate_period_metrics(
        full_data: tuple, hold_data: tuple, decision_date: pd.Timestamp, prefix: str
    ):
        f_val, f_ret, f_atrp, f_trp = full_data
        h_val, h_ret, h_atrp, h_trp = hold_data

        lb_val = f_val.loc[:decision_date]
        lb_ret = f_ret.loc[:decision_date]
        lb_atrp = f_atrp.loc[:decision_date]
        lb_trp = f_trp.loc[:decision_date]

        m = {
            f"full_{prefix}_gain": QuantUtils.calculate_gain(f_val),
            f"full_{prefix}_sharpe": QuantUtils.calculate_sharpe(f_ret),
            f"full_{prefix}_sharpe_atrp": QuantUtils.calculate_sharpe_vol(
                f_ret, f_atrp
            ),
            f"full_{prefix}_sharpe_trp": QuantUtils.calculate_sharpe_vol(f_ret, f_trp),
            f"lookback_{prefix}_gain": QuantUtils.calculate_gain(lb_val),
            f"lookback_{prefix}_sharpe": QuantUtils.calculate_sharpe(lb_ret),
            f"lookback_{prefix}_sharpe_atrp": QuantUtils.calculate_sharpe_vol(
                lb_ret, lb_atrp
            ),
            f"lookback_{prefix}_sharpe_trp": QuantUtils.calculate_sharpe_vol(
                lb_ret, lb_trp
            ),
            f"holding_{prefix}_gain": QuantUtils.calculate_gain(h_val),
            f"holding_{prefix}_sharpe": QuantUtils.calculate_sharpe(h_ret),
            f"holding_{prefix}_sharpe_atrp": QuantUtils.calculate_sharpe_vol(
                h_ret, h_atrp
            ),
            f"holding_{prefix}_sharpe_trp": QuantUtils.calculate_sharpe_vol(
                h_ret, h_trp
            ),
        }

        return m, {"val": f_val, "ret": f_ret, "atrp": f_atrp, "trp": f_trp}


#
