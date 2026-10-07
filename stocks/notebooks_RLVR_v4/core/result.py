import pandas as pd

from typing import Dict, Any
from dataclasses import dataclass

from core.contracts import EngineOutput


@dataclass(frozen=True)
class TaskResult:
    ok: bool
    msg: str = ""
    val: Any = None


class HeadlessReporter:
    """
    Expert Logic: Extracts metrics and timeline metadata from EngineOutput.
    Maintains transparency and trust without needing Plotly.
    """

    @staticmethod
    def get_metadata(res: EngineOutput) -> Dict[str, Any]:
        """Extracts the timeline and ticker list as structured data."""
        return {
            "start": res.start_date.date(),
            "decision": res.decision_date.date(),
            "entry": res.buy_date.date(),
            "end": res.holding_end_date.date(),
            "tickers": res.tickers,
            "ticker_count": len(res.tickers),
        }

    @staticmethod
    def get_metrics_table(res: EngineOutput) -> pd.DataFrame:
        """The existing table logic - keeps the math verified."""
        m = res.perf_metrics
        rows = []
        metric_types = [
            ("Gain", "gain"),
            ("Sharpe", "sharpe"),
            ("Sharpe (ATRP)", "sharpe_atrp"),
            ("Sharpe (TRP)", "sharpe_trp"),
        ]

        for label, key in metric_types:
            # Pylance infers these as dict[str, Any] because they are mixed at birth
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

            # Explicitly type d_row so Pylance knows it's not just for strings
            d_row: dict[str, Any] = {"Metric": f"== {label} Delta"}

            for col in ["Full", "Lookback", "Holding"]:
                # Ensure we are working with floats to avoid NoneType math errors
                p_val = float(p_row.get(col) or 0.0)
                b_val = float(b_row.get(col) or 0.0)
                d_row[col] = p_val - b_val

            rows.extend([p_row, b_row, d_row])

        return pd.DataFrame(rows).set_index("Metric")


#
