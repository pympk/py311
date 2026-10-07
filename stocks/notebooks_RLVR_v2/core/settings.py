import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from core.paths import OUTPUT_DIR


@dataclass
class CacheConfig:
    """
    [SHARED CONSTANTS] A single source of truth for the dataset slices
    and feature lookback window.
    """

    LOOKBACK: int = int(os.getenv("CACHE_LOOKBACK", 41))
    START_DATE: str = os.getenv("CACHE_START_DATE", "1998-01-01")
    END_DATE: str = os.getenv("CACHE_END_DATE", "2030-01-01")

    @classmethod
    def get_filename(cls) -> str:
        """Generates a standardized, descriptive parquet filename."""
        import pandas as pd

        start_yr = pd.Timestamp(cls.START_DATE).strftime("%Y")
        return f"alpha_cache_{cls.LOOKBACK}d_{start_yr}.parquet"


@dataclass
class StrategyParams:
    standard_confidence: float = 1.0
    strong_confidence: float = 1.5
    extreme_confidence: float = 2.5
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    range_high: float = 0.8
    range_low: float = 0.2


@dataclass
class QualityThresholds:
    min_median_dollar_volume: int = 1_000_000
    min_liquidity_percentile: float = 0.40
    max_stale_pct: float = 0.05
    max_same_vol_count: int = 10


def find_checkpoint_path(artifact_source: str | Path) -> Optional[Path]:
    """
    Resolves the matching PyTorch model checkpoint (.pt/.pth) from a results pickle,
    blotter parquet, model stem, or direct checkpoint path.
    """
    path = Path(artifact_source)
    if path.is_file() and path.suffix in (".pt", ".pth"):
        return path

    checkpoint_dir = OUTPUT_DIR / "model_checkpoints"
    stem = path.stem
    stem_clean = re.sub(r"^(?:results_|blotter_df_)", "", stem)

    candidate_stems = [stem_clean]
    if not stem_clean.startswith("model_"):
        candidate_stems.append(f"model_{stem_clean}")

    for s in candidate_stems:
        for ext in (".pt", ".pth"):
            cand1 = checkpoint_dir / f"{s}{ext}"
            if cand1.is_file():
                return cand1
            cand2 = OUTPUT_DIR / f"{s}{ext}"
            if cand2.is_file():
                return cand2

    for search_dir in (checkpoint_dir, OUTPUT_DIR):
        if search_dir.is_dir():
            matches = list(search_dir.glob(f"*{stem_clean}*.pt"))
            if matches:
                return max(matches, key=lambda p: p.stat().st_mtime)

    return None


def extract_run_hyperparameters(
    artifact_source: str | Path,
    metadata: Optional[dict] = None,
    cfg: Optional["TradingConfig"] = None,
    blotter_df: Any = None,
) -> dict[str, Any]:
    """
    Dynamically extracts run hyperparameters prioritizing PyTorch checkpoint payload['grid_params'],
    with metadata, blotter deduction, filename regex, and dynamic config as fallbacks.
    """
    metadata = metadata or {}
    filename_str = Path(artifact_source).name
    cfg_obj = cfg if cfg is not None else TradingConfig()

    # 1. Attempt loading payload['grid_params'] from PyTorch checkpoint (.pt)
    pt_path = find_checkpoint_path(artifact_source)
    payload: dict[str, Any] = {}
    grid_params: dict[str, Any] = {}
    if pt_path is not None:
        try:
            import torch

            loaded = torch.load(pt_path, map_location="cpu", weights_only=False)
            if isinstance(loaded, dict):
                payload = loaded
                raw_gp = payload.get("grid_params", {})
                if isinstance(raw_gp, dict):
                    grid_params = raw_gp
        except Exception:
            pass

    params: dict[str, Any] = {}

    # 1. min_basket_width
    if "min_basket_width" in grid_params:
        params["min_basket_width"] = int(grid_params["min_basket_width"])
    elif "min_width" in grid_params:
        params["min_basket_width"] = int(grid_params["min_width"])
        params["min_width"] = int(grid_params["min_width"])
    elif "min_basket_width" in metadata:
        params["min_basket_width"] = int(metadata["min_basket_width"])
    elif "min_width" in metadata:
        params["min_basket_width"] = int(metadata["min_width"])
    else:
        w_match = re.search(r"_(?:Width|width|w)([0-9]+)(?:_|$)", filename_str)
        if w_match:
            params["min_basket_width"] = int(w_match.group(1))
        else:
            params["min_basket_width"] = int(getattr(cfg_obj, "min_basket_width", 5))

    # 2. min_active_tilt
    if "min_active_tilt" in grid_params:
        val = float(grid_params["min_active_tilt"])
        params["min_active_tilt"] = val / 100.0 if val > 1.0 else val
    elif "tilt" in grid_params:
        val = float(grid_params["tilt"])
        params["min_active_tilt"] = val / 100.0 if val > 1.0 else val
    elif "min_active_tilt" in metadata:
        val = float(metadata["min_active_tilt"])
        params["min_active_tilt"] = val / 100.0 if val > 1.0 else val
    else:
        tilt_match = re.search(
            r"_(?:tilt|Tilt|atilt)_?([0-9.]+)(?:_|$)", filename_str
        ) or re.search(r"atilt_([0-9.]+)", filename_str)
        if tilt_match:
            val = float(tilt_match.group(1))
            params["min_active_tilt"] = val / 100.0 if val > 1.0 else val
        else:
            params["min_active_tilt"] = float(getattr(cfg_obj, "min_active_tilt", 0.20))

    # 3. upside_alpha_mult
    if "UPSIDE_ALPHA_MULT" in grid_params:
        params["upside_alpha_mult"] = float(grid_params["UPSIDE_ALPHA_MULT"])
    elif "upside_alpha_mult" in grid_params:
        params["upside_alpha_mult"] = float(grid_params["upside_alpha_mult"])
    elif "upside_alpha_mult" in metadata:
        params["upside_alpha_mult"] = float(metadata["upside_alpha_mult"])
    else:
        mult_match = re.search(r"_(?:mult|upside|up)_?([0-9.]+)(?:_|$)", filename_str)
        if mult_match:
            raw_val = mult_match.group(1)
            if "." in raw_val:
                params["upside_alpha_mult"] = float(raw_val)
            elif len(raw_val) >= 2 and raw_val.startswith("1"):
                params["upside_alpha_mult"] = float(f"{raw_val[0]}.{raw_val[1:]}")
            else:
                params["upside_alpha_mult"] = float(raw_val)
        else:
            params["upside_alpha_mult"] = float(
                getattr(cfg_obj, "upside_alpha_mult", 1.0)
            )

    # 4. loss_aversion_penalty
    if "loss_aversion_penalty" in grid_params:
        params["loss_aversion_penalty"] = float(grid_params["loss_aversion_penalty"])
    elif "downside_penalty" in grid_params:
        params["loss_aversion_penalty"] = max(
            0.0, float(grid_params["downside_penalty"]) - 1.0
        )
    elif "loss_aversion_penalty" in metadata:
        params["loss_aversion_penalty"] = float(metadata["loss_aversion_penalty"])
    elif "downside_penalty" in metadata:
        params["loss_aversion_penalty"] = max(
            0.0, float(metadata["downside_penalty"]) - 1.0
        )
    else:
        lossav_match = re.search(r"_(?:lossav|loss)_?([0-9.]+)(?:_|$)", filename_str)
        pen_match = re.search(r"_pen_?([0-9.]+)(?:_|$)", filename_str)
        if lossav_match:
            raw_val = lossav_match.group(1)
            if "." in raw_val:
                params["loss_aversion_penalty"] = float(raw_val)
            elif raw_val.startswith("0") and len(raw_val) > 1:
                params["loss_aversion_penalty"] = float(f"0.{raw_val[1:]}")
            else:
                params["loss_aversion_penalty"] = float(raw_val)
        elif pen_match:
            params["loss_aversion_penalty"] = max(0.0, float(pen_match.group(1)) - 1.0)
        else:
            params["loss_aversion_penalty"] = float(
                getattr(cfg_obj, "loss_aversion_penalty", 0.0)
            )

    # 5. max_cash_pct
    if "max_cash_pct" in grid_params:
        params["max_cash_pct"] = float(grid_params["max_cash_pct"])
    elif "max_cash_pct" in metadata:
        params["max_cash_pct"] = float(metadata["max_cash_pct"])
    else:
        cash_match = re.search(r"_cash_?([0-9.]+)(?:_|$)", filename_str)
        if cash_match:
            params["max_cash_pct"] = float(cash_match.group(1))
        else:
            params["max_cash_pct"] = float(getattr(cfg_obj, "max_cash_pct", 0.0))

    # 6. slippage_rate
    if "slippage_rate" in grid_params:
        params["slippage_rate"] = float(grid_params["slippage_rate"])
    elif "slippage_rate" in metadata:
        params["slippage_rate"] = float(metadata["slippage_rate"])
    else:
        params["slippage_rate"] = float(getattr(cfg_obj, "slippage_rate", 0.0010))

    # 7. holding_period (H)
    h_deduced = None
    if "holding_period" in payload:
        h_deduced = int(payload["holding_period"])
    elif "holding_period" in grid_params:
        h_deduced = int(grid_params["holding_period"])
    elif "H" in grid_params:
        h_deduced = int(grid_params["H"])

    if (
        h_deduced is None
        and blotter_df is not None
        and hasattr(blotter_df, "columns")
        and "slippage_daily_simple_loss" in blotter_df.columns
        and "weight_active" in blotter_df.columns
    ):
        valid_slips = blotter_df[
            (blotter_df["slippage_daily_simple_loss"] > 1e-6)
            & (blotter_df["weight_active"] > 1e-6)
        ]
        if not valid_slips.empty:
            row_slip = valid_slips.iloc[0]
            calc_h = (
                params["slippage_rate"]
                * float(row_slip["weight_active"])
                / float(row_slip["slippage_daily_simple_loss"])
            )
            h_rounded = int(round(calc_h))
            if 1 <= h_rounded <= 100:
                h_deduced = h_rounded

    if h_deduced is None:
        if "holding_period" in metadata:
            h_deduced = int(metadata["holding_period"])
        elif "H" in metadata:
            h_deduced = int(metadata["H"])

    if h_deduced is None:
        hp_match = re.search(r"_(?:hold|holding|hp|T)([0-9]+)(?:_|$)", filename_str)
        if hp_match:
            h_deduced = int(hp_match.group(1))
        else:
            h_deduced = int(getattr(cfg_obj, "holding_period", 5))

    params["holding_period"] = h_deduced

    # 8. gamma
    if "GAMMA" in grid_params:
        params["gamma"] = float(grid_params["GAMMA"])
    elif "gamma" in grid_params:
        params["gamma"] = float(grid_params["gamma"])
    elif "gamma" in metadata:
        params["gamma"] = float(metadata["gamma"])
    else:
        g_match = re.search(r"_(?:gamma|g|Gamma)_?([0-9.]+)(?:_|$)", filename_str)
        if g_match:
            raw_g = float(g_match.group(1))
            params["gamma"] = raw_g / 100.0 if raw_g > 1.0 else raw_g
        else:
            params["gamma"] = float(getattr(cfg_obj, "gamma", 0.90))

    # 9. benchmark
    if "benchmark" in payload:
        params["benchmark"] = str(payload["benchmark"])
    elif "benchmark" in grid_params:
        params["benchmark"] = str(grid_params["benchmark"])
    elif "benchmark" in metadata:
        params["benchmark"] = str(metadata["benchmark"])
    else:
        params["benchmark"] = str(getattr(cfg_obj, "benchmark", "SPY"))

    return params


@dataclass
class TradingConfig:
    # ENVIRONMENT
    benchmark_ticker: str = "SPY"
    calendar_ticker: str = "SPY"

    # DATA SANITIZER
    handle_zeros_as_nan: bool = True
    max_data_gap_ffill: int = 1

    # ------------------------------------------------------------------------
    # RULE: SYSTEM-WIDE NaN & ZERO HANDLING POLICY
    # ------------------------------------------------------------------------
    # 1. Prices (Open, High, Low, Close) MUST remain NaN if the asset did not
    #    trade (e.g., Pre-IPO, Halted, Delisted).
    # 2. NEVER fill missing prices with 0.0. This causes division-by-zero,
    #    infinite returns, and breaks pipeline mathematics.
    # 3. Features (like ATRP, TRP, RSI) can be filled with 0.0 where appropriate
    #    to allow cross-sectional math to proceed without dropping the row.
    # 4. Use .bfill() in portfolio simulation (QuantUtils) only to align newly
    #    IPO'd stocks mid-period against the initial capital allocation.
    # ------------------------------------------------------------------------
    nan_price_replacement: float = 0.0  # DEPRECATED/UNSAFE: DO NOT USE FOR PRICES

    # STRATEGY & MATH
    annual_period: int = 252
    atr_period: int = 14
    rsi_period: int = 14
    range_pos_period: int = 20

    # FEATURE ENGINE WINDOWS
    win_5d: int = 5
    win_21d: int = 21
    win_63d: int = 63
    win_126d: int = 126
    win_252d: int = 252

    # FEATURE GUARDRAILS (CLIPS)
    feature_zscore_clip: float = 4.0
    feature_ratio_clip: float = 10.0

    # QUALITY/LIQUIDITY
    quality_window: int = 252
    quality_min_periods: int = 126

    # STRATEGY PARAMETERS & THRESHOLDS
    strategy_params: StrategyParams = field(default_factory=StrategyParams)
    thresholds: QualityThresholds = field(default_factory=QualityThresholds)

    # TRAINING & SIMULATION PARAMETERS (Strategic Pivot Calibrated)
    holding_period: int = 5
    rank_max_offset_percentile: float = 1.0
    rank_max_width: int = 10
    min_basket_width: int = 5  # Diversify: 5 to 10 stocks dampens idiosyncratic drag
    max_cash_pct: float = 0.0  # 100% Gross Equity Exposure (Zero Cash)
    min_active_tilt: float = (
        0.20  # Flexibility: Allows up to 80% benchmark beta shelter
    )

    # ENVIRONMENT CONTROLS
    randomize_start: bool = False
    episode_steps: int = 0

    # INSTITUTIONAL RL REWARD PARAMETERS
    slippage_rate: float = 0.0010  # 10 bps round-trip slippage
    loss_aversion_penalty: float = 0.0  # 0.0 = Linear spread
    upside_alpha_mult: float = 1.0  # 1.0 = Strict linear spread (NO convex gambling)

    # GAE & PPO HYPERPARAMETERS
    gamma: float = 0.90
    gae_lambda: float = 0.95
    learning_rate: float = 2.0e-4
    critic_learning_rate: float = 8.0e-4

    # ENTROPY ANNEALING SCHEDULE
    entropy_coef: float = 0.0050
    entropy_coef_start: float = 0.0050  # Broader exploration
    entropy_coef_end: float = 0.0005  # Controlled convergence

    # PPO BATCHING & LOSS COEFFICIENTS
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    num_steps: int = 512  # 512 steps * 8 envs = 4,096 timesteps per epoch
    num_envs: int = 8
    mini_batch_size: int = 256

    @property
    def dynamic_gamma(self) -> float:
        """Returns self.gamma to prevent accidental override to legacy overlapping horizon values."""
        return self.gamma

    @property
    def benchmark(self) -> str:
        """Dynamic alias for benchmark_ticker conforming to TradingConfig domain model."""
        return self.benchmark_ticker

    @benchmark.setter
    def benchmark(self, val: str) -> None:
        self.benchmark_ticker = str(val)

    @classmethod
    def from_artifact(
        cls,
        artifact_source: str | Path,
        metadata: Optional[dict] = None,
        blotter_df: Any = None,
    ) -> "TradingConfig":
        """Factory creating a TradingConfig dynamically populated from checkpoint payload grid_params."""
        cfg = cls()
        params = extract_run_hyperparameters(artifact_source, metadata, cfg, blotter_df)
        for k, v in params.items():
            if k == "benchmark":
                cfg.benchmark_ticker = str(v)
            elif hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg
