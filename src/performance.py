import numpy as np
import pandas as pd
import os
from src.utils.logger import get_logger


def compute_performance_metrics(pnl: np.ndarray):
    """Compute performance metrics for a single PnL time series."""
    if pnl is None or len(pnl) < 2:
        return None

    daily_ret = np.diff(pnl)
    mean_ret = np.mean(daily_ret)
    vol = np.std(daily_ret)
    sharpe = mean_ret / vol if vol > 0 else np.nan

    equity_curve = 1 + pnl
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    max_dd = np.max(drawdown)

    win_rate = np.mean(daily_ret > 0)

    return {
        "mean_daily_return": mean_ret,
        "volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "final_return": pnl[-1],
    }


def aggregate_performance(
    all_pnls: list[np.ndarray], run_dir: str | None = None, logger=None
):
    """
    Compute metrics for multiple Monte Carlo simulations and return aggregate summary.
    Automatically saves results if a run_dir is provided.
    """
    logger = logger or get_logger(run_dir or ".")
    valid_pnls = [p for p in all_pnls if p is not None and len(p) > 1]

    if not valid_pnls:
        logger.warning("No valid PnL series found. Returning empty results.")
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame([compute_performance_metrics(p) for p in valid_pnls])
    summary = df.describe().T[["mean", "std", "min", "max"]]

    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        perf_csv = os.path.join(run_dir, "performance_summary.csv")
        summary.to_csv(perf_csv)
        logger.info(f"Saved performance summary to {perf_csv}")

    logger.info(
        f"Aggregate Sharpe mean={summary.loc['sharpe', 'mean']:.4f} "
        f"±{summary.loc['sharpe', 'std']:.4f}, "
        f"Max DD mean={summary.loc['max_drawdown', 'mean']:.4f}"
    )

    summary_rounded = summary.round(12)  # round to 6 decimal places
    summary_rounded.to_csv(perf_csv, float_format="%.6f")
    summary_rounded.sort_index(inplace=True)

    return df, summary_rounded
