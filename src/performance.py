import numpy as np
import pandas as pd


def compute_performance_metrics(pnl: np.ndarray):
    """
    Compute performance metrics for a single PnL time series.
    """
    # Daily returns from PnL changes
    daily_ret = np.diff(pnl)
    if len(daily_ret) == 0:
        return None

    mean_ret = np.mean(daily_ret)
    vol = np.std(daily_ret)
    sharpe = mean_ret / vol if vol > 0 else np.nan

    # Max drawdown
    equity_curve = 1 + pnl
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    max_dd = np.max(drawdown)

    # Win rate
    win_rate = np.mean(daily_ret > 0)

    return {
        "mean_daily_return": mean_ret,
        "volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "final_return": pnl[-1],
    }


def aggregate_performance(all_pnls: list[np.ndarray]):
    """
    Compute metrics for multiple Monte Carlo simulations and return aggregate summary.
    """
    results = [compute_performance_metrics(p) for p in all_pnls if p is not None]
    df = pd.DataFrame(results)
    summary = df.describe().T[["mean", "std", "min", "max"]]
    return df, summary
