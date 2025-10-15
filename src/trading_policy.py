import numpy as np
import pandas as pd


def apply_trading_policy(
    prices: np.ndarray,
    regimes: list[str],
    slippage: float = 0.0002,  # 0.0002 = 2 bps
    commission: float = 0.0001,  # 0.0001 = 1 bps
    max_drawdown: float = 0.15,
):
    """
    Apply a simple trading strategy based on regimes:
      - Long in Bull
      - Short in Bear
      - Flat in Neutral

    Includes transaction costs, slippage, and drawdown control.

    Parameters
    ----------
    prices : np.ndarray
        Simulated price series.
    regimes : list[str]
        Regime sequence corresponding to each price.
    slippage : float
        Fractional slippage per trade (e.g. 0.0002 = 2 bps).
    commission : float
        Fixed commission cost per trade.
    max_drawdown : float
        Maximum tolerated drawdown (as fraction of equity).

    Returns
    -------
    pnl : np.ndarray
        Cumulative PnL over time.
    positions : np.ndarray
        Position held at each step (+1, 0, -1).
    """
    n = len(prices)
    positions = np.zeros(n)
    pnl = np.zeros(n)
    equity = 1.0  # start with 1 unit of equity
    peak_equity = 1.0

    # Assign positions by regime
    for t in range(1, n):
        if regimes[t] == "Bull":
            positions[t] = 1
        elif regimes[t] == "Bear":
            positions[t] = -1
        else:
            positions[t] = 0

    # Compute daily returns from price changes
    returns = np.diff(prices) / prices[:-1]

    # Apply trading PnL with transaction costs
    for t in range(1, n):
        trade_cost = 0.0
        if positions[t] != positions[t - 1]:
            trade_cost = slippage + commission

        pnl[t] = pnl[t - 1] + positions[t - 1] * returns[t - 1] - trade_cost
        equity = 1 + pnl[t]
        peak_equity = max(peak_equity, equity)

        # Max drawdown check
        if (peak_equity - equity) / peak_equity > max_drawdown:
            positions[t:] = 0  # close out all positions
            pnl[t:] = pnl[t]
            break

    return pnl, positions
