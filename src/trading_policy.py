import numpy as np
from src.utils.logger import get_logger


def apply_trading_policy(
    prices: np.ndarray,
    regimes: list[str],
    slippage: float = 0.0002,
    commission: float = 0.0001,
    max_drawdown: float = 0.15,
    persistence: int = 3,
    long_only: bool = False,
    vol_weight: bool = False,
    regime_vols: dict | None = None,
    logger=None,
):
    """Apply a regime-based trading policy with optional persistence,
    volatility weighting, and long-only constraints.
    """
    logger = logger or get_logger(".")
    n = len(prices)
    positions = np.zeros(n)
    pnl = np.zeros(n)
    equity = 1.0
    peak_equity = 1.0

    # Smooth regimes for persistence
    stable_regimes = regimes.copy()
    for i in range(persistence, n):
        recent = regimes[i - persistence : i]
        if len(set(recent)) == 1:
            stable_regimes[i] = regimes[i]
        else:
            stable_regimes[i] = stable_regimes[i - 1]

    # Assign positions by regime
    for t in range(1, n):
        r = stable_regimes[t]
        if long_only:
            positions[t] = 1 if r == "Bull" else 0
        else:
            positions[t] = 1 if r == "Bull" else -1 if r == "Bear" else 0

        # Volatility weighting
        if vol_weight and regime_vols is not None:
            sigma = regime_vols.get(r, 1e-6)
            norm_factor = 1 / max(sigma, 1e-6)
            positions[t] *= norm_factor / 100  # keep leverage realistic

    # Compute daily returns
    returns = np.diff(prices) / prices[:-1]

    for t in range(1, n):
        trade_cost = slippage + commission if positions[t] != positions[t - 1] else 0.0
        pnl[t] = pnl[t - 1] + positions[t - 1] * returns[t - 1] - trade_cost
        equity = 1 + pnl[t]
        peak_equity = max(peak_equity, equity)

        # Drawdown stop
        if (peak_equity - equity) / peak_equity > max_drawdown:
            positions[t:] = 0
            pnl[t:] = pnl[t]
            logger.info(f"Max drawdown triggered at step {t}, trading halted.")
            break

    flips = np.sum(np.diff(positions) != 0)
    turnover = flips / len(positions)

    logger.debug(
        f"Trading policy applied: long_only={long_only}, "
        f"vol_weight={vol_weight}, persistence={persistence}, "
        f"turnover={turnover:.4f}"
    )

    return pnl, positions, turnover
