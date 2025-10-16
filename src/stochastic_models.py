import numpy as np
import pandas as pd
from src.utils.logger import get_logger


def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    n_steps: int,
    dt: float = 1 / 252,
    seed: int | None = None,
    logger=None,
):
    """Simulate a single Geometric Brownian Motion (GBM) price path."""
    logger = logger or get_logger(".")
    rng = np.random.default_rng(seed)

    W = rng.standard_normal(n_steps)
    W = np.cumsum(W) * np.sqrt(dt)
    t = np.linspace(0, n_steps * dt, n_steps)

    S = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)
    logger.debug(f"Simulated GBM path: μ={mu:.5f}, σ={sigma:.5f}, n_steps={n_steps}")
    return S


def simulate_regime_gbm(
    P: pd.DataFrame,
    start_state: str,
    params: dict,
    n_steps: int = 252,
    S0: float = 100,
    seed: int | None = None,
    logger=None,
):
    """
    Simulate a regime-switching GBM price path using a Markov chain.
    """
    logger = logger or get_logger(".")
    rng = np.random.default_rng(seed)

    states = list(P.index)
    if start_state not in states:
        logger.error(f"Invalid start_state '{start_state}'. Must be one of {states}.")
        raise ValueError(f"Invalid start_state '{start_state}'")

    current_state = start_state
    prices = [S0]
    regime_seq = [current_state]

    dt = 1 / 252
    for step in range(n_steps - 1):
        # Sample next regime
        probs = P.loc[current_state].to_numpy(dtype=float)
        next_state = rng.choice(states, p=probs)
        regime_seq.append(next_state)
        current_state = next_state

        # Regime-specific GBM step
        mu = params[next_state]["mu"]
        sigma = params[next_state]["sigma"]
        dW = rng.normal(0, np.sqrt(dt))
        new_price = prices[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
        prices.append(new_price)

    logger.debug(f"Simulated regime GBM with start={start_state}, steps={n_steps}")
    return np.array(prices), regime_seq
