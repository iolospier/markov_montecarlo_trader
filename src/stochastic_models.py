import numpy as np
import pandas as pd
from src.utils.logger import get_logger


def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    n_steps: int,
    dt: float = 1 / 252,
    rng=None,
    logger=None,
):
    """Simulate a single Geometric Brownian Motion (GBM) path using a shared RNG."""
    logger = logger or get_logger(".")
    rng = rng or np.random.default_rng()  # if none provided

    W = rng.standard_normal(n_steps)
    W = np.cumsum(W) * np.sqrt(dt)
    t = np.linspace(0, n_steps * dt, n_steps)
    S = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)
    logger.debug(f"Simulated GBM: mu={mu:.5f}, sigma={sigma:.5f}, steps={n_steps}")
    return S


def simulate_regime_gbm(
    P, start_state, params, n_steps=252, S0=100, rng=None, logger=None
):
    """Simulate a regime-switching GBM using a shared RNG for reproducibility."""
    logger = logger or get_logger(".")
    rng = rng or np.random.default_rng()

    states = list(P.index)
    if start_state not in states:
        raise ValueError(
            f"Invalid start_state '{start_state}', must be one of {states}"
        )

    prices = [S0]
    regime_seq = [start_state]
    current_state = start_state
    dt = 1 / 252

    for _ in range(n_steps - 1):
        probs = P.loc[current_state].values
        next_state = rng.choice(states, p=probs)
        regime_seq.append(next_state)
        current_state = next_state

        mu = params[next_state]["mu"]
        sigma = params[next_state]["sigma"]
        dW = rng.normal(0, np.sqrt(dt))
        new_price = prices[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
        prices.append(new_price)

    logger.debug(f"Simulated regime GBM with {n_steps} steps from {start_state}.")
    return np.array(prices), regime_seq
