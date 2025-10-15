import numpy as np
import pandas as pd


def simulate_gbm(S0: float, mu: float, sigma: float, n_steps: int, dt: float = 1 / 252):
    """
    Simulate a single Geometric Brownian Motion (GBM) price path.
    """
    # Random shocks (Brownian increments)
    W = np.random.standard_normal(n_steps)
    W = np.cumsum(W) * np.sqrt(dt)
    t = np.linspace(0, n_steps * dt, n_steps)

    # GBM formula
    S = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)
    return S


def simulate_regime_gbm(
    P: pd.DataFrame,
    start_state: str,
    params: dict,
    n_steps: int = 252,
    S0: float = 100,
):
    """
    Simulate a regime-switching GBM price path using a Markov chain.

    Parameters
    ----------
    P : pd.DataFrame
        Transition matrix between regimes.
    start_state : str
        Starting regime, e.g. "Neutral".
    params : dict
        Dictionary of regime-specific (mu, sigma) parameters.
        Example:
            {
                "Bull": {"mu": 0.0005, "sigma": 0.01},
                "Neutral": {"mu": 0.0, "sigma": 0.007},
                "Bear": {"mu": -0.0005, "sigma": 0.015}
            }
    n_steps : int
        Number of simulation steps (e.g. 252 for 1 trading year).
    S0 : float
        Initial price.

    Returns
    -------
    prices : np.ndarray
        Simulated price series.
    regimes : list[str]
        Sequence of regimes corresponding to each price.
    """
    states = list(P.index)
    current_state = start_state
    prices = [S0]
    regime_seq = [current_state]

    for _ in range(n_steps - 1):
        # Determine next regime via Markov transition probabilities
        probs = P.loc[current_state].to_numpy(dtype=float)
        next_state = np.random.choice(states, p=probs)
        regime_seq.append(next_state)
        current_state = next_state

        # Regime-specific GBM step
        mu = params[next_state]["mu"]
        sigma = params[next_state]["sigma"]
        dt = 1 / 252
        dW = np.random.normal(0, np.sqrt(dt))
        new_price = prices[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
        prices.append(new_price)

    return np.array(prices), regime_seq
