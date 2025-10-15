import numpy as np
import pandas as pd
from src.stochastic_models import simulate_regime_gbm


def run_monte_carlo(
    P: pd.DataFrame,
    params: dict,
    n_paths: int = 1000,
    n_steps: int = 252,
    S0: float = 100.0,
    start_state: str = "Neutral",
    random_seed: int | None = None,
):
    """
    Run a Monte Carlo simulation of regime-switching GBM price paths.

    Parameters
    ----------
    P : pd.DataFrame
        Transition matrix between regimes.
    params : dict
        Dictionary of regime-specific (mu, sigma) parameters.
    n_paths : int
        Number of Monte Carlo simulations.
    n_steps : int
        Number of time steps per simulation.
    S0 : float
        Initial price.
    start_state : str
        Starting regime.
    random_seed : int or None
        Seed for reproducibility.

    Returns
    -------
    results : dict
        {
            "prices": np.ndarray of shape (n_paths, n_steps),
            "final_prices": np.ndarray of shape (n_paths,),
            "returns": np.ndarray of shape (n_paths,),
        }
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    all_prices = np.zeros((n_paths, n_steps))
    all_returns = np.zeros(n_paths)

    for i in range(n_paths):
        prices, _ = simulate_regime_gbm(P, start_state, params, n_steps, S0)
        all_prices[i, :] = prices
        all_returns[i] = prices[-1] / prices[0] - 1

    results = {
        "prices": all_prices,
        "final_prices": all_prices[:, -1],
        "returns": all_returns,
    }

    return results
