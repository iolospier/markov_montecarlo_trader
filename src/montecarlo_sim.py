import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from src.stochastic_models import simulate_regime_gbm
from src.utils.logger import get_logger
from src.utils.timer import timed


@timed
def run_monte_carlo(
    P,
    params,
    n_paths=1000,
    n_steps=252,
    S0=100.0,
    start_state="Neutral",
    rng=None,
    logger=None,
):
    """Run Monte Carlo simulation with a shared RNG for full reproducibility."""
    logger = logger or get_logger(".")
    rng = rng or np.random.default_rng()

    all_prices = np.zeros((n_paths, n_steps))
    all_returns = np.zeros(n_paths)
    logger.info(f"Running Monte Carlo: {n_paths} paths × {n_steps} steps")

    def simulate_path(_):
        try:
            prices, _ = simulate_regime_gbm(
                P, start_state, params, n_steps, int(S0), rng=rng, logger=logger
            )
            return prices, prices[-1] / prices[0] - 1
        except Exception as e:
            logger.warning(f"Path failed: {e}")
            return (
                np.full(n_steps, np.nan),
                np.nan,
            )  # Ensure a valid tuple is always returned

    from joblib import Parallel, delayed

    results = Parallel(n_jobs=-1)(delayed(simulate_path)(i) for i in range(n_paths))

    for i, result in enumerate(results):
        if result is None:
            logger.warning(f"Skipping path {i} due to None result.")
            continue
        prices, ret = result
        all_prices[i, :] = prices
        all_returns[i] = ret

    logger.info("Monte Carlo simulation complete.")
    return {
        "prices": all_prices,
        "final_prices": all_prices[:, -1],
        "returns": all_returns,
    }
