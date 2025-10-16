import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from src.stochastic_models import simulate_regime_gbm
from src.utils.logger import get_logger
from src.utils.timer import timed


@timed
def run_monte_carlo(
    P: pd.DataFrame,
    params: dict,
    n_paths: int = 1000,
    n_steps: int = 252,
    S0: float = 100.0,
    start_state: str = "Neutral",
    random_seed: int | None = None,
    logger=None,
):
    """
    Run a Monte Carlo simulation of regime-switching GBM price paths.
    """
    logger = logger or get_logger(".")
    rng = np.random.default_rng(random_seed)

    all_prices = np.zeros((n_paths, n_steps))
    all_returns = np.zeros(n_paths)

    logger.info(f"Running Monte Carlo simulation: {n_paths} paths × {n_steps} steps")

    def simulate_path(i):
        local_seed = None if random_seed is None else random_seed + i
        try:
            prices, _ = simulate_regime_gbm(
                P, start_state, params, n_steps, S0, seed=local_seed
            )
            return prices, prices[-1] / prices[0] - 1
        except Exception as e:
            logger.warning(f"Path {i} failed: {e}")
            return np.full(n_steps, np.nan), np.nan

    results = Parallel(n_jobs=-1)(delayed(simulate_path)(i) for i in range(n_paths))

    valid_results = [res for res in results if res is not None]
    for i, (prices, ret) in enumerate(valid_results):
        all_prices[i, :] = prices
        all_returns[i] = ret

    summary = {
        "prices": all_prices,
        "final_prices": all_prices[:, -1],
        "returns": all_returns,
    }

    logger.info("Monte Carlo simulation complete.")
    return summary
