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
    """Run Monte Carlo simulation with shared RNG and return prices + regimes."""
    logger = logger or get_logger(".")
    rng = rng or np.random.default_rng()

    all_prices = np.zeros((n_paths, n_steps))
    all_returns = np.zeros(n_paths)
    all_regimes = []
    logger.info(f"Running Monte Carlo: {n_paths} paths × {n_steps} steps")

    def simulate_path(_):
        try:
            prices, regimes = simulate_regime_gbm(
                P,
                start_state,
                params,
                n_steps,
                int(S0),
                rng=rng,
                logger=logger,
            )
            return prices, regimes, prices[-1] / prices[0] - 1
        except Exception as e:
            logger.warning(f"Path failed: {e}")
            # Return dummy arrays to preserve alignment
            return np.full(n_steps, np.nan), [], np.nan  # Ensure valid structure

    results = Parallel(n_jobs=-1)(delayed(simulate_path)(i) for i in range(n_paths))

    for i, result in enumerate(results):
        if result is None or len(result) != 3:
            logger.warning(f"Skipping path {i}: invalid result structure.")
            continue
        prices, regimes, ret = result
        if prices is None or len(prices) == 0:
            logger.warning(f"Skipping path {i}: empty prices.")
            continue
        all_prices[i, :] = prices
        all_returns[i] = ret
        all_regimes.append(regimes)

    logger.info("Monte Carlo simulation complete.")
    logger.info(
        f"Monte Carlo output shapes: {all_prices.shape}, "
        f"regimes count={len(all_regimes)}"
    )

    return {
        "prices": all_prices,
        "final_prices": all_prices[:, -1],
        "returns": all_returns,
        "regimes": all_regimes,
    }
