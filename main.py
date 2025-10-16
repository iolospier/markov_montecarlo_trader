#  main.py
from src.utils.config_loader import load_config
from src.utils.seed_utils import set_global_seed
from src.utils.io_utils import create_run_dirs
from src.utils.logger import get_logger
from src.utils.timer import timed
from src.data_loader import init_data
from src.regimes import (
    classify_regimes,
    estimate_transition_matrix,
    estimate_regime_params,
)
from src.stochastic_models import simulate_regime_gbm
from src.montecarlo_sim import run_monte_carlo
from src.trading_policy import apply_trading_policy
from src.performance import aggregate_performance
import pandas as pd
import os
import numpy as np
import glob


@timed
def main():
    # Setup
    config = load_config("config/params.yaml")
    run_dir = create_run_dirs("results")
    seed = config.get("general", {}).get("seed", 42)
    set_global_seed(seed)
    GLOBAL_RNG = np.random.default_rng(seed)
    logger = get_logger(run_dir)

    logger.info("=== Starting MultiTicker Markov Monte Carlo Simulation ===")

    #  Load Data
    data_cfg = config["data"]
    datasets = init_data(
        tickers=data_cfg.get("tickers"),
        ticker=data_cfg.get("ticker"),
        start=data_cfg["start_date"],
        save_dir=data_cfg.get("save_dir", "data"),
        logger=logger,
    )

    #  Loop over each ticker
    for ticker, df in datasets.items():
        logger.info(f"--- Processing {ticker} ---")
        ticker_dir = os.path.join(run_dir, ticker)
        os.makedirs(ticker_dir, exist_ok=True)

        # Regime classification & parameter estimation
        df = classify_regimes(df, logger=logger)
        P = estimate_transition_matrix(df, logger=logger)
        params = estimate_regime_params(df, logger=logger)
        regime_vols = {k: v["sigma"] for k, v in params.items()}

        #  Simulation
        sim_cfg = config["simulation"]

        results = run_monte_carlo(
            P,
            params,
            n_paths=sim_cfg["n_paths"],
            n_steps=sim_cfg["n_steps"],
            S0=sim_cfg["S0"],
            start_state=sim_cfg["start_state"],
            rng=GLOBAL_RNG,
            logger=logger,
        )

        all_pnls = []
        for i in range(sim_cfg["n_paths"]):
            prices = results["prices"][i]
            # Use the corresponding regime sequence if it exists
            regimes = results.get("regimes", [])
            regimes_i = regimes[i] if i < len(regimes) else []

            if len(prices) == 0 or len(regimes_i) == 0:
                logger.warning(f"Skipping path {i} due to empty prices/regimes.")
                continue

            pnl, positions, turnover = apply_trading_policy(
                prices,
                regimes_i,
                slippage=config["trading"]["slippage"],
                commission=config["trading"]["commission"],
                max_drawdown=config["trading"]["max_drawdown"],
                persistence=config["trading"]["persistence"],
                long_only=config["trading"]["long_only"],
                vol_weight=config["trading"]["vol_weight"],
                regime_vols={k: v["sigma"] for k, v in params.items()},
                logger=logger,
            )
            all_pnls.append(pnl)

        aggregate_performance(all_pnls, run_dir=ticker_dir, logger=logger)

        logger.info(f"Finished {ticker}. Results saved to {ticker_dir}")

    #  Combine all ticker results
    all_summaries = []
    for f in glob.glob(os.path.join(run_dir, "*/performance_summary.csv")):
        try:
            df = pd.read_csv(f)
            ticker = os.path.basename(os.path.dirname(f))
            df["ticker"] = ticker
            all_summaries.append(df)
        except Exception as e:
            logger.warning(f"Failed to read {f}: {e}")

    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined_path = os.path.join(run_dir, "aggregate_results.csv")
        combined.to_csv(combined_path, float_format="%.6f", index=False)
        logger.info(f"Saved combined summary to {combined_path}")
    else:
        logger.warning("No performance summaries found to aggregate.")

    logger.info("All simulations complete.")
    logger.info(f"All outputs in: {run_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        print("Fatal error during run:", e)
        traceback.print_exc()
