# main.py
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
from src.trading_policy import apply_trading_policy
from src.performance import aggregate_performance


@timed
def main():
    # Setup
    config = load_config("config/params.yaml")
    run_dir = create_run_dirs("results")
    seed = config.get("general", {}).get("seed", 42)
    set_global_seed(seed)
    logger = get_logger(run_dir)

    logger.info("=== Starting Markov Monte Carlo Simulation ===")

    # Data
    df = init_data(
        config["data"]["ticker"],
        config["data"]["start_date"],
        config["data"]["save_path"],
        logger=logger,
    )
    df = classify_regimes(df, logger=logger)
    P = estimate_transition_matrix(df, logger=logger)
    params = estimate_regime_params(df, logger=logger)

    # Simulation
    sim_cfg = config["simulation"]
    prices, regimes = simulate_regime_gbm(
        P,
        start_state=sim_cfg["start_state"],
        params=params,
        n_steps=sim_cfg["n_steps"],
        S0=sim_cfg["S0"],
        seed=seed,
        logger=logger,
    )

    regime_vols = {k: v["sigma"] for k, v in params.items()}

    pnl, positions, turnover = apply_trading_policy(
        prices,
        regimes,
        slippage=config["trading"]["slippage"],
        commission=config["trading"]["commission"],
        max_drawdown=config["trading"]["max_drawdown"],
        persistence=config["trading"]["persistence"],
        long_only=config["trading"]["long_only"],
        vol_weight=config["trading"]["vol_weight"],
        regime_vols=regime_vols,
        logger=logger,
    )

    # Performance
    aggregate_performance([pnl], run_dir=run_dir, logger=logger)

    logger.info("Simulation complete. Results saved.")
    logger.info(f"All outputs in: {run_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        print("Fatal error during run:", e)
        traceback.print_exc()
