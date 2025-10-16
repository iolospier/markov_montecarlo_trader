# ===============================================
#  optuna_optimiser.py
#  Bayesian Optimisation for Markov Monte Carlo Trader
# ===============================================

import os
import optuna
import yaml
import numpy as np
from src.data_loader import init_data
from src.regimes import (
    classify_regimes,
    estimate_transition_matrix,
    estimate_regime_params,
)
from src.stochastic_models import simulate_regime_gbm
from src.trading_policy import apply_trading_policy
from src.performance import aggregate_performance
from src.utils.logger import get_logger
from src.utils.seed_utils import set_global_seed
from src.utils.io_utils import create_run_dirs
from src.utils.timer import timed


def objective(trial):
    """Objective function for Optuna Bayesian optimisation."""
    with open("config/params.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    with open("config/optuna_config.yaml", "r") as f:
        opt_cfg = yaml.safe_load(f)

    # setup logger and seed
    run_dir = opt_cfg.get("run_dir", "results")
    logger = get_logger(run_dir)
    seed = opt_cfg["optuna"].get("seed", 42)
    set_global_seed(seed + trial.number)

    # search space
    ss = opt_cfg["search_space"]
    regimes_cfg = cfg["regimes"]
    trading_cfg = cfg["trading"]

    # sample hyperparameters
    regimes_cfg["scale_mu"] = trial.suggest_float("scale_mu", *ss["scale_mu"])
    trading_cfg["persistence"] = trial.suggest_int("persistence", *ss["persistence"])
    trading_cfg["slippage"] = trial.suggest_float("slippage", *ss["slippage"])
    trading_cfg["commission"] = trial.suggest_float("commission", *ss["commission"])
    trading_cfg["max_drawdown"] = trial.suggest_float(
        "max_drawdown", *ss["max_drawdown"]
    )
    trading_cfg["long_only"] = trial.suggest_categorical("long_only", ss["long_only"])
    trading_cfg["vol_weight"] = trial.suggest_categorical(
        "vol_weight", ss["vol_weight"]
    )

    # data prep
    df = init_data(
        cfg["data"]["ticker"],
        cfg["data"]["start_date"],
        cfg["data"]["save_path"],
        logger=logger,
    )
    df = classify_regimes(df)
    P = estimate_transition_matrix(df)
    params = estimate_regime_params(df)

    # regime scaling
    for r in params:
        params[r]["mu"] *= regimes_cfg["scale_mu"]

    sim_cfg = cfg["simulation"]
    n_paths = sim_cfg["n_paths"]
    all_pnls = []

    logger.info(
        f"Trial {trial.number}: scale_mu={regimes_cfg['scale_mu']:.3f}, "
        f"persistence={trading_cfg['persistence']}, slippage={trading_cfg['slippage']:.5f}, "
        f"commission={trading_cfg['commission']:.5f}"
    )

    # Monte Carlo simulation
    for i in range(n_paths):
        local_seed = seed + i
        try:
            prices, regimes = simulate_regime_gbm(
                P,
                start_state=sim_cfg["start_state"],
                params=params,
                n_steps=sim_cfg["n_steps"],
                S0=sim_cfg["S0"],
                seed=local_seed,
            )

            regime_vols = {k: v["sigma"] for k, v in params.items()}
            pnl, positions, turnover = apply_trading_policy(
                prices,
                regimes,
                slippage=trading_cfg["slippage"],
                commission=trading_cfg["commission"],
                max_drawdown=trading_cfg["max_drawdown"],
                persistence=trading_cfg["persistence"],
                long_only=trading_cfg["long_only"],
                vol_weight=trading_cfg["vol_weight"],
                regime_vols=regime_vols,
            )
            all_pnls.append(pnl)

        except Exception as e:
            logger.warning(f"Path {i} failed in trial {trial.number}: {e}")

    # performance evaluation
    df_metrics, summary = aggregate_performance(all_pnls)
    mean_sharpe = float(np.mean(df_metrics["sharpe"]))
    logger.info(f"Trial {trial.number} Sharpe={mean_sharpe:.4f}")

    return mean_sharpe


@timed
def main():
    """Run Optuna optimisation."""
    with open("config/optuna_config.yaml", "r") as f:
        opt_cfg = yaml.safe_load(f)

    run_dir = create_run_dirs("results")
    logger = get_logger(run_dir)
    set_global_seed(opt_cfg["optuna"].get("seed", 42))

    study = optuna.create_study(
        direction=opt_cfg["optuna"]["direction"],
        sampler=optuna.samplers.TPESampler(seed=opt_cfg["optuna"]["seed"]),
    )

    logger.info("Starting Optuna Bayesian optimisation...")

    study.optimize(
        objective,
        n_trials=opt_cfg["optuna"]["n_trials"],
        timeout=opt_cfg["optuna"]["timeout"],
        show_progress_bar=True,
    )

    logger.info("Optimisation complete.")
    logger.info(f"Best Sharpe: {study.best_value:.4f}")

    for k, v in study.best_params.items():
        logger.info(f"  {k}: {v}")

    study.trials_dataframe().to_csv(
        os.path.join(run_dir, "optuna_trials.csv"), index=False
    )
    with open(os.path.join(run_dir, "optuna_best.yaml"), "w") as f:
        yaml.dump(study.best_params, f)


if __name__ == "__main__":
    main()
