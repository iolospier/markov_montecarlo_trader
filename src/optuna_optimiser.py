# ===============================================
#  optuna_optimiser.py
#  Bayesian Optimisation for Markov Monte Carlo Trader
# ===============================================

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


def objective(trial):
    # load configs
    with open("config/params.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    with open("config/optuna_config.yaml", "r") as f:
        opt_cfg = yaml.safe_load(f)

    # create search space
    ss = opt_cfg["search_space"]
    regimes_cfg = cfg["regimes"]
    trading_cfg = cfg["trading"]

    # sample hyperparameters
    regimes_cfg["scale_mu"] = trial.suggest_float(
        "scale_mu", ss["scale_mu"][0], ss["scale_mu"][1]
    )
    trading_cfg["persistence"] = trial.suggest_int(
        "persistence", ss["persistence"][0], ss["persistence"][1]
    )
    trading_cfg["slippage"] = trial.suggest_float(
        "slippage", ss["slippage"][0], ss["slippage"][1]
    )
    trading_cfg["commission"] = trial.suggest_float(
        "commission", ss["commission"][0], ss["commission"][1]
    )
    trading_cfg["max_drawdown"] = trial.suggest_float(
        "max_drawdown", ss["max_drawdown"][0], ss["max_drawdown"][1]
    )
    trading_cfg["long_only"] = trial.suggest_categorical("long_only", ss["long_only"])
    trading_cfg["vol_weight"] = trial.suggest_categorical(
        "vol_weight", ss["vol_weight"]
    )

    # preprocess data
    df = init_data(
        cfg["data"]["ticker"], cfg["data"]["start_date"], cfg["data"]["save_path"]
    )
    df = classify_regimes(df)
    P = estimate_transition_matrix(df)
    params = estimate_regime_params(df)

    # stress test dependent on config
    for r in params:
        params[r]["mu"] *= regimes_cfg["scale_mu"]

    sim_cfg = cfg["simulation"]
    N = sim_cfg["n_paths"]
    all_pnls = []

    # monte carlo simulations
    for i in range(N):
        prices, regimes = simulate_regime_gbm(
            P,
            start_state=sim_cfg["start_state"],
            params=params,
            n_steps=sim_cfg["n_steps"],
            S0=sim_cfg["S0"],
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

    # performance eval
    df_metrics, summary = aggregate_performance(all_pnls)
    mean_sharpe = np.mean(df_metrics["sharpe"])

    return float(mean_sharpe)


def main():
    with open("config/optuna_config.yaml", "r") as f:
        opt_cfg = yaml.safe_load(f)

    study = optuna.create_study(
        direction=opt_cfg["optuna"]["direction"],
        sampler=optuna.samplers.TPESampler(seed=opt_cfg["optuna"]["seed"]),
    )

    study.optimize(
        objective,
        n_trials=opt_cfg["optuna"]["n_trials"],
        timeout=opt_cfg["optuna"]["timeout"],
        show_progress_bar=True,
    )

    print("\nBest Parameters Found:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"\nBest Sharpe Ratio: {study.best_value:.4f}")

    # Save results
    study.trials_dataframe().to_csv("results/optuna_trials.csv", index=False)
    with open("results/optuna_best.yaml", "w") as f:
        yaml.dump(study.best_params, f)


if __name__ == "__main__":
    main()
