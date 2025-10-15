from src.data_loader import init_data
from src.regimes import (
    classify_regimes,
    estimate_transition_matrix,
    estimate_regime_params,
)
from src.montecarlo_sim import run_monte_carlo
from src.trading_policy import apply_trading_policy
from src.performance import aggregate_performance
import matplotlib.pyplot as plt
from src.stochastic_models import simulate_regime_gbm
import numpy as np

import yaml

with open("config/params.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# load and preprocess data
data_cfg = cfg["data"]
df = init_data(data_cfg["ticker"], data_cfg["start_date"], data_cfg["save_path"])
df = classify_regimes(df)
P = estimate_transition_matrix(df)
params = estimate_regime_params(df)

# stress test dependent on config
if cfg["regimes"]["auto_estimate"]:
    scale_mu = cfg["regimes"]["scale_mu"]
    for r in params:
        params[r]["mu"] *= scale_mu

# run monte carlo simulations
sim_cfg = cfg["simulation"]
trading_cfg = cfg["trading"]
N = sim_cfg["n_paths"]

all_pnls = []
for i in range(N):
    prices, regimes = simulate_regime_gbm(
        P,
        start_state=sim_cfg["start_state"],
        params=params,
        n_steps=sim_cfg["n_steps"],
        S0=sim_cfg["S0"],
    )
    regime_vols = {k: v["sigma"] for k, v in params.items()}

    pnl, positions = apply_trading_policy(
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

# evaluate performance
df_metrics, summary = aggregate_performance(all_pnls)

print("\nAggregate Performance Summary:")
print(summary.round(6))

if cfg["performance"]["plot_sharpe_hist"]:
    plt.figure(figsize=(8, 4))
    plt.hist(df_metrics["sharpe"].dropna(), bins=30, edgecolor="k", alpha=0.7)
    plt.title("Distribution of Sharpe Ratios (Monte Carlo Simulations)")
    plt.xlabel("Sharpe Ratio")
    plt.ylabel("Frequency")
    plt.show()
