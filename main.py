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

# Load historical data and classify regimes
df = init_data("QQQ", "2018-01-01", "../data/qqq.csv")
df = classify_regimes(df)
P = estimate_transition_matrix(df)
params = estimate_regime_params(df)
scale_mu = 1.5
for r in params:
    params[r]["mu"] *= scale_mu

# Run Monte Carlo simulations
N = 2000  # number of paths
all_pnls = []
for i in range(N):
    prices, regimes = simulate_regime_gbm(P, start_state="Neutral", params=params)
    regime_vols = {k: v["sigma"] for k, v in params.items()}
    pnl, positions = apply_trading_policy(
        prices,
        regimes,
        slippage=0.0002,
        commission=0.0001,
        max_drawdown=0.15,
        persistence=3,
        long_only=False,
        vol_weight=True,
        regime_vols=regime_vols,
    )

    all_pnls.append(pnl)

# Use performance module to assess results
df_metrics, summary = aggregate_performance(all_pnls)

print("\nAggregate Performance Summary:")
print(summary.round(6))

# Visualize distribution of Sharpe ratios
plt.figure(figsize=(8, 4))
plt.hist(df_metrics["sharpe"].dropna(), bins=30, edgecolor="k", alpha=0.7)
plt.title("Distribution of Sharpe Ratios (Monte Carlo Simulations)")
plt.xlabel("Sharpe Ratio")
plt.ylabel("Frequency")
plt.show()
