from multiprocessing import Pool, cpu_count
import itertools
import numpy as np
import pandas as pd
from functools import partial
from src.regimes import (
    classify_regimes,
    estimate_transition_matrix,
    estimate_regime_params,
)
from src.stochastic_models import simulate_regime_gbm
from src.trading_policy import apply_trading_policy
from src.performance import aggregate_performance


def eval_config(train_df, val_len, cfg, n_paths=400, S0=100):
    """Fit model on train_df and evaluate on validation horizon."""
    train_df = classify_regimes(train_df, k=cfg["k"])
    P = estimate_transition_matrix(train_df)
    params = estimate_regime_params(train_df)
    vols = {r: v["sigma"] for r, v in params.items()}

    pnls, turns = [], []
    for _ in range(n_paths):
        prices, regimes = simulate_regime_gbm(
            P, "Neutral", params, n_steps=val_len, S0=S0
        )
        pnl, pos, turnover = apply_trading_policy(
            prices,
            regimes,
            slippage=cfg["slippage"],
            commission=cfg["commission"],
            max_drawdown=cfg["max_dd"],
            persistence=cfg["persistence"],
            long_only=cfg["long_only"],
            vol_weight=cfg["vol_weight"],
            regime_vols=vols,
        )
        flips = np.sum(np.diff(pos) != 0)
        turns.append(flips / len(pos))
        pnls.append(pnl)

    df_metrics, _ = aggregate_performance(pnls)
    sharpe = df_metrics["sharpe"].mean()
    max_dd = df_metrics["max_drawdown"].mean()
    turn = np.mean(turns)
    score = sharpe - cfg["lambda_dd"] * max_dd - cfg["lambda_turn"] * turn
    return {
        "cfg": cfg,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "turn": turn,
        "score": score,
    }


def _single_task(args):
    df, fold_end, val_len, cfg, n_paths = args
    train_df = df.loc[:fold_end].copy()
    res = eval_config(train_df, val_len, cfg, n_paths)
    return res["score"]


def walk_forward_grid(df, folds, grid, val_len, n_paths=400):
    """Parallelised walk-forward grid search."""
    tasks = list(itertools.product(folds, grid))
    n_workers = min(cpu_count(), 8)
    print(f"Running grid search on {n_workers} cores ...")

    pool_args = [(df, fold_end, val_len, cfg, n_paths) for (fold_end, cfg) in tasks]
    scores = []

    with Pool(processes=n_workers) as pool:
        results = pool.map(_single_task, pool_args)

    # Re-aggregate results per parameter set
    for i, cfg in enumerate(grid):
        fold_scores = results[i * len(folds) : (i + 1) * len(folds)]
        cfg["score_mean"] = np.mean(fold_scores)
        cfg["score_std"] = np.std(fold_scores)
        scores.append(cfg)

    return pd.DataFrame(scores).sort_values("score_mean", ascending=False)
