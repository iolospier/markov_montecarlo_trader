import numpy as np
import pandas as pd
import itertools
from multiprocessing import Pool, cpu_count
from functools import partial
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
from src.utils.timer import timed


def eval_config(train_df, val_len, cfg, n_paths=400, S0=100, seed=42, logger=None):
    """Fit model on training data and evaluate on a validation horizon."""
    logger = logger or get_logger(".")
    rng = np.random.default_rng(seed)
    set_global_seed(seed)

    train_df = classify_regimes(train_df, k=cfg["k"])
    P = estimate_transition_matrix(train_df)
    params = estimate_regime_params(train_df)
    vols = {r: v["sigma"] for r, v in params.items()}

    pnls, turns = [], []

    for i in range(n_paths):
        local_seed = seed + i
        try:
            prices, regimes = simulate_regime_gbm(
                P, "Neutral", params, n_steps=val_len, S0=S0, seed=local_seed
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
        except Exception as e:
            logger.warning(f"Path {i} failed in eval_config: {e}")

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


def _single_task(df, fold_end, val_len, cfg, n_paths, seed, logger=None):
    """Helper to run a single grid task."""
    train_df = df.loc[:fold_end].copy()
    return eval_config(train_df, val_len, cfg, n_paths, seed=seed, logger=logger)[
        "score"
    ]


@timed
def walk_forward_grid(df, folds, grid, val_len, n_paths=400, base_seed=42, logger=None):
    """Parallelised walk-forward grid search with logging and reproducibility."""
    logger = logger or get_logger(".")
    n_workers = min(cpu_count(), 8)
    logger.info(f"Running walk-forward grid search on {n_workers} cores...")

    tasks = list(itertools.product(folds, grid))
    pool_args = [
        (df, fold_end, val_len, cfg, n_paths, base_seed + i)
        for i, (fold_end, cfg) in enumerate(tasks)
    ]

    scores = []
    with Pool(processes=n_workers) as pool:
        results = pool.starmap(
            _single_task,
            [
                (df, fold_end, val_len, cfg, n_paths, base_seed + i, logger)
                for i, (fold_end, cfg) in enumerate(tasks)
            ],
        )

    for i, cfg in enumerate(grid):
        fold_scores = results[i * len(folds) : (i + 1) * len(folds)]
        cfg["score_mean"] = np.mean(fold_scores)
        cfg["score_std"] = np.std(fold_scores)
        scores.append(cfg)

    df_out = pd.DataFrame(scores).sort_values("score_mean", ascending=False)
    logger.info("Grid search complete.")
    return df_out
