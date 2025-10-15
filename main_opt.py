from src.data_loader import init_data
from src.regimes import (
    classify_regimes,
    estimate_transition_matrix,
    estimate_regime_params,
)
from src.optimiser import walk_forward_grid
import yaml

if __name__ == "__main__":
    # Load config
    with open("config/opt_params.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    df = init_data("QQQ", "2018-01-01", "../data/qqq.csv")

    folds = cfg["opt"]["folds"]
    grid = [
        {
            "k": k,
            "persistence": p,
            "long_only": lo,
            "vol_weight": vw,
            "slippage": sl,
            "commission": cm,
            "max_dd": md,
            "lambda_dd": cfg["opt"]["lambda_dd"],
            "lambda_turn": cfg["opt"]["lambda_turn"],
        }
        for k in cfg["opt"]["k_vals"]
        for p in cfg["opt"]["persistence"]
        for lo in cfg["opt"]["long_only"]
        for vw in cfg["opt"]["vol_weight"]
        for sl in cfg["opt"]["slippage"]
        for cm in cfg["opt"]["commission"]
        for md in cfg["opt"]["max_dd"]
    ]

    val_len = cfg["opt"]["val_len"]

    print(f"Data initialised and saved to ../data/qqq.csv")
    print(f"Running grid search on {len(grid)} parameter combinations ...")

    results = walk_forward_grid(
        df, folds, grid, val_len, n_paths=cfg["simulation"]["n_paths"]
    )

    print("Optimisation complete.")
    print(results.head())
    results.to_csv("optimisation_results.csv", index=False)
