# main_opt.py
from src.utils.config_loader import load_config
from src.utils.seed_utils import set_global_seed
from src.utils.io_utils import create_run_dirs
from src.utils.logger import get_logger
from src.utils.timer import timed
from src.optuna_optimiser import main as run_optuna


@timed
def main():
    config = load_config("config/optuna_config.yaml")
    run_dir = create_run_dirs("results")
    set_global_seed(config["optuna"].get("seed", 42))
    logger = get_logger(run_dir)

    logger.info("=== Starting Optuna Optimisation ===")
    run_optuna()
    logger.info("=== Optimisation Complete ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        print("Fatal error:", e)
        traceback.print_exc()
