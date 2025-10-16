import logging, os, datetime


def get_logger(run_dir):
    log_file = os.path.join(run_dir, "logs", "run.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger("markov_montecarlo")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        ch = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger
