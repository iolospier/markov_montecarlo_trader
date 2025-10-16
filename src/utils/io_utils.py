import os, datetime, shutil


def create_run_dirs(base_dir="results"):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{ts}")
    subdirs = ["plots", "logs", "data"]
    for s in subdirs:
        os.makedirs(os.path.join(run_dir, s), exist_ok=True)
    return run_dir
