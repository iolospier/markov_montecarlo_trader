import yaml, os


def load_config(path: str):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    config["project_root"] = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
    return config
