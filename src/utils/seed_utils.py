import numpy as np, random


def set_global_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
