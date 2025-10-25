import random
import numpy as np
import torch
from .constants import SEED


def reset(seed: int=SEED):
    # init
    print(f"SETTING ALL SEEDS TO {seed}...")
    # python default
    random.seed(seed)
    # numpy
    np.random.seed(seed)
    # torch
    torch.manual_seed(seed)
    # gpu
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # final
    print("ALL SEEDS SET")