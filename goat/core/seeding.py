from __future__ import annotations

import os
import random
from typing import Any


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    try:
        import numpy as np

        np.random.seed(int(seed))
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass


def get_device(prefer_cuda: bool = True) -> Any:
    import torch

    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")

