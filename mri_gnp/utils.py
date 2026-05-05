import json
import random
from pathlib import Path

import numpy as np
import torch


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(payload, path):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_plain_types(value):
    if isinstance(value, dict):
        return {str(key): to_plain_types(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_types(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
