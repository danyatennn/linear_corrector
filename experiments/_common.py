import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from config import DATA


def load_pools():
    path = os.path.join(DATA.cache_dir, "scores.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run `.venv/bin/python scripts/train_sasrec.py` first.")
    z = np.load(path)
    return {k: z[k] for k in z.files}


def chunk_pseudo_users(scores, relevant, k, rng):
    idx = rng.permutation(len(scores))
    scores, relevant = np.asarray(scores)[idx], np.asarray(relevant)[idx]
    n_full = len(scores) // k
    return [(scores[i * k:(i + 1) * k], relevant[i * k:(i + 1) * k])
            for i in range(n_full)]
