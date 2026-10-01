"""Disjoint seed spaces for training and evaluation deals.

Training envs draw game seeds from ``[0, TRAIN_SEED_LIMIT)``. Evaluation and
arena matches use seeds at or above ``EVAL_SEED_BASE``, so a model is never
evaluated on a deal sequence it could have trained on.
"""

from __future__ import annotations

from typing import List

import numpy as np

TRAIN_SEED_LIMIT = 2**62
EVAL_SEED_BASE = 2**62
_EVAL_SEED_SPAN = 2**61


def eval_game_seeds(seed: int, n: int) -> List[int]:
    """``n`` distinct evaluation game seeds derived from ``seed``."""
    rng = np.random.default_rng(seed)
    out: List[int] = []
    seen = set()
    while len(out) < n:
        s = int(rng.integers(0, _EVAL_SEED_SPAN))
        if s not in seen:
            seen.add(s)
            out.append(EVAL_SEED_BASE + s)
    return out


def is_eval_seed(game_seed: int) -> bool:
    return game_seed >= EVAL_SEED_BASE
