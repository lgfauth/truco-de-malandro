"""Single source of truth for training hyperparameters and league settings.

``HYPERPARAMS`` holds the MaskablePPO arguments. A rollout collects
``n_envs * n_steps`` = 2048 transitions (about 110 matches) per update; with
sparse +/-1 match rewards the larger batch is noticeably more stable than the
512-step rollouts the original ``truco_ppo_1M`` checkpoint was trained with
(``n_steps=512, batch_size=32``, still recorded inside that zip).

Every run writes the resolved config to ``runs/<id>/config.json``.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

HYPERPARAMS: Dict[str, Any] = {
    "learning_rate": 3e-4,
    "n_envs": 8,
    "n_steps": 256,            # per env -> 2048 transitions per rollout
    "batch_size": 128,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    # 0.05 kept entropy near 0.7 for a whole 3M run and the policy plateaued
    # early; 0.01 trained the strongest model so far (see README results).
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "normalize_advantage": True,
    "net_arch": [256, 256],
}

# Opponent mix sampled at the start of every training match.
DEFAULT_LEAGUE: Dict[str, float] = {
    "random": 0.10,
    "rule": 0.30,
    "snapshot": 0.30,   # a uniformly chosen earlier snapshot of the agent
    "latest": 0.30,     # the agent's most recent weights
}

# Fixed evaluation opponents. "prev" is the snapshot saved just before the
# checkpoint under evaluation.
DEFAULT_EVAL_OPPONENTS: List[str] = ["random", "rule", "prev", "ppo:truco_ppo_1M"]


@dataclass
class TrainConfig:
    name: str = "league"
    total_timesteps: int = 3_000_000
    seed: int = 0
    obs_version: str = "v2"
    hyperparams: Dict[str, Any] = field(default_factory=lambda: copy.deepcopy(HYPERPARAMS))
    league: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_LEAGUE))
    snapshot_every: int = 100_000     # timesteps between saved snapshots
    latest_refresh: int = 20_000      # timesteps between "latest" opponent updates
    max_snapshots_in_pool: int = 20   # most recent snapshots eligible as opponents
    eval_every: int = 100_000         # timesteps between arena evaluations
    eval_games: int = 500             # per opponent; each deal played from both seats
    eval_seed: int = 123
    eval_opponents: List[str] = field(default_factory=lambda: list(DEFAULT_EVAL_OPPONENTS))
    eval_workers: int = 4
    vec_env: str = "subproc"          # "subproc" (CLI) or "dummy" (in-process)
    init_from: str = ""               # optional checkpoint to continue from

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        cfg = cls(**known)
        merged = copy.deepcopy(HYPERPARAMS)
        merged.update(cfg.hyperparams or {})
        cfg.hyperparams = merged
        return cfg


def ppo_kwargs(hp: Dict[str, Any]) -> Dict[str, Any]:
    """MaskablePPO constructor kwargs from a hyperparameter dict."""
    kw = {k: v for k, v in hp.items() if k not in ("n_envs", "net_arch")}
    kw["policy_kwargs"] = {"net_arch": list(hp.get("net_arch", [64, 64]))}
    return kw
