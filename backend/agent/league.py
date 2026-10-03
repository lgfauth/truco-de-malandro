"""League training env: every match is against an opponent drawn from a pool.

Pool kinds (weights in :data:`agent.config.DEFAULT_LEAGUE`):
    random     uniform legal actions
    rule       :class:`arena.players.RulePlayer`
    snapshot   an earlier checkpoint of the agent (``<run>/checkpoints/*.zip``,
               after any snapshots inherited from the run it continues)
    latest     the agent's most recent weights (``<run>/latest.zip``)

Checkpoints are exchanged through files so the same env works in-process
(``DummyVecEnv``) and in worker processes (``SubprocVecEnv``). The learner
sits at a random seat each match, so it learns both leading and answering
the first hand.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from arena.players import ArenaPlayer, PPOPlayer, RandomPlayer, RulePlayer
from truco.encoding import (
    ACT_ACCEPT,
    ACT_CALL_TRUCO,
    ACT_RUN,
    NUM_ACTIONS,
    action_mask,
    decode_action,
    legal_action_ids,
    to_legal_action,
)
from truco.game import Player, TrucoGame, other
from truco.obs import obs_dim, observe
from truco.seeds import TRAIN_SEED_LIMIT

LATEST_NAME = "latest.zip"
CHECKPOINTS_DIR = "checkpoints"


def list_snapshots(run_dir: Path) -> List[Path]:
    return sorted((run_dir / CHECKPOINTS_DIR).glob("step_*.zip"))


class OpponentPool:
    """Samples an opponent kind by weight and builds/caches the player."""

    def __init__(self, weights: Dict[str, float], run_dir: Optional[Path],
                 max_snapshots: int = 20,
                 extra_snapshots: Sequence[str | Path] = ()) -> None:
        unknown = set(weights) - {"random", "rule", "snapshot", "latest"}
        if unknown:
            raise ValueError(f"Unknown league entries: {sorted(unknown)}")
        self.weights = {k: float(v) for k, v in weights.items() if v > 0}
        self.run_dir = run_dir
        self.max_snapshots = max_snapshots
        self.extra_snapshots = [Path(p) for p in extra_snapshots if Path(p).is_file()]
        self._random = RandomPlayer()
        self._rule = RulePlayer()
        # path -> (mtime, player)
        self._models: Dict[str, Tuple[float, PPOPlayer]] = {}

    def _load(self, path: Path) -> Optional[PPOPlayer]:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return None
        key = str(path)
        cached = self._models.get(key)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        try:
            from sb3_contrib import MaskablePPO

            model = MaskablePPO.load(key, device="cpu")
        except Exception:
            # File mid-write or unreadable: keep the previous weights if any.
            return cached[1] if cached else None
        player = PPOPlayer(path, deterministic=False, model=model, reseed=False)
        self._models[key] = (mtime, player)
        if len(self._models) > self.max_snapshots + 2:
            oldest = next(iter(self._models))
            if oldest != key:
                self._models.pop(oldest)
        return player

    def _snapshots(self) -> List[Path]:
        own = list_snapshots(self.run_dir) if self.run_dir is not None else []
        return (self.extra_snapshots + own)[-self.max_snapshots:]

    def _available(self) -> Dict[str, float]:
        avail = {}
        for kind, w in self.weights.items():
            if kind in ("random", "rule"):
                avail[kind] = w
            elif (kind == "latest" and self.run_dir is not None
                  and (self.run_dir / LATEST_NAME).exists()):
                avail[kind] = w
            elif kind == "snapshot" and self._snapshots():
                avail[kind] = w
        return avail or {"random": 1.0}

    def sample(self, rng: random.Random) -> Tuple[str, ArenaPlayer]:
        avail = self._available()
        kinds = list(avail)
        kind = rng.choices(kinds, weights=[avail[k] for k in kinds])[0]
        player: Optional[ArenaPlayer] = None
        if kind == "random":
            player = self._random
        elif kind == "rule":
            player = self._rule
        elif kind == "latest":
            player = self._load(self.run_dir / LATEST_NAME)
        elif kind == "snapshot":
            player = self._load(rng.choice(self._snapshots()))
        if player is None:
            kind, player = "random", self._random
        return kind, player


class LeagueEnv(gym.Env):
    """Learner vs a sampled opponent; one episode = one match, reward +/-1."""

    metadata = {"render_modes": []}

    def __init__(self, seed: Optional[int] = None, obs_version: str = "v2",
                 league: Optional[Dict[str, float]] = None,
                 run_dir: Optional[str | Path] = None,
                 max_snapshots: int = 20,
                 extra_snapshots: Sequence[str | Path] = ()) -> None:
        super().__init__()
        self.obs_version = obs_version
        self.pool = OpponentPool(league or {"random": 1.0},
                                 Path(run_dir) if run_dir else None, max_snapshots,
                                 extra_snapshots)
        self._init_seed = seed
        self._seeded = False
        self._rng = random.Random()
        self.game: Optional[TrucoGame] = None
        self.seat = Player.P0
        self.opponent_kind = "random"
        self.opponent: ArenaPlayer = RandomPlayer()
        self._stats: Dict[str, int] = {}
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(obs_dim(obs_version),),
                                            dtype=np.float32)

    # --- Gym API -----------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is None and not self._seeded:
            seed = self._init_seed
        super().reset(seed=seed)
        self._seeded = True
        self._rng = random.Random(int(self.np_random.integers(0, 2**63)))
        game_seed = (options or {}).get("game_seed")
        if game_seed is None:
            game_seed = int(self.np_random.integers(0, TRAIN_SEED_LIMIT))
        self.game = TrucoGame(seed=int(game_seed))
        self.seat = Player(self._rng.randrange(2))
        self.opponent_kind, self.opponent = self.pool.sample(self._rng)
        self.opponent.reset(f"{game_seed}:{int(self.seat)}:opp")
        self._stats = {"hands": 1, "truco_calls": 0, "runs": 0, "bet_responses": 0}
        self._hand = self.game.state.hand
        self._play_opponent()
        return self._observe(), self._info()

    def step(self, action: int):
        g = self.game
        legal = legal_action_ids(g, self.seat)
        a = int(action)
        if a not in legal:
            # MaskablePPO never gets here; kept for unmasked use.
            return self._observe(), -1.0, False, False, {**self._info(), "illegal_action": True}
        self._stats["bet_responses"] += int(ACT_ACCEPT in legal)
        self._stats["truco_calls"] += int(a == ACT_CALL_TRUCO)
        self._stats["runs"] += int(a == ACT_RUN)
        g.step(decode_action(a))
        self._track_hand()
        self._play_opponent()

        terminated = g.state.winner is not None
        reward = 0.0
        info = self._info()
        if terminated:
            won = g.state.winner == self.seat
            reward = 1.0 if won else -1.0
            info["match"] = {"opponent": self.opponent_kind, "won": won,
                             "seat": int(self.seat), **self._stats}
        return self._observe(), reward, terminated, False, info

    def action_masks(self) -> np.ndarray:
        return action_mask(self.game, self.seat)

    # --- Internals ---------------------------------------------------------
    def _track_hand(self) -> None:
        if self.game.state.hand is not self._hand and self.game.state.winner is None:
            self._stats["hands"] += 1
            self._hand = self.game.state.hand

    def _play_opponent(self) -> None:
        g = self.game
        opp = other(self.seat)
        while (g.state.winner is None and g.state.hand is not None
               and g.state.hand.current_player == opp):
            try:
                a = int(self.opponent.decide(g, opp))
            except Exception:
                a = -1
            pa, _ = to_legal_action(g, a)
            g.step(pa)
            self._track_hand()

    def _observe(self) -> np.ndarray:
        return observe(self.game, self.seat, self.obs_version)

    def _info(self) -> Dict[str, Any]:
        return {"legal_actions": legal_action_ids(self.game, self.seat),
                "opponent": self.opponent_kind}


def save_atomic(model, path: Path) -> None:
    """Save ``model`` to ``path`` so readers never see a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"_tmp_{path.name}")
    model.save(str(tmp))
    for _ in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # Windows: a reader has the file open
            import time

            time.sleep(0.05)
    os.replace(tmp, path)
