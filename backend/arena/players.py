"""Arena players and the spec registry that builds them.

A player sees the full :class:`TrucoGame` plus its seat and must return an
action id in 0..6 (see :mod:`truco.encoding`). It is trusted not to peek at
the opponent's hand: the built-in players only use ``observe()`` or their own
cards. Illegal answers are not fatal — the arena counts them and falls back
to the first legal action.

Specs:
    random                  uniform over legal actions
    rule                    :class:`RulePlayer`
    ppo:<path or name>      MaskablePPO checkpoint (``name`` → models/<name>.zip)
    ppo-stoch:<path|name>   same, sampling instead of argmax
    py:<module>:<Class>     any importable class; optional ``:arg`` is passed
                            to the constructor (plug-in point for new players)
"""

from __future__ import annotations

import importlib
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from truco.encoding import (
    ACT_ACCEPT,
    ACT_CALL_TRUCO,
    ACT_RAISE,
    ACT_RUN,
    action_mask,
    legal_action_ids,
)
from truco.game import Player as Seat
from truco.game import RoundResult, TrucoGame, card_strength
from truco.obs import observe, version_for_dim

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class ArenaPlayer:
    """Base class. Subclasses implement :meth:`decide`."""

    name: str = "player"

    def reset(self, match_key: str) -> None:
        """Called before each match with a key unique to (deal, seat, role).

        Stochastic players should reseed from it so results do not depend on
        how matches are split across worker processes.
        """

    def decide(self, game: TrucoGame, seat: Seat) -> int:
        raise NotImplementedError


class RandomPlayer(ArenaPlayer):
    name = "random"

    def __init__(self) -> None:
        self._rng = random.Random(0)

    def reset(self, match_key: str) -> None:
        self._rng = random.Random(match_key)

    def decide(self, game: TrucoGame, seat: Seat) -> int:
        return self._rng.choice(legal_action_ids(game, seat))


# --- Rule-based baseline ----------------------------------------------------
def _card_value(strength: int) -> float:
    """Map card_strength (0..9 plain, 100..103 manilha) to [0, 1]."""
    if strength >= 100:
        return 0.8 + 0.2 * (strength - 100) / 3
    return strength / 12.0


def hand_quality(game: TrucoGame, seat: Seat) -> float:
    """Rough 0..1 estimate of how good ``seat``'s hand is right now.

    Averages the two best remaining cards and shifts by the rounds already
    won or lost in this hand. Uses only the seat's own cards and the table.
    """
    h = game.state.hand
    cards = h.hands[seat]
    values = sorted((_card_value(card_strength(c, h.vira)) for c in cards),
                    reverse=True)
    base = float(np.mean(values[:2])) if values else 0.0
    mine = RoundResult.P0_WIN if seat == Seat.P0 else RoundResult.P1_WIN
    for r in h.rounds:
        if r.result is None or r.result == RoundResult.TIE:
            continue
        base += 0.25 if r.result == mine else -0.25
    return max(0.0, min(1.0, base))


class RulePlayer(ArenaPlayer):
    """Deterministic heuristic baseline.

    - Good hand: lead with the strongest card and ask for truco when very
      strong; answer bets by accepting (raising when very strong).
    - Weak hand: lead with the weakest card and run from bets.
    - Answering a card: win it with the cheapest card that does, otherwise
      throw the weakest.
    """

    name = "rule"

    def __init__(self, good: float = 0.5, strong: float = 0.75,
                 very_strong: float = 0.9) -> None:
        self.good = good
        self.strong = strong
        self.very_strong = very_strong

    def decide(self, game: TrucoGame, seat: Seat) -> int:
        legal = legal_action_ids(game, seat)
        h = game.state.hand
        q = hand_quality(game, seat)

        if ACT_ACCEPT in legal:  # a bet or the Mão de 11 decision is pending
            if ACT_RAISE in legal and q >= self.very_strong:
                return ACT_RAISE
            return ACT_ACCEPT if q >= self.good else ACT_RUN

        if ACT_CALL_TRUCO in legal and q >= self.strong:
            return ACT_CALL_TRUCO

        cards = h.hands[seat]
        strengths = [card_strength(c, h.vira) for c in cards]
        order = sorted(range(len(cards)), key=lambda i: strengths[i])
        table = h.rounds[-1].plays
        if table:
            to_beat = card_strength(table[0][1], h.vira)
            winners = [i for i in order if strengths[i] > to_beat]
            return winners[0] if winners else order[0]
        return order[-1] if q >= self.good else order[0]


# --- PPO checkpoints --------------------------------------------------------
_MODEL_CACHE: Dict[str, object] = {}


def resolve_checkpoint(ref: str) -> Path:
    """Accept a path (with or without .zip) or a bare name under models/."""
    candidates = [Path(ref), Path(ref + ".zip"),
                  MODELS_DIR / ref, MODELS_DIR / (ref + ".zip")]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(f"Checkpoint not found: {ref}")


def load_ppo(path: Path):
    from sb3_contrib import MaskablePPO

    key = str(path)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = MaskablePPO.load(key, device="cpu")
    return _MODEL_CACHE[key]


class PPOPlayer(ArenaPlayer):
    """A MaskablePPO checkpoint. The observation version follows the model."""

    def __init__(self, checkpoint: str | Path, deterministic: bool = True,
                 model=None, reseed: bool = True) -> None:
        """``model`` skips loading (the path is then only a label).

        ``reseed`` makes stochastic play reproducible per match by reseeding
        torch's global RNG; turn it off when sharing a process with a model
        that is being trained.
        """
        self.path = resolve_checkpoint(str(checkpoint)) if model is None else Path(str(checkpoint))
        self.model = model if model is not None else load_ppo(self.path)
        self.obs_version = version_for_dim(int(self.model.observation_space.shape[0]))
        self.deterministic = deterministic
        self.reseed = reseed
        self.name = f"ppo:{self.path.stem}" + ("" if deterministic else ":stoch")

    def reset(self, match_key: str) -> None:
        if not self.deterministic and self.reseed:
            # SB3 samples through torch's global RNG.
            import torch

            torch.manual_seed(random.Random(match_key).getrandbits(63))

    def decide(self, game: TrucoGame, seat: Seat) -> int:
        action, _ = self.model.predict(
            observe(game, seat, self.obs_version),
            action_masks=action_mask(game, seat),
            deterministic=self.deterministic,
        )
        return int(np.asarray(action).item())


# --- Spec registry ----------------------------------------------------------
PlayerFactory = Callable[[Optional[str]], ArenaPlayer]

PLAYER_FACTORIES: Dict[str, PlayerFactory] = {
    "random": lambda arg: RandomPlayer(),
    "rule": lambda arg: RulePlayer(),
    "ppo": lambda arg: PPOPlayer(_require(arg, "ppo")),
    "ppo-stoch": lambda arg: PPOPlayer(_require(arg, "ppo-stoch"), deterministic=False),
    "py": lambda arg: _import_player(_require(arg, "py")),
}


def register_player(prefix: str, factory: PlayerFactory) -> None:
    """Make ``<prefix>[:arg]`` specs build players with ``factory(arg)``."""
    PLAYER_FACTORIES[prefix] = factory


def _require(arg: Optional[str], prefix: str) -> str:
    if not arg:
        raise ValueError(f"Spec '{prefix}' needs an argument: '{prefix}:<value>'")
    return arg


def _import_player(arg: str) -> ArenaPlayer:
    parts = arg.split(":", 2)
    if len(parts) < 2:
        raise ValueError("Use py:<module>:<Class>[:<arg>]")
    cls = getattr(importlib.import_module(parts[0]), parts[1])
    player = cls(parts[2]) if len(parts) == 3 else cls()
    if not hasattr(player, "decide"):
        raise TypeError(f"{parts[0]}.{parts[1]} has no decide(game, seat)")
    return player


def is_spec(text: str) -> bool:
    """True if ``text`` names a registered player (not a bare file path)."""
    return text.partition(":")[0] in PLAYER_FACTORIES


def as_spec(text: str) -> str:
    """Turn a bare checkpoint path/name into a ``ppo:`` spec."""
    return text if is_spec(text) else f"ppo:{text}"


def make_player(spec: str) -> ArenaPlayer:
    prefix, _, arg = spec.partition(":")
    try:
        factory = PLAYER_FACTORIES[prefix]
    except KeyError:
        raise ValueError(
            f"Unknown player spec {spec!r}; known: {sorted(PLAYER_FACTORIES)}"
        ) from None
    player = factory(arg or None)
    if not getattr(player, "name", None) or player.name == ArenaPlayer.name:
        player.name = spec
    return player


def list_checkpoints(models_dir: Path = MODELS_DIR) -> List[Path]:
    return sorted(models_dir.glob("*.zip"))
