"""Player-relative observations of a :class:`TrucoGame`.

``observe(game, player)`` is the single source of truth for what an agent
sees. It never exposes the opponent's hidden cards.

Version ``v1`` (24 floats in [0, 1]) is the layout the ``truco_ppo_*``
checkpoints were trained on. Card slots store ``(card_id + 1) / NUM_CARDS``
so 0 means "empty".

    [0..2]   own hand (3 slots)
    [3]      vira
    [4..9]   rounds: 3 rounds x 2 plays (in play order)
    [10]     own score / TARGET
    [11]     opponent score / TARGET
    [12]     current stake / 12
    [13]     pending stake / 12 (0 if none)
    [14]     has pending stake (0/1)
    [15]     iron hand (0/1)
    [16]     awaiting Mão de 11 response (0/1)
    [17]     Mão de 11 for me (0/1)
    [18]     Mão de 11 for the opponent (0/1)
    [19]     it is my turn (0/1)
    [20]     I am the dealer (0/1)
    [21..23] reserved, always 0 (the old Mão de 11 rule revealed the
             opponent's cards here; the Paulista rule reveals nothing)
"""

from __future__ import annotations

import numpy as np

from .encoding import NUM_CARDS, card_to_id
from .game import Player, TrucoGame, other

OBS_V1_DIM = 24
OBS_DIMS = {"v1": OBS_V1_DIM}
DEFAULT_OBS_VERSION = "v1"


def obs_dim(version: str = DEFAULT_OBS_VERSION) -> int:
    try:
        return OBS_DIMS[version]
    except KeyError:
        raise ValueError(f"Unknown observation version: {version!r}") from None


def version_for_dim(dim: int) -> str:
    """Observation version whose vector has ``dim`` entries."""
    for version, d in OBS_DIMS.items():
        if d == dim:
            return version
    raise ValueError(f"No observation version with {dim} dims")


def observe(game: TrucoGame, player: Player,
            version: str = DEFAULT_OBS_VERSION) -> np.ndarray:
    if version == "v1":
        return _observe_v1(game, player)
    raise ValueError(f"Unknown observation version: {version!r}")


def _card(c) -> float:
    return (card_to_id(c) + 1) / NUM_CARDS


def _observe_v1(game: TrucoGame, me: Player) -> np.ndarray:
    opp = other(me)
    obs = np.zeros(OBS_V1_DIM, dtype=np.float32)
    s = game.state
    h = s.hand
    target = game.TARGET_SCORE

    if h is not None:
        for i, c in enumerate(h.hands[me][:3]):
            obs[i] = _card(c)
        obs[3] = _card(h.vira)

        slot = 4
        for r in range(3):
            if r < len(h.rounds):
                plays = h.rounds[r].plays
                for k in range(2):
                    if k < len(plays):
                        obs[slot] = _card(plays[k][1])
                    slot += 1
            else:
                slot += 2

        obs[12] = h.stake / 12.0
        obs[13] = (h.pending_stake or 0) / 12.0
        obs[14] = 1.0 if h.pending_stake is not None else 0.0
        obs[16] = 1.0 if h.awaiting_mao11_response else 0.0
        obs[19] = 1.0 if h.current_player == me else 0.0

    obs[10] = s.scores[int(me)] / target
    obs[11] = s.scores[int(opp)] / target
    obs[15] = 1.0 if s.iron_hand else 0.0
    obs[17] = 1.0 if s.mao11_player == me else 0.0
    obs[18] = 1.0 if s.mao11_player == opp else 0.0
    obs[20] = 1.0 if s.dealer == me else 0.0
    return obs
