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

from .encoding import NUM_CARDS, RANK_TO_IDX, card_to_id
from .game import Player, RoundResult, TrucoGame, card_strength, manilha_rank, other

OBS_V1_DIM = 24

# --- v2 layout ----------------------------------------------------------------
# Every card the agent may see gets identity (rank/suit one-hots) and, more
# importantly, its strength relative to the vira, so manilhas are explicit.
#
#   own hand, 3 slots x 17 (in hand order, so slot i == action i):
#       present, rank one-hot (10), suit one-hot (4), strength, is_manilha
#   vira: rank one-hot (10), suit one-hot (4)
#   manilha rank one-hot (10)
#   table, 3 rounds x 2 plays x 4: present, played_by_me, strength, is_manilha
#   round results, 3 x 3: won by me, won by opponent, tie
#   current round (one-hot 3), leading this round, card to beat (strength)
#   seen cards: 40 bits (vira + every card played so far in this hand)
#   scores: mine / 12, opponent / 12
#   stake one-hot over 1,3,6,9,12; pending stake one-hot over 3,6,9,12
#   flags: iron hand, awaiting Mão de 11, Mão de 11 mine, Mão de 11 opponent,
#          my turn, I am the dealer, I may raise (not locked), I made the
#          pending call
#   remaining cards: mine / 3, opponent / 3
_CARD_SLOT = 17
_STAKES = [1, 3, 6, 9, 12]
_PENDING = [3, 6, 9, 12]
OBS_V2_DIM = (3 * _CARD_SLOT + 14 + 10 + 3 * 2 * 4 + 9 + 3 + 1 + 1
              + NUM_CARDS + 2 + len(_STAKES) + len(_PENDING) + 8 + 2)

OBS_DIMS = {"v1": OBS_V1_DIM, "v2": OBS_V2_DIM}
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
    if version == "v2":
        return _observe_v2(game, player)
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


def _strength01(card, vira) -> float:
    """Strength on a 0..1 scale: plain cards 0..9/13, manilhas 10..13/13."""
    s = card_strength(card, vira)
    return (10 + (s - 100)) / 13.0 if s >= 100 else s / 13.0


def _observe_v2(game: TrucoGame, me: Player) -> np.ndarray:
    opp = other(me)
    obs = np.zeros(OBS_V2_DIM, dtype=np.float32)
    s = game.state
    h = s.hand
    target = game.TARGET_SCORE
    i = 0

    def put_card(card) -> None:
        nonlocal i
        obs[i] = 1.0
        obs[i + 1 + RANK_TO_IDX[card.rank]] = 1.0
        obs[i + 11 + int(card.suit)] = 1.0
        obs[i + 15] = _strength01(card, h.vira)
        obs[i + 16] = 1.0 if card.rank == manilha else 0.0
        i += _CARD_SLOT

    if h is not None:
        manilha = manilha_rank(h.vira)
        # Own hand, in hand order.
        cards = h.hands[me]
        for k in range(3):
            if k < len(cards):
                put_card(cards[k])
            else:
                i += _CARD_SLOT
        # Vira and manilha rank.
        obs[i + RANK_TO_IDX[h.vira.rank]] = 1.0
        obs[i + 10 + int(h.vira.suit)] = 1.0
        i += 14
        obs[i + RANK_TO_IDX[manilha]] = 1.0
        i += 10
        # Table.
        seen = [h.vira]
        for r in range(3):
            plays = h.rounds[r].plays if r < len(h.rounds) else []
            for k in range(2):
                if k < len(plays):
                    p, card = plays[k]
                    seen.append(card)
                    obs[i] = 1.0
                    obs[i + 1] = 1.0 if p == me else 0.0
                    obs[i + 2] = _strength01(card, h.vira)
                    obs[i + 3] = 1.0 if card.rank == manilha else 0.0
                i += 4
        # Round results.
        mine = RoundResult.P0_WIN if me == Player.P0 else RoundResult.P1_WIN
        for r in range(3):
            res = h.rounds[r].result if r < len(h.rounds) else None
            if res is not None:
                if res == RoundResult.TIE:
                    obs[i + 2] = 1.0
                else:
                    obs[i + (0 if res == mine else 1)] = 1.0
            i += 3
        # Current round.
        cur = h.rounds[-1]
        obs[i + min(len(h.rounds), 3) - 1] = 1.0
        i += 3
        obs[i] = 1.0 if not cur.plays else 0.0
        i += 1
        if cur.plays and cur.plays[0][0] == opp:
            obs[i] = _strength01(cur.plays[0][1], h.vira)
        i += 1
        # Seen cards.
        for card in seen:
            obs[i + card_to_id(card)] = 1.0
        i += NUM_CARDS
    else:
        i += 3 * _CARD_SLOT + 14 + 10 + 24 + 9 + 3 + 1 + 1 + NUM_CARDS

    obs[i] = s.scores[int(me)] / target
    obs[i + 1] = s.scores[int(opp)] / target
    i += 2

    if h is not None:
        if h.stake in _STAKES:
            obs[i + _STAKES.index(h.stake)] = 1.0
        i += len(_STAKES)
        if h.pending_stake in _PENDING:
            obs[i + _PENDING.index(h.pending_stake)] = 1.0
        i += len(_PENDING)
    else:
        i += len(_STAKES) + len(_PENDING)

    obs[i] = 1.0 if s.iron_hand else 0.0
    obs[i + 1] = 1.0 if h is not None and h.awaiting_mao11_response else 0.0
    obs[i + 2] = 1.0 if s.mao11_player == me else 0.0
    obs[i + 3] = 1.0 if s.mao11_player == opp else 0.0
    if h is not None:
        obs[i + 4] = 1.0 if h.current_player == me else 0.0
        obs[i + 6] = 0.0 if h.truco_locked_for == me else 1.0
        obs[i + 7] = 1.0 if h.truco_caller == me and h.pending_stake is not None else 0.0
    obs[i + 5] = 1.0 if s.dealer == me else 0.0
    i += 8

    if h is not None:
        obs[i] = len(h.hands[me]) / 3.0
        obs[i + 1] = len(h.hands[opp]) / 3.0
    i += 2
    assert i == OBS_V2_DIM, (i, OBS_V2_DIM)
    return obs
