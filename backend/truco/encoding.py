"""Card and action encodings shared by the env, the agents, the API and the arena.

Action ids (0..6) are the public interface every player speaks:

    0, 1, 2  play the card at that index of the player's hand
    3        call truco
    4        accept (truco or Mão de 11)
    5        run (fold)
    6        raise
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .game import ActionType, Card, Player, PlayerAction, Rank, TrucoGame


# --- Card <-> index encoding (40 cards, 10 ranks * 4 suits) -----------------
RANKS: List[Rank] = [Rank.FOUR, Rank.FIVE, Rank.SIX, Rank.SEVEN,
                     Rank.QUEEN, Rank.JACK, Rank.KING, Rank.ACE,
                     Rank.TWO, Rank.THREE]
RANK_TO_IDX = {r: i for i, r in enumerate(RANKS)}
NUM_CARDS = 40


def card_to_id(card: Card) -> int:
    return RANK_TO_IDX[card.rank] * 4 + int(card.suit)


# --- Action encoding --------------------------------------------------------
ACT_PLAY_0 = 0
ACT_PLAY_1 = 1
ACT_PLAY_2 = 2
ACT_CALL_TRUCO = 3
ACT_ACCEPT = 4
ACT_RUN = 5
ACT_RAISE = 6
NUM_ACTIONS = 7

ACTION_NAMES = {
    ACT_PLAY_0: "play_0",
    ACT_PLAY_1: "play_1",
    ACT_PLAY_2: "play_2",
    ACT_CALL_TRUCO: "truco",
    ACT_ACCEPT: "accept",
    ACT_RUN: "run",
    ACT_RAISE: "raise",
}


def decode_action(a: int) -> PlayerAction:
    if a in (ACT_PLAY_0, ACT_PLAY_1, ACT_PLAY_2):
        return PlayerAction(ActionType.PLAY_CARD, card_index=a)
    if a == ACT_CALL_TRUCO:
        return PlayerAction(ActionType.CALL_TRUCO)
    if a == ACT_ACCEPT:
        return PlayerAction(ActionType.ACCEPT)
    if a == ACT_RUN:
        return PlayerAction(ActionType.RUN)
    if a == ACT_RAISE:
        return PlayerAction(ActionType.RAISE)
    raise ValueError(f"Invalid action id: {a}")


def encode_action(pa: PlayerAction) -> int:
    if pa.type == ActionType.PLAY_CARD:
        return int(pa.card_index)
    return {
        ActionType.CALL_TRUCO: ACT_CALL_TRUCO,
        ActionType.ACCEPT: ACT_ACCEPT,
        ActionType.RUN: ACT_RUN,
        ActionType.RAISE: ACT_RAISE,
    }[pa.type]


def actions_equal(a: PlayerAction, b: PlayerAction) -> bool:
    return a.type == b.type and a.card_index == b.card_index


# --- Legality helpers -------------------------------------------------------
def legal_action_ids(game: TrucoGame, player: Player) -> List[int]:
    """Sorted legal action ids for ``player`` (empty if it is not their turn)."""
    s = game.state
    if s.winner is not None or s.hand is None or s.hand.current_player != player:
        return []
    return sorted({encode_action(a) for a in game.legal_actions()})


def action_mask(game: TrucoGame, player: Player) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
    for a in legal_action_ids(game, player):
        mask[a] = 1
    return mask


def to_legal_action(game: TrucoGame, action_id: int) -> Tuple[PlayerAction, bool]:
    """Map ``action_id`` to a legal :class:`PlayerAction` for the player to act.

    Returns ``(action, fallback)``. When ``action_id`` is not legal the first
    legal action is returned and ``fallback`` is True.
    """
    legal = game.legal_actions()
    try:
        pa = decode_action(int(action_id))
    except ValueError:
        return legal[0], True
    if any(actions_equal(pa, la) for la in legal):
        return pa, False
    return legal[0], True
