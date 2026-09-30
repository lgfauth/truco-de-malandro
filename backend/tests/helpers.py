"""Shared builders for rule tests: deal a hand with chosen cards."""

from __future__ import annotations

from typing import List, Optional

from truco.game import (
    ActionType,
    Card,
    Hand,
    Player,
    PlayerAction,
    Rank,
    Suit,
    TrucoGame,
)

# With a 4 as vira the manilha is 5; everything below is non-manilha unless
# a test asks for a FIVE.
VIRA = Card(Rank.FOUR, Suit.DIAMONDS)


def C(rank: Rank, suit: Suit = Suit.DIAMONDS) -> Card:
    return Card(rank, suit)


def deal(p0: List[Card], p1: List[Card], *, first: Player = Player.P0,
         vira: Card = VIRA, scores: Optional[List[int]] = None,
         dealer: Optional[Player] = None) -> TrucoGame:
    """Game whose current hand holds exactly the given cards.

    Scores and dealer are set before the hand is replaced, so special modes
    (Mão de 11, iron hand) are recomputed by ``_start_hand`` and then the
    cards are overridden.
    """
    g = TrucoGame(seed=0)
    s = g.state
    if scores is not None:
        s.scores = list(scores)
    if dealer is not None:
        s.dealer = dealer
    else:
        s.dealer = Player.P1 if first == Player.P0 else Player.P0
    g._start_hand()
    h = s.hand
    s.hand = Hand(
        vira=vira,
        hands={Player.P0: list(p0), Player.P1: list(p1)},
        first_to_play=first,
        current_player=h.current_player,
        stake=h.stake,
        awaiting_mao11_response=h.awaiting_mao11_response,
    )
    return g


def play(g: TrucoGame, idx: int = 0) -> None:
    g.step(PlayerAction(ActionType.PLAY_CARD, card_index=idx))


def act(g: TrucoGame, t: ActionType) -> None:
    g.step(PlayerAction(t))
