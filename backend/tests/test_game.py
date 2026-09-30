"""Characterization tests for the rules engine (truco/game.py)."""

from __future__ import annotations

import random

import pytest

from helpers import C, act, deal, play
from truco.game import (
    ActionType,
    Card,
    Deck,
    Player,
    Rank,
    RoundResult,
    Suit,
    TrucoGame,
    card_strength,
    compare_cards,
    manilha_rank,
)

A = ActionType


# --- Cards ------------------------------------------------------------------
def test_deck_has_40_unique_cards():
    cards = Deck.full()
    assert len(cards) == 40
    assert len(set(cards)) == 40


@pytest.mark.parametrize("vira,manilha", [
    (Rank.FOUR, Rank.FIVE),
    (Rank.SEVEN, Rank.QUEEN),
    (Rank.QUEEN, Rank.JACK),
    (Rank.ACE, Rank.TWO),
    (Rank.THREE, Rank.FOUR),  # wraps around
])
def test_manilha_follows_vira_cyclically(vira, manilha):
    assert manilha_rank(Card(vira, Suit.HEARTS)) == manilha


def test_non_manilha_order():
    vira = Card(Rank.FOUR, Suit.DIAMONDS)  # manilha = 5
    order = [Rank.SIX, Rank.SEVEN, Rank.QUEEN, Rank.JACK, Rank.KING,
             Rank.ACE, Rank.TWO, Rank.THREE]
    strengths = [card_strength(Card(r, Suit.CLUBS), vira) for r in order]
    assert strengths == sorted(strengths)
    assert len(set(strengths)) == len(strengths)


def test_manilha_beats_everything_and_suit_breaks_ties():
    vira = Card(Rank.FOUR, Suit.DIAMONDS)
    three = Card(Rank.THREE, Suit.CLUBS)
    zap = Card(Rank.FIVE, Suit.CLUBS)
    copas = Card(Rank.FIVE, Suit.HEARTS)
    espadas = Card(Rank.FIVE, Suit.SPADES)
    ouros = Card(Rank.FIVE, Suit.DIAMONDS)
    assert compare_cards(ouros, three, vira) == 1
    assert compare_cards(zap, copas, vira) == 1
    assert compare_cards(copas, espadas, vira) == 1
    assert compare_cards(espadas, ouros, vira) == 1


def test_same_rank_non_manilha_ties():
    assert compare_cards(C(Rank.KING, Suit.CLUBS), C(Rank.KING, Suit.HEARTS),
                         C(Rank.FOUR)) == 0


# --- Dealing ----------------------------------------------------------------
def test_same_seed_same_deal_and_different_seed_different_deal():
    def snapshot(g):
        h = g.state.hand
        return (tuple(h.hands[Player.P0]), tuple(h.hands[Player.P1]), h.vira)

    assert snapshot(TrucoGame(seed=5)) == snapshot(TrucoGame(seed=5))
    assert snapshot(TrucoGame(seed=5)) != snapshot(TrucoGame(seed=6))


def test_initial_state():
    g = TrucoGame(seed=1)
    s = g.state
    h = s.hand
    assert s.scores == [0, 0]
    assert s.dealer == Player.P1
    assert h.first_to_play == Player.P0 == h.current_player
    assert h.stake == 1 and h.pending_stake is None
    assert len(h.hands[Player.P0]) == len(h.hands[Player.P1]) == 3
    dealt = h.hands[Player.P0] + h.hands[Player.P1] + [h.vira]
    assert len(set(dealt)) == 7


# --- Rounds -----------------------------------------------------------------
def _hand_after(g):
    """Reference to the current hand (survives _finish_hand starting a new one)."""
    return g.state.hand


def test_two_round_wins_take_the_hand_and_dealer_rotates():
    g = deal([C(Rank.THREE), C(Rank.TWO), C(Rank.ACE)],
             [C(Rank.SIX), C(Rank.SEVEN), C(Rank.KING)])
    h = _hand_after(g)
    play(g); play(g)
    assert h.rounds[0].result == RoundResult.P0_WIN
    assert h.current_player == Player.P0  # round winner leads
    play(g); play(g)
    assert h.winner == Player.P0
    assert g.state.scores == [1, 0]
    assert g.state.hand is not h
    assert g.state.dealer == Player.P0
    assert g.state.hand.current_player == Player.P1


def test_round_loser_does_not_lead_next_round():
    g = deal([C(Rank.SIX), C(Rank.TWO), C(Rank.ACE)],
             [C(Rank.THREE), C(Rank.SEVEN), C(Rank.KING)])
    play(g); play(g)
    assert g.state.hand.rounds[0].result == RoundResult.P1_WIN
    assert g.state.hand.current_player == Player.P1


def test_tie_first_round_then_second_round_decides():
    g = deal([C(Rank.KING, Suit.CLUBS), C(Rank.SIX), C(Rank.ACE)],
             [C(Rank.KING, Suit.HEARTS), C(Rank.THREE), C(Rank.SEVEN)])
    h = _hand_after(g)
    play(g); play(g)
    assert h.rounds[0].result == RoundResult.TIE
    assert h.current_player == Player.P0  # leader of the tied round leads again
    play(g); play(g)
    assert h.winner == Player.P1
    assert len(h.rounds) == 2


def test_win_first_then_tie_second_first_round_winner_takes_it():
    g = deal([C(Rank.THREE), C(Rank.KING, Suit.CLUBS), C(Rank.ACE)],
             [C(Rank.SIX), C(Rank.KING, Suit.HEARTS), C(Rank.SEVEN)])
    h = _hand_after(g)
    play(g); play(g)
    play(g); play(g)
    assert h.rounds[1].result == RoundResult.TIE
    assert h.winner == Player.P0
    assert len(h.rounds) == 2


def test_two_ties_then_third_round_decides():
    g = deal([C(Rank.KING, Suit.CLUBS), C(Rank.ACE, Suit.CLUBS), C(Rank.SIX)],
             [C(Rank.KING, Suit.HEARTS), C(Rank.ACE, Suit.HEARTS), C(Rank.THREE)])
    h = _hand_after(g)
    for _ in range(3):
        play(g); play(g)
    assert [r.result for r in h.rounds] == [RoundResult.TIE, RoundResult.TIE,
                                            RoundResult.P1_WIN]
    assert h.winner == Player.P1


def test_win_loss_tie_goes_to_second_round_winner_CURRENT_BEHAVIOR():
    # Characterizes the pre-fix behavior; the Paulista rule gives it to the
    # first round winner (P0).
    g = deal([C(Rank.THREE), C(Rank.SIX), C(Rank.KING, Suit.CLUBS)],
             [C(Rank.SIX, Suit.SPADES), C(Rank.THREE, Suit.SPADES),
              C(Rank.KING, Suit.HEARTS)])
    h = _hand_after(g)
    for _ in range(3):
        play(g); play(g)
    assert [r.result for r in h.rounds] == [RoundResult.P0_WIN,
                                            RoundResult.P1_WIN, RoundResult.TIE]
    assert h.winner == Player.P1
    assert g.state.scores == [0, 1]


def test_three_ties_go_to_hand_leader_CURRENT_BEHAVIOR():
    g = deal([C(Rank.KING, Suit.CLUBS), C(Rank.ACE, Suit.CLUBS), C(Rank.SIX, Suit.CLUBS)],
             [C(Rank.KING, Suit.HEARTS), C(Rank.ACE, Suit.HEARTS), C(Rank.SIX, Suit.HEARTS)])
    h = _hand_after(g)
    for _ in range(3):
        play(g); play(g)
    assert h.winner == Player.P0
    assert g.state.scores == [1, 0]


# --- Truco ------------------------------------------------------------------
def _fresh():
    return deal([C(Rank.THREE), C(Rank.TWO), C(Rank.ACE)],
                [C(Rank.SIX), C(Rank.SEVEN), C(Rank.KING)])


def _types(g):
    return {(a.type, a.card_index) for a in g.legal_actions()}


def test_call_truco_then_accept():
    g = _fresh()
    h = g.state.hand
    act(g, A.CALL_TRUCO)
    assert h.pending_stake == 3 and h.current_player == Player.P1
    assert _types(g) == {(A.ACCEPT, None), (A.RUN, None), (A.RAISE, None)}
    act(g, A.ACCEPT)
    assert h.stake == 3 and h.pending_stake is None
    assert h.current_player == Player.P0
    # Caller cannot raise again until the opponent does.
    assert (A.CALL_TRUCO, None) not in _types(g)
    play(g)
    assert (A.CALL_TRUCO, None) in _types(g)  # P1 may ask for 6


def test_run_gives_previous_stake_to_caller():
    g = _fresh()
    act(g, A.CALL_TRUCO)
    act(g, A.RUN)
    assert g.state.scores == [1, 0]


def test_raise_chain_accept_restores_interrupted_player():
    g = _fresh()
    h = g.state.hand
    act(g, A.CALL_TRUCO)   # P0 asks 3
    act(g, A.RAISE)        # P1 asks 6
    assert h.stake == 3 and h.pending_stake == 6
    assert h.current_player == Player.P0
    act(g, A.ACCEPT)
    assert h.stake == 6
    assert h.current_player == Player.P0


def test_run_after_raise_pays_the_accepted_level():
    g = _fresh()
    act(g, A.CALL_TRUCO)
    act(g, A.RAISE)
    act(g, A.RUN)          # P0 folds to the 6
    assert g.state.scores == [0, 3]


def test_truco_mid_round_returns_turn_to_caller():
    g = _fresh()
    h = g.state.hand
    play(g)                # P0 leads
    act(g, A.CALL_TRUCO)   # P1 calls before answering the card
    act(g, A.ACCEPT)
    assert h.current_player == Player.P1


def test_raise_is_capped_near_the_target():
    g = deal([C(Rank.THREE), C(Rank.TWO), C(Rank.ACE)],
             [C(Rank.SIX), C(Rank.SEVEN), C(Rank.KING)], scores=[10, 0])
    act(g, A.CALL_TRUCO)
    assert _types(g) == {(A.ACCEPT, None), (A.RUN, None)}


# --- Special hands ------------------------------------------------------------
def test_iron_hand_is_worth_three_and_has_no_truco():
    g = deal([C(Rank.THREE), C(Rank.TWO), C(Rank.ACE)],
             [C(Rank.SIX), C(Rank.SEVEN), C(Rank.KING)], scores=[11, 11])
    s = g.state
    assert s.iron_hand and s.open_hand_for is None
    assert s.hand.stake == 3
    assert not s.hand.awaiting_mao11_response
    assert all(a.type == A.PLAY_CARD for a in g.legal_actions())


def test_mao_de_11_opponent_decides_CURRENT_BEHAVIOR():
    g = deal([C(Rank.THREE), C(Rank.TWO), C(Rank.ACE)],
             [C(Rank.SIX), C(Rank.SEVEN), C(Rank.KING)], scores=[11, 5])
    s = g.state
    assert s.open_hand_for == Player.P0
    assert s.hand.awaiting_mao11_response
    assert s.hand.current_player == Player.P1
    assert _types(g) == {(A.ACCEPT, None), (A.RUN, None)}
    act(g, A.ACCEPT)
    assert s.hand.stake == 3
    assert s.hand.current_player == s.hand.first_to_play


def test_mao_de_11_run_concedes_the_match_CURRENT_BEHAVIOR():
    g = deal([C(Rank.THREE), C(Rank.TWO), C(Rank.ACE)],
             [C(Rank.SIX), C(Rank.SEVEN), C(Rank.KING)], scores=[11, 5])
    act(g, A.RUN)
    assert g.state.scores == [12, 5]
    assert g.state.winner == Player.P0


# --- Match end ----------------------------------------------------------------
def test_match_ends_at_twelve():
    g = deal([C(Rank.THREE), C(Rank.TWO), C(Rank.ACE)],
             [C(Rank.SIX), C(Rank.SEVEN), C(Rank.KING)], scores=[10, 0])
    act(g, A.CALL_TRUCO)
    act(g, A.ACCEPT)
    play(g); play(g); play(g); play(g)
    assert g.state.scores == [13, 0]
    assert g.state.winner == Player.P0
    assert g.legal_actions() == []
    with pytest.raises(RuntimeError):
        play(g)


def test_random_playouts_always_terminate_consistently():
    for seed in range(300):
        g = TrucoGame(seed=seed)
        rng = random.Random(seed)
        prev = [0, 0]
        for _ in range(2000):
            if g.state.winner is not None:
                break
            legal = g.legal_actions()
            assert legal, "no legal action in a live game"
            g.step(rng.choice(legal))
            s = g.state.scores
            assert s[0] >= prev[0] and s[1] >= prev[1]
            prev = list(s)
        w = g.state.winner
        assert w is not None
        assert g.state.scores[int(w)] >= 12
        assert g.state.scores[1 - int(w)] < 12
