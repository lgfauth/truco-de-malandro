"""Pure game rules for Truco Mineiro (Paulista variant).

Two-player heads-up Truco. No RL, no I/O — only rules and state transitions.

Card hierarchy (non-manilha): 3 > 2 > A > K > J > Q > 7 > 6 > 5 > 4
Manilha: the rank immediately following the "vira" in the cyclic order
    4 -> 5 -> 6 -> 7 -> Q -> J -> K -> A -> 2 -> 3 -> 4 ...
Manilha tiebreak by suit: Clubs > Hearts > Spades > Diamonds.

Match: best to 12 points.
Hand (mão): best of 3 rounds. Points awarded depend on the current stake
    (1 -> 3 -> 6 -> 9 -> 12) when "truco" is escalated.
Mão de 11: when one team has 11 points, the next hand is played open
    and worth a fixed 1 point (no truco allowed).
Mão de 10 (iron hand): when both teams have 11 points, hand is worth 1 point
    and players play with hidden cards (no truco).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import List, Optional, Tuple
import random


class Suit(IntEnum):
    DIAMONDS = 0  # Ouros
    SPADES = 1    # Espadas
    HEARTS = 2    # Copas
    CLUBS = 3     # Paus


class Rank(IntEnum):
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    QUEEN = 10
    JACK = 11
    KING = 12
    ACE = 13
    TWO = 14
    THREE = 15


# Cyclic rank order used to compute the manilha from the vira.
# Following the Paulista convention: 4,5,6,7,Q,J,K,A,2,3 then wraps.
_RANK_CYCLE: List[Rank] = [
    Rank.FOUR, Rank.FIVE, Rank.SIX, Rank.SEVEN,
    Rank.QUEEN, Rank.JACK, Rank.KING, Rank.ACE,
    Rank.TWO, Rank.THREE,
]

# Strength of non-manilha cards (higher = stronger).
_NON_MANILHA_STRENGTH = {
    Rank.FOUR: 0,
    Rank.FIVE: 1,
    Rank.SIX: 2,
    Rank.SEVEN: 3,
    Rank.QUEEN: 4,
    Rank.JACK: 5,
    Rank.KING: 6,
    Rank.ACE: 7,
    Rank.TWO: 8,
    Rank.THREE: 9,
}


@dataclass(frozen=True)
class Card:
    rank: Rank
    suit: Suit

    def __repr__(self) -> str:
        rank_str = {
            Rank.FOUR: "4", Rank.FIVE: "5", Rank.SIX: "6", Rank.SEVEN: "7",
            Rank.QUEEN: "Q", Rank.JACK: "J", Rank.KING: "K", Rank.ACE: "A",
            Rank.TWO: "2", Rank.THREE: "3",
        }[self.rank]
        suit_str = {
            Suit.DIAMONDS: "♦", Suit.SPADES: "♠",
            Suit.HEARTS: "♥", Suit.CLUBS: "♣",
        }[self.suit]
        return f"{rank_str}{suit_str}"


def manilha_rank(vira: Card) -> Rank:
    """The rank that becomes manilha for a given vira."""
    idx = _RANK_CYCLE.index(vira.rank)
    return _RANK_CYCLE[(idx + 1) % len(_RANK_CYCLE)]


def card_strength(card: Card, vira: Card) -> int:
    """Return the absolute strength of a card given the vira.

    Manilhas are stronger than any non-manilha. Among manilhas, suit decides:
    Clubs > Hearts > Spades > Diamonds.
    """
    m_rank = manilha_rank(vira)
    if card.rank == m_rank:
        # Manilha: base 100 + suit value (Suit IntEnum already orders
        # Diamonds<Spades<Hearts<Clubs, which matches the rule).
        return 100 + int(card.suit)
    return _NON_MANILHA_STRENGTH[card.rank]


def compare_cards(a: Card, b: Card, vira: Card) -> int:
    """Return 1 if a beats b, -1 if b beats a, 0 if tie."""
    sa, sb = card_strength(a, vira), card_strength(b, vira)
    if sa > sb:
        return 1
    if sb > sa:
        return -1
    return 0


class Deck:
    """40-card Truco deck (no 8/9/10)."""

    @staticmethod
    def full() -> List[Card]:
        return [Card(r, s) for r in Rank for s in Suit]

    @staticmethod
    def shuffled(rng: Optional[random.Random] = None) -> List[Card]:
        rng = rng or random.Random()
        cards = Deck.full()
        rng.shuffle(cards)
        return cards


class Player(IntEnum):
    P0 = 0
    P1 = 1


def other(p: Player) -> Player:
    return Player.P1 if p == Player.P0 else Player.P0


class RoundResult(IntEnum):
    P0_WIN = 0
    P1_WIN = 1
    TIE = 2


class ActionType(Enum):
    PLAY_CARD = "play_card"
    CALL_TRUCO = "call_truco"   # raise stake to next level
    ACCEPT = "accept"           # accept current raise
    RUN = "run"                 # fold; opponent gets previous stake
    RAISE = "raise"             # counter-raise to next level


@dataclass
class PlayerAction:
    type: ActionType
    card_index: Optional[int] = None  # only when type == PLAY_CARD


# Stake progression. The 1 is the base value of an unraised hand.
# Raises move the *pending* value: 1 -> 3 -> 6 -> 9 -> 12.
_STAKE_LADDER = [1, 3, 6, 9, 12]


@dataclass
class Round:
    """A single round (vaza) within a hand."""
    plays: List[Tuple[Player, Card]] = field(default_factory=list)
    result: Optional[RoundResult] = None

    def is_complete(self) -> bool:
        return self.result is not None


@dataclass
class Hand:
    """One hand (mão) — best of 3 rounds for a stake."""
    vira: Card
    hands: dict  # Player -> List[Card]
    first_to_play: Player
    rounds: List[Round] = field(default_factory=lambda: [Round()])
    current_player: Player = Player.P0
    # Stake state
    stake: int = 1                # value if hand ends now (winner takes this)
    pending_stake: Optional[int] = None  # value being negotiated by truco call
    truco_caller: Optional[Player] = None  # who issued the pending raise
    truco_locked_for: Optional[Player] = None  # cannot re-raise until opp raises
    # Player whose normal turn was interrupted by the first truco call in the
    # current chain. Restored on accept so that after a multi-raise sequence
    # the original active player resumes — a plain flip of current_player is
    # only correct after a single call/accept pair.
    pre_truco_player: Optional[Player] = None
    # Mão de 11: opponent of the player at 11 must accept (play for 3) or run
    # (concede 1) before any card is played. While True, current_player is the
    # responder.
    awaiting_mao11_response: bool = False
    # Result
    winner: Optional[Player] = None
    folded: bool = False

    def is_over(self) -> bool:
        return self.winner is not None


@dataclass
class GameState:
    """Full match state."""
    scores: List[int] = field(default_factory=lambda: [0, 0])  # [P0, P1]
    dealer: Player = Player.P1   # non-dealer leads first hand
    hand: Optional[Hand] = None
    winner: Optional[Player] = None
    # Special hand modes (computed when hand starts)
    iron_hand: bool = False      # both at 11 — fixed 3 points, no truco
    open_hand_for: Optional[Player] = None  # this player has 11 — opp sees their cards


class TrucoGame:
    """Heads-up Truco match controller."""

    TARGET_SCORE = 12

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self.state = GameState()
        self._start_hand()

    # ------------------------------------------------------------------
    # Hand lifecycle
    # ------------------------------------------------------------------
    def _start_hand(self) -> None:
        s = self.state
        # Guard: if a player already reached TARGET_SCORE, finalise the match
        # instead of dealing a new hand. This handles any edge case where
        # _finish_hand called _start_hand despite the game being over.
        if s.scores[0] >= self.TARGET_SCORE or s.scores[1] >= self.TARGET_SCORE:
            s.winner = Player.P0 if s.scores[0] >= self.TARGET_SCORE else Player.P1
            return
        deck = Deck.shuffled(self._rng)
        # Deal 3 to each, then flip vira.
        p0_cards = [deck.pop(), deck.pop(), deck.pop()]
        p1_cards = [deck.pop(), deck.pop(), deck.pop()]
        vira = deck.pop()

        first = other(s.dealer)

        # Determine special modes
        p0_at_11 = s.scores[0] == self.TARGET_SCORE - 1
        p1_at_11 = s.scores[1] == self.TARGET_SCORE - 1
        s.iron_hand = p0_at_11 and p1_at_11
        if s.iron_hand:
            s.open_hand_for = None
        elif p0_at_11:
            s.open_hand_for = Player.P0
        elif p1_at_11:
            s.open_hand_for = Player.P1
        else:
            s.open_hand_for = None

        initial_stake = 3 if s.iron_hand else 1
        awaiting = s.open_hand_for is not None
        responder = other(s.open_hand_for) if s.open_hand_for is not None else first

        s.hand = Hand(
            vira=vira,
            hands={Player.P0: p0_cards, Player.P1: p1_cards},
            first_to_play=first,
            current_player=responder if awaiting else first,
            stake=initial_stake,
            awaiting_mao11_response=awaiting,
        )

    def _finish_hand(self, winner: Player, points: int) -> None:
        s = self.state
        s.scores[int(winner)] += points
        if s.scores[int(winner)] >= self.TARGET_SCORE:
            s.winner = winner
            return
        # Alternate dealer for next hand.
        s.dealer = other(s.dealer)
        self._start_hand()

    # ------------------------------------------------------------------
    # Legal actions
    # ------------------------------------------------------------------
    def legal_actions(self) -> List[PlayerAction]:
        s = self.state
        if s.winner is not None or s.hand is None:
            return []
        h = s.hand
        actions: List[PlayerAction] = []

        if h.awaiting_mao11_response:
            # Only the responder may accept (play for 3) or run (concede 1).
            actions.append(PlayerAction(ActionType.ACCEPT))
            actions.append(PlayerAction(ActionType.RUN))
            return actions

        if h.pending_stake is not None:
            # Responder must accept, run, or counter-raise.
            actions.append(PlayerAction(ActionType.ACCEPT))
            actions.append(PlayerAction(ActionType.RUN))
            next_stake = self._next_stake_after(h.pending_stake)
            if next_stake is not None and not self._truco_disabled():
                actions.append(PlayerAction(ActionType.RAISE))
            return actions

        # Normal turn: play a card, optionally call truco first.
        for i in range(len(h.hands[h.current_player])):
            actions.append(PlayerAction(ActionType.PLAY_CARD, card_index=i))
        if self._can_call_truco():
            actions.append(PlayerAction(ActionType.CALL_TRUCO))
        return actions

    def _truco_disabled(self) -> bool:
        s = self.state
        # Iron hand never allows raises. While the Mão de 11 response is
        # pending, no truco call is possible either.
        if s.iron_hand:
            return True
        if s.hand is not None and s.hand.awaiting_mao11_response:
            return True
        return False

    def _can_call_truco(self) -> bool:
        s = self.state
        if self._truco_disabled():
            return False
        h = s.hand
        if h.pending_stake is not None:
            return False
        # If last raise was made by the current player, they cannot raise again
        # until the opponent raises back.
        if h.truco_locked_for == h.current_player:
            return False
        next_stake = self._next_stake_after(h.stake)
        return next_stake is not None

    def _next_stake_after(self, value: int) -> Optional[int]:
        try:
            idx = _STAKE_LADDER.index(value)
        except ValueError:
            return None
        if idx + 1 >= len(_STAKE_LADDER):
            return None
        # If raising would put either player over the target, cap at remaining.
        # Truco rule: cannot raise to a value that exceeds what's needed.
        s = self.state
        remaining = self.TARGET_SCORE - max(s.scores)
        candidate = _STAKE_LADDER[idx + 1]
        if candidate > remaining and value >= remaining:
            return None
        return candidate

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action: PlayerAction) -> None:
        s = self.state
        if s.winner is not None or s.hand is None:
            raise RuntimeError("Game is over.")
        h = s.hand

        if action.type == ActionType.PLAY_CARD:
            self._handle_play(action.card_index)
        elif action.type == ActionType.CALL_TRUCO:
            self._handle_call_truco()
        elif action.type == ActionType.ACCEPT:
            self._handle_accept()
        elif action.type == ActionType.RUN:
            self._handle_run()
        elif action.type == ActionType.RAISE:
            self._handle_raise()
        else:
            raise ValueError(f"Unknown action: {action}")

    def _handle_play(self, card_index: Optional[int]) -> None:
        h = self.state.hand
        if h.pending_stake is not None:
            raise RuntimeError("Must respond to truco before playing.")
        if card_index is None:
            raise ValueError("card_index required for PLAY_CARD")
        hand_cards = h.hands[h.current_player]
        if not (0 <= card_index < len(hand_cards)):
            raise ValueError("Invalid card index.")
        card = hand_cards.pop(card_index)
        rnd = h.rounds[-1]
        rnd.plays.append((h.current_player, card))

        if len(rnd.plays) == 2:
            self._resolve_round()
        else:
            h.current_player = other(h.current_player)

    def _resolve_round(self) -> None:
        h = self.state.hand
        rnd = h.rounds[-1]
        (p_a, card_a), (p_b, card_b) = rnd.plays
        cmp = compare_cards(card_a, card_b, h.vira)
        if cmp > 0:
            rnd.result = RoundResult.P0_WIN if p_a == Player.P0 else RoundResult.P1_WIN
            next_leader = p_a
        elif cmp < 0:
            rnd.result = RoundResult.P0_WIN if p_b == Player.P0 else RoundResult.P1_WIN
            next_leader = p_b
        else:
            rnd.result = RoundResult.TIE
            next_leader = h.first_to_play  # tied round: first-to-play stays leader

        winner = self._check_hand_winner()
        if winner is not None:
            h.winner = winner
            self._finish_hand(winner, h.stake)
            return

        # Start next round
        h.rounds.append(Round())
        h.current_player = next_leader
        h.first_to_play = next_leader

    def _check_hand_winner(self) -> Optional[Player]:
        """Decide the hand winner based on completed rounds. None if undecided.

        Truco best-of-3 with tie shortcuts:
            - 2 round wins → take the hand.
            - Round 1 tied, round 2 has a winner → that winner takes the hand
              immediately (no round 3).
            - Round 2 tied, round 1 had a winner → round 1 winner takes the
              hand immediately.
            - Rounds 1 and 2 both tied → round 3 decides.
            - All three tied → the hand-leader (first_to_play) wins.
        """
        h = self.state.hand
        results = [r.result for r in h.rounds if r.is_complete()]
        wins = [0, 0]
        for r in results:
            if r == RoundResult.P0_WIN:
                wins[0] += 1
            elif r == RoundResult.P1_WIN:
                wins[1] += 1

        if wins[0] >= 2:
            return Player.P0
        if wins[1] >= 2:
            return Player.P1
        if len(results) < 2:
            return None

        # Apply early-resolution tie rules after round 2 completes.
        r1, r2 = results[0], results[1]
        if r1 == RoundResult.TIE and r2 != RoundResult.TIE:
            return Player.P0 if r2 == RoundResult.P0_WIN else Player.P1
        if r2 == RoundResult.TIE and r1 != RoundResult.TIE:
            return Player.P0 if r1 == RoundResult.P0_WIN else Player.P1
        # Both r1 and r2 tied → round 3 decides.
        if len(results) < 3:
            return None

        # All three played and still undecided (only possible if r1 and r2
        # both tied). The third round, if non-tied, wins it; if all three
        # tied, the original leader takes the hand.
        r3 = results[2]
        if r3 != RoundResult.TIE:
            return Player.P0 if r3 == RoundResult.P0_WIN else Player.P1
        return h.first_to_play

    # -------- Truco handling --------
    def _handle_call_truco(self) -> None:
        h = self.state.hand
        if not self._can_call_truco():
            raise RuntimeError("Cannot call truco now.")
        next_stake = self._next_stake_after(h.stake)
        # First call in this chain: snapshot the active player so accept can
        # restore them after any number of intervening raises.
        if h.pre_truco_player is None:
            h.pre_truco_player = h.current_player
        h.pending_stake = next_stake
        h.truco_caller = h.current_player
        # Responder must answer.
        h.current_player = other(h.current_player)

    def _handle_accept(self) -> None:
        s = self.state
        h = s.hand
        if h.awaiting_mao11_response:
            # Mão de 11: responder accepts to play the hand worth 3 points.
            h.stake = 3
            h.awaiting_mao11_response = False
            h.current_player = h.first_to_play
            return
        if h.pending_stake is None:
            raise RuntimeError("No pending truco to accept.")
        h.stake = h.pending_stake
        # Caller is locked from raising again until opponent raises.
        h.truco_locked_for = h.truco_caller
        h.pending_stake = None
        h.truco_caller = None
        # Restore the player whose normal turn was interrupted by the first
        # call in this chain. After a single call/accept this equals
        # other(current); after raise chains the simple flip would route the
        # turn to the wrong player (whose card has already been played).
        h.current_player = h.pre_truco_player if h.pre_truco_player is not None \
            else other(h.current_player)
        h.pre_truco_player = None

    def _handle_run(self) -> None:
        s = self.state
        h = s.hand
        if h.awaiting_mao11_response:
            # Mão de 11: responder concedes 1 point to the player at 11.
            winner = s.open_hand_for
            h.folded = True
            h.winner = winner
            self._finish_hand(winner, 1)
            return
        if h.pending_stake is None:
            raise RuntimeError("No pending truco to refuse.")
        # Folder loses; opponent gets the *previous* stake (before the raise).
        winner = h.truco_caller
        h.folded = True
        h.winner = winner
        self._finish_hand(winner, h.stake)

    def _handle_raise(self) -> None:
        h = self.state.hand
        if h.pending_stake is None:
            raise RuntimeError("No pending truco to raise on.")
        if self._truco_disabled():
            raise RuntimeError("Raises disabled this hand.")
        next_stake = self._next_stake_after(h.pending_stake)
        if next_stake is None:
            raise RuntimeError("No higher stake available.")
        # Counter-raise: previous accepted stake stays as the fallback (h.stake)
        # if the original caller folds. The new pending stake supersedes.
        h.stake = h.pending_stake  # implicit accept of the previous level
        h.pending_stake = next_stake
        h.truco_caller = h.current_player
        h.current_player = other(h.current_player)
