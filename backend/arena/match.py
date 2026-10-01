"""Play one match between two arena players and collect per-side statistics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from truco.encoding import (
    ACT_ACCEPT,
    ACT_CALL_TRUCO,
    ACT_RAISE,
    ACT_RUN,
    ACTION_NAMES,
    legal_action_ids,
    to_legal_action,
)
from truco.game import Hand
from truco.game import Player as Seat
from truco.game import TrucoGame

from .players import ArenaPlayer

MAX_DECISIONS = 5000  # safety net; real matches take well under 200


@dataclass
class SideStats:
    decisions: int = 0
    illegal: int = 0        # returned a valid id that was not legal
    fallbacks: int = 0      # arena substituted the first legal action
    errors: int = 0         # decide() raised or returned a non-action
    truco_calls: int = 0
    raises: int = 0
    accepts: int = 0
    runs: int = 0
    bet_responses: int = 0  # decisions facing a bet or the Mão de 11 choice
    points: int = 0
    hands_won: int = 0

    def add(self, other: "SideStats") -> None:
        for k, v in asdict(other).items():
            setattr(self, k, getattr(self, k) + v)


@dataclass
class MatchResult:
    game_seed: int
    a_seat: int
    a_won: bool
    scores: Dict[str, int]
    hands: int
    drawn_hands: int
    stats: Dict[str, SideStats] = field(default_factory=dict)


def _card(c) -> str:
    return str(c)


def _state_snapshot(game: TrucoGame, seat: Seat, legal: List[int]) -> Dict[str, Any]:
    """Readable decision context from ``seat``'s point of view."""
    s = game.state
    h = s.hand
    return {
        "scores": [s.scores[int(seat)], s.scores[1 - int(seat)]],
        "vira": _card(h.vira),
        "hand": [_card(c) for c in h.hands[seat]],
        "rounds": [
            {"plays": [["me" if p == seat else "opp", _card(c)] for p, c in r.plays],
             "result": None if r.result is None else int(r.result)}
            for r in h.rounds
        ],
        "stake": h.stake,
        "pending_stake": h.pending_stake,
        "mao11": s.mao11_player is not None,
        "iron_hand": s.iron_hand,
        "legal": legal,
    }


def play_match(a: ArenaPlayer, b: ArenaPlayer, game_seed: int, a_seat: int,
               log: Optional[List[dict]] = None) -> MatchResult:
    """Play a full match. ``a`` sits at ``a_seat`` (0 = P0, who leads first).

    When ``log`` is given, one dict per decision is appended; each carries
    the result of the hand it belongs to.
    """
    game = TrucoGame(seed=game_seed)
    seat_a = Seat(a_seat)
    roles = {seat_a: ("a", a), Seat(1 - a_seat): ("b", b)}
    a.reset(f"{game_seed}:{a_seat}:a")
    b.reset(f"{game_seed}:{a_seat}:b")
    stats = {"a": SideStats(), "b": SideStats()}

    hands = 1
    drawn = 0
    hand_idx = 0
    pending_log: List[dict] = []
    current_hand: Hand = game.state.hand

    def close_hand(h: Hand) -> None:
        nonlocal drawn
        if h.drawn:
            drawn += 1
            winner_role = None
        else:
            winner_role = roles[h.winner][0]
            stats[winner_role].hands_won += 1
        if log is not None:
            result = {"winner": winner_role, "stake": h.stake,
                      "folded": h.folded, "drawn": h.drawn}
            for entry in pending_log:
                entry["hand_result"] = result
                log.append(entry)
            pending_log.clear()

    for _ in range(MAX_DECISIONS):
        if game.state.winner is not None:
            break
        h = game.state.hand
        seat = h.current_player
        role, player = roles[seat]
        st = stats[role]
        legal = legal_action_ids(game, seat)
        facing_bet = ACT_ACCEPT in legal

        try:
            raw = player.decide(game, seat)
            chosen = int(raw)
        except Exception:
            chosen = None
            st.errors += 1

        if chosen is not None and chosen in legal:
            action_id, fallback = chosen, False
        else:
            if chosen is not None and 0 <= chosen < len(ACTION_NAMES):
                st.illegal += 1
            elif chosen is not None:
                st.errors += 1
            action_id, fallback = legal[0], True
        if fallback:
            st.fallbacks += 1

        st.decisions += 1
        st.bet_responses += int(facing_bet)
        st.truco_calls += int(action_id == ACT_CALL_TRUCO)
        st.raises += int(action_id == ACT_RAISE)
        st.accepts += int(action_id == ACT_ACCEPT)
        st.runs += int(action_id == ACT_RUN)

        if log is not None:
            pending_log.append({
                "game_seed": game_seed, "a_seat": a_seat, "hand": hand_idx,
                "role": role, "seat": int(seat),
                "state": _state_snapshot(game, seat, legal),
                "action": action_id, "action_name": ACTION_NAMES[action_id],
                "raw_action": chosen, "fallback": fallback,
            })

        before = list(game.state.scores)
        pa, _ = to_legal_action(game, action_id)
        game.step(pa)
        after = game.state.scores
        stats[roles[Seat.P0][0]].points += after[0] - before[0]
        stats[roles[Seat.P1][0]].points += after[1] - before[1]

        if game.state.hand is not current_hand or game.state.winner is not None:
            close_hand(current_hand)
            if game.state.winner is None:
                hands += 1
                hand_idx += 1
                current_hand = game.state.hand
    else:
        raise RuntimeError(f"Match {game_seed} did not finish")

    winner = game.state.winner
    return MatchResult(
        game_seed=game_seed,
        a_seat=a_seat,
        a_won=winner == seat_a,
        scores={"a": game.state.scores[a_seat], "b": game.state.scores[1 - a_seat]},
        hands=hands,
        drawn_hands=drawn,
        stats=stats,
    )
