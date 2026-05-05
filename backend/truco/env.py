"""Gymnasium environment wrapping :class:`TrucoGame` for single-agent RL.

The learner controls P0. P1 is a uniform random policy over legal actions in
this base environment — it will be replaced by self-play later. The episode is
a full match (first to 12 points). Reward is sparse: +1 for winning the match,
-1 for losing, 0 otherwise. Illegal actions are penalized with -1 without
advancing the underlying game.
"""

from __future__ import annotations

import random
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .game import (
    ActionType,
    Card,
    Player,
    PlayerAction,
    Rank,
    Suit,
    TrucoGame,
)


# --- Card <-> index encoding (40 cards, 10 ranks * 4 suits) -----------------
_RANKS = [Rank.FOUR, Rank.FIVE, Rank.SIX, Rank.SEVEN,
          Rank.QUEEN, Rank.JACK, Rank.KING, Rank.ACE,
          Rank.TWO, Rank.THREE]
_RANK_TO_IDX = {r: i for i, r in enumerate(_RANKS)}
NUM_CARDS = 40


def card_to_id(card: Card) -> int:
    return _RANK_TO_IDX[card.rank] * 4 + int(card.suit)


# --- Action encoding --------------------------------------------------------
ACT_PLAY_0 = 0
ACT_PLAY_1 = 1
ACT_PLAY_2 = 2
ACT_CALL_TRUCO = 3
ACT_ACCEPT = 4
ACT_RUN = 5
ACT_RAISE = 6
NUM_ACTIONS = 7


def _decode_action(a: int) -> PlayerAction:
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


def _encode_action(pa: PlayerAction) -> int:
    if pa.type == ActionType.PLAY_CARD:
        return int(pa.card_index)
    return {
        ActionType.CALL_TRUCO: ACT_CALL_TRUCO,
        ActionType.ACCEPT: ACT_ACCEPT,
        ActionType.RUN: ACT_RUN,
        ActionType.RAISE: ACT_RAISE,
    }[pa.type]


# --- Observation layout -----------------------------------------------------
# All values are floats. Card slots store (card_id + 1) / NUM_CARDS so that 0
# represents "empty"; normalized cards lie in (0, 1].
#   [0..2]   P0 hand cards (3 slots)
#   [3]      vira
#   [4..9]   rounds: 3 rounds x 2 plays (in play order)
#   [10]     score P0 / TARGET
#   [11]     score P1 / TARGET
#   [12]     current stake / 12
#   [13]     pending stake / 12 (0 if none)
#   [14]     has pending stake (0/1)
#   [15]     iron hand (0/1)
#   [16]     awaiting mao11 response (0/1)
#   [17]     open hand for P0 (0/1)  — opponent sees P0's cards
#   [18]     open hand for P1 (0/1)  — P0 sees P1's cards
#   [19]     current player == P0 (0/1)
#   [20]     dealer == P0 (0/1)
#   [21..23] P1 hand cards (3 slots) — only filled when open_hand_for == P1
#            (Mão de 11: P1 has 11, P0 is the responder and sees their cards),
#            otherwise zero.
OBS_DIM = 24


class TrucoEnv(gym.Env):
    """Heads-up Truco environment from P0's perspective."""

    metadata = {"render_modes": ["human"]}

    TARGET_SCORE = TrucoGame.TARGET_SCORE
    ILLEGAL_ACTION_REWARD = -1.0

    def __init__(self, seed: Optional[int] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode
        self._seed = seed
        self._opp_rng = random.Random(seed)
        self.game: Optional[TrucoGame] = None

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
            self._opp_rng = random.Random(seed)
        self.game = TrucoGame(seed=self._seed)
        # If P1 leads, let it play until P0 must act.
        self._play_opponent_until_p0()
        return self._observe(), self._info()

    def step(self, action: int):
        assert self.game is not None, "Call reset() before step()."
        pa = _decode_action(int(action))

        # Validate against legal actions for P0.
        if self.game.state.hand is None or self.game.state.winner is not None:
            return self._observe(), 0.0, True, False, self._info()

        if self.game.state.hand.current_player != Player.P0:
            # Should not happen if reset/step are used correctly.
            return self._observe(), 0.0, False, False, self._info()

        legal = self.game.legal_actions()
        if not any(_actions_equal(pa, la) for la in legal):
            return (
                self._observe(),
                self.ILLEGAL_ACTION_REWARD,
                False,
                False,
                {**self._info(), "illegal_action": True},
            )

        self.game.step(pa)
        # Let opponent act until P0 must respond or the match is over.
        self._play_opponent_until_p0()

        terminated = self.game.state.winner is not None
        reward = 0.0
        if terminated:
            reward = 1.0 if self.game.state.winner == Player.P0 else -1.0
        return self._observe(), reward, terminated, False, self._info()

    def render(self):
        if self.game is None:
            print("<env not reset>")
            return
        s = self.game.state
        print(f"Score: P0={s.scores[0]}  P1={s.scores[1]}  dealer={s.dealer.name}")
        if s.winner is not None:
            print(f"Match winner: {s.winner.name}")
            return
        h = s.hand
        print(f"Vira: {h.vira}   stake={h.stake}   pending={h.pending_stake}")
        print(f"Iron hand: {s.iron_hand}   open_for: {s.open_hand_for}")
        print(f"P0 hand: {[str(c) for c in h.hands[Player.P0]]}")
        if s.open_hand_for == Player.P1:
            print(f"P1 hand (open): {[str(c) for c in h.hands[Player.P1]]}")
        for i, rnd in enumerate(h.rounds):
            plays = [(p.name, str(c)) for p, c in rnd.plays]
            print(f"  Round {i+1}: {plays}  result={rnd.result}")
        print(f"To act: {h.current_player.name}   "
              f"awaiting_mao11={h.awaiting_mao11_response}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _play_opponent_until_p0(self) -> None:
        g = self.game
        while (g.state.winner is None
               and g.state.hand is not None
               and g.state.hand.current_player == Player.P1):
            legal = g.legal_actions()
            if not legal:
                break
            pa = self._opp_rng.choice(legal)
            g.step(pa)

    def _observe(self) -> np.ndarray:
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        g = self.game
        s = g.state
        h = s.hand

        if h is not None:
            for i, c in enumerate(h.hands[Player.P0][:3]):
                obs[i] = (card_to_id(c) + 1) / NUM_CARDS
            obs[3] = (card_to_id(h.vira) + 1) / NUM_CARDS

            slot = 4
            for r in range(3):
                if r < len(h.rounds):
                    plays = h.rounds[r].plays
                    for k in range(2):
                        if k < len(plays):
                            _, card = plays[k]
                            obs[slot] = (card_to_id(card) + 1) / NUM_CARDS
                        slot += 1
                else:
                    slot += 2

            obs[12] = h.stake / 12.0
            obs[13] = (h.pending_stake or 0) / 12.0
            obs[14] = 1.0 if h.pending_stake is not None else 0.0
            obs[16] = 1.0 if h.awaiting_mao11_response else 0.0
            obs[19] = 1.0 if h.current_player == Player.P0 else 0.0

        obs[10] = s.scores[0] / self.TARGET_SCORE
        obs[11] = s.scores[1] / self.TARGET_SCORE
        obs[15] = 1.0 if s.iron_hand else 0.0
        obs[17] = 1.0 if s.open_hand_for == Player.P0 else 0.0
        obs[18] = 1.0 if s.open_hand_for == Player.P1 else 0.0
        obs[20] = 1.0 if s.dealer == Player.P0 else 0.0

        if h is not None and s.open_hand_for == Player.P1:
            for i, c in enumerate(h.hands[Player.P1][:3]):
                obs[21 + i] = (card_to_id(c) + 1) / NUM_CARDS
        return obs

    def _info(self) -> dict[str, Any]:
        g = self.game
        if g is None:
            return {}
        if g.state.hand is None or g.state.hand.current_player != Player.P0:
            legal_ids: list[int] = []
        else:
            legal_ids = sorted({_encode_action(a) for a in g.legal_actions()})
        mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
        for a in legal_ids:
            mask[a] = 1
        return {
            "legal_actions": legal_ids,
            "action_mask": mask,
            "scores": tuple(g.state.scores),
        }

    # sb3_contrib's MaskablePPO discovers masks by calling ``action_masks``
    # (plural) on the env directly. This is the canonical interface — no
    # wrapper needed.
    def action_masks(self) -> np.ndarray:
        return self._info()["action_mask"]


def _actions_equal(a: PlayerAction, b: PlayerAction) -> bool:
    return a.type == b.type and a.card_index == b.card_index
