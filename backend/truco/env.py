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

from .encoding import (  # noqa: F401  (re-exported for callers)
    ACT_ACCEPT,
    ACT_CALL_TRUCO,
    ACT_PLAY_0,
    ACT_PLAY_1,
    ACT_PLAY_2,
    ACT_RAISE,
    ACT_RUN,
    NUM_ACTIONS,
    NUM_CARDS,
    action_mask,
    actions_equal,
    card_to_id,
    decode_action,
    legal_action_ids,
)
from .game import Player, TrucoGame
from .obs import DEFAULT_OBS_VERSION, obs_dim, observe
from .seeds import TRAIN_SEED_LIMIT

# Observation layout: see :mod:`truco.obs`. ``OBS_DIM`` is the v1 size used by
# the existing checkpoints.
OBS_DIM = obs_dim("v1")


class TrucoEnv(gym.Env):
    """Heads-up Truco environment from P0's perspective."""

    metadata = {"render_modes": ["human"]}

    TARGET_SCORE = TrucoGame.TARGET_SCORE
    ILLEGAL_ACTION_REWARD = -1.0

    def __init__(self, seed: Optional[int] = None, render_mode: Optional[str] = None,
                 obs_version: str = DEFAULT_OBS_VERSION):
        super().__init__()
        self.render_mode = render_mode
        self.obs_version = obs_version
        # Applied on the first reset() that does not pass its own seed.
        self._init_seed = seed
        self._seeded = False
        self._opp_rng = random.Random()
        self.game: Optional[TrucoGame] = None
        self.game_seed: Optional[int] = None

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim(obs_version),), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new match with a fresh deal sequence.

        Following the Gymnasium convention the env's RNG is seeded once (by
        ``seed`` here or the constructor seed on the first reset) and every
        later reset derives a new game from it. ``options={"game_seed": s}``
        forces a specific deal sequence (used for fixed evaluation sets).
        """
        if seed is None and not self._seeded:
            seed = self._init_seed
        super().reset(seed=seed)
        self._seeded = True
        self._opp_rng = random.Random(int(self.np_random.integers(0, 2**63)))
        game_seed = (options or {}).get("game_seed")
        if game_seed is None:
            game_seed = int(self.np_random.integers(0, TRAIN_SEED_LIMIT))
        self.game_seed = int(game_seed)
        self.game = TrucoGame(seed=self.game_seed)
        # If P1 leads, let it play until P0 must act.
        self._play_opponent_until_p0()
        return self._observe(), self._info()

    def step(self, action: int):
        assert self.game is not None, "Call reset() before step()."
        pa = decode_action(int(action))

        # Validate against legal actions for P0.
        if self.game.state.hand is None or self.game.state.winner is not None:
            return self._observe(), 0.0, True, False, self._info()

        if self.game.state.hand.current_player != Player.P0:
            # Should not happen if reset/step are used correctly.
            return self._observe(), 0.0, False, False, self._info()

        legal = self.game.legal_actions()
        if not any(actions_equal(pa, la) for la in legal):
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
        print(f"Iron hand: {s.iron_hand}   mao11: {s.mao11_player}")
        print(f"P0 hand: {[str(c) for c in h.hands[Player.P0]]}")
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
        return observe(self.game, Player.P0, self.obs_version)

    def _info(self) -> dict[str, Any]:
        g = self.game
        if g is None:
            return {}
        return {
            "legal_actions": legal_action_ids(g, Player.P0),
            "action_mask": action_mask(g, Player.P0),
            "scores": tuple(g.state.scores),
        }

    # sb3_contrib's MaskablePPO discovers masks by calling ``action_masks``
    # (plural) on the env directly. This is the canonical interface — no
    # wrapper needed.
    def action_masks(self) -> np.ndarray:
        return self._info()["action_mask"]

