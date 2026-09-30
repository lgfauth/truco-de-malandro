"""Characterization tests for the Gymnasium env (truco/env.py)."""

from __future__ import annotations

import numpy as np

from truco.encoding import NUM_ACTIONS, legal_action_ids
from truco.env import OBS_DIM, TrucoEnv
from truco.game import Player


def _hand_key(env):
    h = env.game.state.hand
    return (tuple(h.hands[Player.P0]), tuple(h.hands[Player.P1]), h.vira)


def test_reset_returns_valid_obs_and_mask():
    env = TrucoEnv(seed=3)
    obs, info = env.reset()
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert env.game.state.hand.current_player == Player.P0
    mask = env.action_masks()
    assert mask.shape == (NUM_ACTIONS,)
    assert sorted(np.flatnonzero(mask).tolist()) == info["legal_actions"]
    assert info["legal_actions"] == legal_action_ids(env.game, Player.P0)


def test_illegal_action_is_penalized_without_advancing():
    env = TrucoEnv(seed=3)
    obs, info = env.reset()
    illegal = next(a for a in range(NUM_ACTIONS) if a not in info["legal_actions"])
    obs2, reward, terminated, truncated, info2 = env.step(illegal)
    assert reward == TrucoEnv.ILLEGAL_ACTION_REWARD
    assert not terminated and not truncated
    assert info2.get("illegal_action") is True
    np.testing.assert_array_equal(obs, obs2)


def test_episodes_terminate_with_plus_minus_one():
    rng = np.random.default_rng(0)
    for seed in range(50):
        env = TrucoEnv(seed=seed)
        obs, info = env.reset()
        total, done, steps = 0.0, False, 0
        while not done:
            a = int(rng.choice(info["legal_actions"]))
            obs, r, terminated, truncated, info = env.step(a)
            total += r
            steps += 1
            done = terminated or truncated
            assert steps < 2000
        assert total in (1.0, -1.0)
        assert env.game.state.winner is not None
        assert (total > 0) == (env.game.state.winner == Player.P0)


def test_env_obs_matches_observe_for_p0():
    from truco.obs import observe

    env = TrucoEnv(seed=11)
    obs, info = env.reset()
    np.testing.assert_array_equal(obs, observe(env.game, Player.P0))


def test_fixed_seed_repeats_the_same_deal_every_reset_CURRENT_BEHAVIOR():
    # Documents the bug fixed in Phase 1.
    env = TrucoEnv(seed=42)
    env.reset()
    first = _hand_key(env)
    env.reset()
    assert _hand_key(env) == first


def test_first_legal_policy_always_finishes():
    # Replaces the old debug_env.py script.
    for seed in range(50):
        env = TrucoEnv(seed=seed)
        obs, info = env.reset()
        done = False
        while not done:
            assert info["legal_actions"], "no legal actions in a live state"
            obs, r, terminated, truncated, info = env.step(info["legal_actions"][0])
            done = terminated or truncated
        assert env.game.state.winner is not None
