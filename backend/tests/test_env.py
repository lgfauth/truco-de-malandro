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


def _deals(env, n):
    out = []
    for _ in range(n):
        env.reset()
        out.append(_hand_key(env))
    return out


def test_consecutive_resets_deal_new_hands():
    env = TrucoEnv(seed=42)
    deals = _deals(env, 1000)
    assert len(set(deals)) == 1000


def test_same_initial_seed_reproduces_the_sequence():
    assert _deals(TrucoEnv(seed=42), 50) == _deals(TrucoEnv(seed=42), 50)
    assert _deals(TrucoEnv(seed=42), 50) != _deals(TrucoEnv(seed=43), 50)


def test_reset_seed_matches_constructor_seed_and_reseeds():
    a = TrucoEnv()
    a.reset(seed=7)
    first = _hand_key(a)
    rest = _deals(a, 5)
    b = TrucoEnv(seed=7)
    assert _deals(b, 6) == [first] + rest
    # An explicit seed later restarts the sequence.
    a.reset(seed=7)
    assert _hand_key(a) == first


def test_whole_episode_is_reproducible():
    def trajectory(seed):
        env = TrucoEnv(seed=seed)
        traj = []
        for _ in range(3):
            obs, info = env.reset()
            done = False
            while not done:
                a = info["legal_actions"][0]
                obs, r, terminated, truncated, info = env.step(a)
                traj.append((a, r, tuple(env.game.state.scores)))
                done = terminated or truncated
        return traj

    assert trajectory(5) == trajectory(5)


def test_unseeded_envs_differ():
    assert _deals(TrucoEnv(), 5) != _deals(TrucoEnv(), 5)


def test_training_and_eval_seed_spaces_are_disjoint():
    from truco.seeds import eval_game_seeds, is_eval_seed

    env = TrucoEnv(seed=0)
    for _ in range(200):
        env.reset()
        assert not is_eval_seed(env.game_seed)
    ev = eval_game_seeds(123, 500)
    assert len(set(ev)) == 500 and all(is_eval_seed(s) for s in ev)
    assert eval_game_seeds(123, 500) == ev


def test_game_seed_option_forces_the_deal():
    a, b = TrucoEnv(seed=1), TrucoEnv(seed=2)
    a.reset(options={"game_seed": 99})
    b.reset(options={"game_seed": 99})
    assert _hand_key(a) == _hand_key(b)


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
