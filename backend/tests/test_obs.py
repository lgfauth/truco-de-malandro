"""observe() must reproduce the v1 layout the existing checkpoints expect.

``data/obs_v1_golden.npz`` was first recorded from the original
``TrucoEnv._observe`` (P0) and ``_p1_view`` (P1) implementations over 300
random matches, and matched ``observe()`` exactly. It was re-recorded with
``data/make_obs_golden.py`` when the Mão de 11 and round-tie rules changed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from truco.encoding import action_mask, decode_action
from truco.game import Player, TrucoGame
from truco.obs import OBS_V1_DIM, observe

GOLDEN = Path(__file__).parent / "data" / "obs_v1_golden.npz"


def test_observe_v1_matches_golden_trajectories():
    d = np.load(GOLDEN)
    offsets = d["offsets"]
    for k, seed in enumerate(d["seeds"]):
        g = TrucoGame(seed=int(seed))
        for i in range(offsets[k], offsets[k + 1]):
            np.testing.assert_array_equal(observe(g, Player.P0), d["obs_p0"][i])
            np.testing.assert_array_equal(observe(g, Player.P1), d["obs_p1"][i])
            np.testing.assert_array_equal(action_mask(g, Player.P0), d["mask_p0"][i])
            np.testing.assert_array_equal(action_mask(g, Player.P1), d["mask_p1"][i])
            g.step(decode_action(int(d["actions"][i])))
        assert g.state.winner is not None


def test_observe_is_symmetric_and_bounded():
    g = TrucoGame(seed=7)
    o0, o1 = observe(g, Player.P0), observe(g, Player.P1)
    assert o0.shape == o1.shape == (OBS_V1_DIM,)
    assert o0.min() >= 0 and o0.max() <= 1
    # Shared public info (vira, table, stake) is identical from both sides.
    assert o0[3] == o1[3]
    np.testing.assert_array_equal(o0[12:17], o1[12:17])
    # Private info swaps.
    assert o0[19] + o1[19] == 1
    assert o0[20] + o1[20] == 1


def test_observe_never_reveals_opponent_cards():
    import random

    for seed in range(100):
        g = TrucoGame(seed=seed)
        rng = random.Random(seed)
        while g.state.winner is None:
            assert not observe(g, Player.P0)[21:].any()
            assert not observe(g, Player.P1)[21:].any()
            g.step(rng.choice(g.legal_actions()))


# --- v2 ---------------------------------------------------------------------
import random as _random

import pytest

from truco.encoding import RANK_TO_IDX
from truco.game import Deck, card_strength, manilha_rank
from truco.obs import OBS_V2_DIM, obs_dim, version_for_dim


def _random_states(n_matches=40):
    for seed in range(n_matches):
        g = TrucoGame(seed=seed)
        rng = _random.Random(seed)
        while g.state.winner is None:
            yield g
            g.step(rng.choice(g.legal_actions()))


def test_versions_are_detectable_from_dim():
    assert obs_dim("v2") == OBS_V2_DIM
    assert version_for_dim(24) == "v1"
    assert version_for_dim(OBS_V2_DIM) == "v2"
    with pytest.raises(ValueError):
        obs_dim("v9")


def test_v2_bounds_and_hand_slots_follow_hand_order():
    for g in _random_states():
        for p in (Player.P0, Player.P1):
            o = observe(g, p, "v2")
            assert o.shape == (OBS_V2_DIM,) and o.min() >= 0 and o.max() <= 1
            h = g.state.hand
            for k, card in enumerate(h.hands[p]):
                base = 17 * k
                assert o[base] == 1.0
                assert o[base + 1 + RANK_TO_IDX[card.rank]] == 1.0
                assert o[base + 11 + int(card.suit)] == 1.0
                assert o[base + 16] == float(card.rank == manilha_rank(h.vira))
            for k in range(len(h.hands[p]), 3):
                assert not o[17 * k:17 * (k + 1)].any()


def test_v2_strength_ranks_manilhas_on_top():
    g = TrucoGame(seed=3)
    h = g.state.hand
    o = observe(g, Player.P0, "v2")
    strengths = [o[17 * k + 15] for k in range(3)]
    raw = [card_strength(c, h.vira) for c in h.hands[Player.P0]]
    assert sorted(range(3), key=lambda k: strengths[k]) == sorted(range(3), key=lambda k: raw[k])


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_observation_does_not_depend_on_opponent_hidden_cards(version):
    checked = 0
    for g in _random_states(30):
        h = g.state.hand
        for me in (Player.P0, Player.P1):
            opp = Player(1 - int(me))
            if not h.hands[opp]:
                continue
            before = observe(g, me, version)
            visible = set(h.hands[me]) | {h.vira} | {c for r in h.rounds for _, c in r.plays}
            unseen = [c for c in Deck.full() if c not in visible and c not in h.hands[opp]]
            saved = list(h.hands[opp])
            h.hands[opp] = unseen[:len(saved)]
            try:
                np.testing.assert_array_equal(before, observe(g, me, version))
            finally:
                h.hands[opp] = saved
            checked += 1
    assert checked > 1000
