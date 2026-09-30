"""observe() must reproduce the v1 layout the existing checkpoints expect.

``data/obs_v1_golden.npz`` was recorded from the original ``TrucoEnv._observe``
(P0) and ``_p1_view`` (P1) implementations over 300 random matches.
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


def test_observe_never_reveals_hidden_opponent_cards():
    for seed in range(50):
        g = TrucoGame(seed=seed)
        if g.state.open_hand_for is None:
            assert not observe(g, Player.P0)[21:].any()
            assert not observe(g, Player.P1)[21:].any()
