"""Re-record obs_v1_golden.npz from the current observe(). Run from backend/:

    python tests/data/make_obs_golden.py

Only do this when a rule change legitimately alters trajectories.
"""

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from truco.encoding import action_mask, encode_action  # noqa: E402
from truco.game import Player, TrucoGame  # noqa: E402
from truco.obs import observe  # noqa: E402

seeds, actions, obs0, obs1, m0, m1, offs = [], [], [], [], [], [], [0]
for seed in range(300):
    g = TrucoGame(seed=seed)
    rng = random.Random(1000 + seed)
    while g.state.winner is None:
        obs0.append(observe(g, Player.P0)); obs1.append(observe(g, Player.P1))
        m0.append(action_mask(g, Player.P0)); m1.append(action_mask(g, Player.P1))
        pa = rng.choice(g.legal_actions())
        actions.append(encode_action(pa))
        g.step(pa)
    seeds.append(seed)
    offs.append(len(actions))

np.savez_compressed(
    Path(__file__).with_name("obs_v1_golden.npz"),
    seeds=np.array(seeds), offsets=np.array(offs),
    actions=np.array(actions, dtype=np.int8),
    obs_p0=np.array(obs0), obs_p1=np.array(obs1),
    mask_p0=np.array(m0), mask_p1=np.array(m1),
)
print(f"{len(actions)} states from {len(seeds)} matches")
