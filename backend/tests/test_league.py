"""League env, opponent pool and the run pipeline."""

from __future__ import annotations

import json
import random

import numpy as np
from sb3_contrib import MaskablePPO

from agent.config import HYPERPARAMS, TrainConfig, ppo_kwargs
from agent.league import LATEST_NAME, LeagueEnv, OpponentPool
from agent.train import train_league
from truco.obs import OBS_V2_DIM


def _play_episode(env, rng):
    obs, info = env.reset()
    done, reward = False, 0.0
    while not done:
        legal = np.flatnonzero(env.action_masks())
        assert len(legal) and set(legal.tolist()) == set(info["legal_actions"])
        obs, reward, terminated, truncated, info = env.step(int(rng.choice(legal)))
        done = terminated or truncated
    return reward, info


def test_league_env_plays_both_seats_and_reports_matches():
    env = LeagueEnv(seed=0, obs_version="v2", league={"random": 1, "rule": 1})
    rng = np.random.default_rng(0)
    seats, kinds = set(), set()
    for _ in range(60):
        reward, info = _play_episode(env, rng)
        m = info["match"]
        assert reward in (1.0, -1.0) and m["won"] == (reward > 0)
        assert m["hands"] >= 1
        seats.add(m["seat"])
        kinds.add(m["opponent"])
    assert seats == {0, 1}
    assert kinds == {"random", "rule"}
    assert env.observation_space.shape == (OBS_V2_DIM,)


def test_pool_falls_back_when_snapshots_are_missing(tmp_path):
    pool = OpponentPool({"snapshot": 1, "latest": 1}, tmp_path)
    kind, _ = pool.sample(random.Random(0))
    assert kind == "random"


def test_pool_uses_snapshots_and_latest(tmp_path):
    env = LeagueEnv(seed=0, obs_version="v2")
    model = MaskablePPO("MlpPolicy", env, n_steps=64, batch_size=64, verbose=0, device="cpu")
    (tmp_path / "checkpoints").mkdir()
    model.save(str(tmp_path / "checkpoints" / "step_000000000.zip"))
    model.save(str(tmp_path / LATEST_NAME))
    pool = OpponentPool({"snapshot": 1, "latest": 1}, tmp_path)
    rng = random.Random(1)
    kinds = {pool.sample(rng)[0] for _ in range(20)}
    assert kinds == {"snapshot", "latest"}
    league_env = LeagueEnv(seed=1, obs_version="v2", league={"snapshot": 1},
                           run_dir=tmp_path)
    reward, info = _play_episode(league_env, np.random.default_rng(1))
    assert info["match"]["opponent"] == "snapshot"


def test_ppo_kwargs_split():
    kw = ppo_kwargs(HYPERPARAMS)
    assert "n_envs" not in kw and kw["policy_kwargs"]["net_arch"] == HYPERPARAMS["net_arch"]
    assert HYPERPARAMS["n_envs"] * HYPERPARAMS["n_steps"] == 2048


def test_train_league_writes_a_complete_run(tmp_path):
    cfg = TrainConfig(name="t", total_timesteps=512, seed=1, obs_version="v2",
                      snapshot_every=256, latest_refresh=128, eval_every=256,
                      eval_games=20, eval_workers=1, vec_env="dummy",
                      eval_opponents=["random", "rule", "prev"])
    cfg.hyperparams.update({"n_envs": 2, "n_steps": 64, "batch_size": 64,
                            "n_epochs": 2, "net_arch": [32, 32]})
    seen = []
    run_dir = train_league(cfg, runs_dir=tmp_path, on_metric=seen.append)
    run = json.loads((run_dir / "run.json").read_text())
    assert run["status"] == "finished" and run["best"] is not None
    conf = json.loads((run_dir / "config.json").read_text())
    assert conf["config"]["hyperparams"]["n_steps"] == 64 and "versions" in conf
    lines = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert len(lines) == len(seen) >= 2
    assert {"random", "rule"} <= set(lines[0]["eval"])
    assert "prev" in lines[1]["eval"]
    ev = lines[0]["eval"]["random"]
    assert ev["games"] == 20 and len(ev["ci95"]) == 2
    for name in ("final.zip", "best.zip", LATEST_NAME):
        assert (run_dir / name).exists()
    assert len(list((run_dir / "checkpoints").glob("step_*.zip"))) >= 3
