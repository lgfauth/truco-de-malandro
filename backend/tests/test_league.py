"""League env, opponent pool and the run pipeline."""

from __future__ import annotations

import json
import random

from pathlib import Path

import numpy as np
import pytest
from sb3_contrib import MaskablePPO

from agent.config import HYPERPARAMS, TrainConfig, ppo_kwargs
from agent.league import LATEST_NAME, LeagueEnv, OpponentPool
from agent.train import checkpoint_obs_version, source_snapshots, train_league
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


def test_pool_samples_inherited_snapshots(tmp_path):
    env = LeagueEnv(seed=0, obs_version="v2")
    model = MaskablePPO("MlpPolicy", env, n_steps=64, batch_size=64, verbose=0, device="cpu")
    model.save(str(tmp_path / "old.zip"))
    pool = OpponentPool({"snapshot": 1}, None,
                        extra_snapshots=[tmp_path / "old.zip", tmp_path / "missing.zip"])
    assert pool.extra_snapshots == [tmp_path / "old.zip"]
    assert pool.sample(random.Random(0))[0] == "snapshot"


def test_source_snapshots(tmp_path):
    run = tmp_path / "r"
    (run / "checkpoints").mkdir(parents=True)
    (run / "run.json").write_text("{}")
    for t in (0, 10, 20):
        (run / "checkpoints" / f"step_{t:09d}.zip").write_bytes(b"")
    names = lambda ps: [p.name for p in ps]
    assert names(source_snapshots(run / "best.zip", 2)) == ["step_000000010.zip", "step_000000020.zip"]
    assert names(source_snapshots(run / "checkpoints" / "step_000000010.zip", 5)) == [
        "step_000000000.zip", "step_000000010.zip"]
    assert source_snapshots(tmp_path / "model.zip", 5) == []


def test_ppo_kwargs_split():
    kw = ppo_kwargs(HYPERPARAMS)
    assert "n_envs" not in kw and kw["policy_kwargs"]["net_arch"] == HYPERPARAMS["net_arch"]
    assert HYPERPARAMS["n_envs"] * HYPERPARAMS["n_steps"] == 2048


def _small_cfg(**kw) -> TrainConfig:
    cfg = TrainConfig(name="t", total_timesteps=512, seed=1, obs_version="v2",
                      snapshot_every=256, latest_refresh=128, eval_every=256,
                      eval_games=20, eval_workers=1, vec_env="dummy",
                      eval_opponents=["random", "rule", "prev"], **kw)
    cfg.hyperparams.update({"n_envs": 2, "n_steps": 64, "batch_size": 64,
                            "n_epochs": 2, "net_arch": [32, 32]})
    return cfg


def test_train_league_writes_a_complete_run(tmp_path):
    cfg = _small_cfg()
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


def test_continuing_a_checkpoint(tmp_path):
    src_dir = train_league(_small_cfg(), runs_dir=tmp_path)
    src = src_dir / "best.zip"
    # obs_version and net_arch come from the checkpoint, not from this config.
    cfg = _small_cfg(init_from=str(src), init_overrides={"learning_rate": 1e-4})
    cfg.name, cfg.obs_version = "cont", "v1"
    cfg.hyperparams["net_arch"] = [64, 64]
    run_dir = train_league(cfg, runs_dir=tmp_path)
    assert checkpoint_obs_version(run_dir / "final.zip") == "v2"
    conf = json.loads((run_dir / "config.json").read_text())["config"]
    assert conf["obs_version"] == "v2" and conf["init_from"] == str(src)
    assert conf["hyperparams"]["learning_rate"] == 1e-4
    assert conf["hyperparams"]["net_arch"] == [32, 32]
    assert conf["extra_snapshots"] and all(
        Path(p).parent == src_dir / "checkpoints" for p in conf["extra_snapshots"])
    lines = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert "init" in lines[0]["eval"]
    fixed = [r for k, r in lines[0]["eval"].items() if k not in ("prev", "init")]
    assert lines[0]["eval_score"] == sum(r["wins"] for r in fixed) / sum(r["games"] for r in fixed)


def test_continuing_rejects_architecture_overrides(tmp_path):
    cfg = _small_cfg(init_from="x.zip", init_overrides={"net_arch": [8]})
    with pytest.raises(ValueError):
        train_league(cfg, runs_dir=tmp_path)
