"""API: live /game/* endpoints, runs, arena jobs and training."""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

import agent.train
import api.arena_jobs
import api.players
import api.routes
import api.runs
from main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def _play_out(client, state):
    gid = state["game_id"]
    for _ in range(3000):
        if state["terminated"]:
            return state
        assert "p1_hand" not in state and "cheat_active" not in state
        if state["hand_ending"]:
            state = client.post("/game/next-hand", json={"game_id": gid}).json()
            continue
        r = client.post("/game/action", json={"game_id": gid, "action": state["legal_actions"][0]})
        assert r.status_code == 200, r.text
        state = r.json()
    raise AssertionError("game did not finish")


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_full_game_against_default_model(client):
    state = client.post("/game/new").json()
    assert state["legal_actions"]
    assert state["opponent"]["ref"] == "ppo:models/truco_ppo_1M.zip"
    state = _play_out(client, state)
    assert state["match_winner"] in (0, 1)
    assert max(state["scores"]["p0"], state["scores"]["p1"]) >= 12
    assert set(state["stats"]) == {"p0", "p1"}
    again = client.get("/game/state", params={"game_id": state["game_id"]}).json()
    assert again["game_id"] == state["game_id"]


def test_empty_body_keeps_the_default_opponent(client):
    state = client.post("/game/new", json={}).json()
    assert state["opponent"]["ref"] == "ppo:models/truco_ppo_1M.zip"


@pytest.mark.parametrize("opponent", ["facil", "medio", "rule", "ppo:models/truco_ppo_1M.zip"])
def test_game_against_selected_opponent(client, opponent):
    state = client.post("/game/new", json={"opponent": opponent}).json()
    state = _play_out(client, state)
    assert state["terminated"]


def test_human_actions_are_counted(client):
    state = client.post("/game/new", json={"opponent": "rule"}).json()
    while 3 not in state["legal_actions"]:
        if state["hand_ending"]:
            state = client.post("/game/next-hand", json={"game_id": state["game_id"]}).json()
        else:
            state = client.post("/game/action", json={"game_id": state["game_id"],
                                                      "action": state["legal_actions"][0]}).json()
    state = client.post("/game/action", json={"game_id": state["game_id"], "action": 3}).json()
    assert state["stats"]["p0"]["truco_calls"] == 1


@pytest.mark.parametrize("bad", ["py:os:system", "ppo:../main.py", "ppo:models/../x.zip",
                                 "ppo:C:/x.zip", "ppo:models/missing.zip", "nope"])
def test_unsafe_or_unknown_opponents_are_rejected(client, bad):
    assert client.post("/game/new", json={"opponent": bad}).status_code in (400, 404)


def test_illegal_action_rejected(client):
    state = client.post("/game/new").json()
    illegal = next(a for a in range(7) if a not in state["legal_actions"])
    r = client.post("/game/action", json={"game_id": state["game_id"], "action": illegal})
    assert r.status_code == 400


def test_unknown_game(client):
    assert client.get("/game/state", params={"game_id": "nope"}).status_code == 404


# --- runs ---------------------------------------------------------------------
@pytest.fixture()
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(api.runs, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(api.players, "RUNS_DIR", tmp_path)
    return tmp_path


def _fake_run(root, run_id="20260101-000000-x"):
    d = root / run_id
    (d / "checkpoints").mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({"id": run_id, "name": "x", "status": "finished",
                                            "started_at": 1.0, "ended_at": 11.0,
                                            "timesteps": 20, "best": None}))
    (d / "config.json").write_text(json.dumps({"config": {"obs_version": "v2"}, "seed": 0}))
    lines = []
    for t in (0, 10):
        ckpt = f"checkpoints/step_{t:09d}.zip"
        (d / ckpt).write_bytes(b"")
        lines.append({"timestep": t, "episode": t, "checkpoint": ckpt, "train": {},
                      "eval": {"random": {"win_rate": 0.5 + t / 100, "ci95": [0.4, 0.6], "games": 100},
                               "prev": {"win_rate": 0.5, "ci95": [0.4, 0.6], "games": 100}},
                      "eval_score": 0.5})
    (d / "metrics.jsonl").write_text("\n".join(json.dumps(l) for l in lines) + "\n")
    return run_id


def test_runs_endpoints(client, runs_dir):
    rid = _fake_run(runs_dir)
    runs = client.get("/runs").json()
    assert [r["id"] for r in runs] == [rid]
    assert runs[0]["duration_s"] == 10.0 and runs[0]["n_metrics"] == 2
    assert client.get(f"/runs/{rid}").json()["config"]["seed"] == 0
    m = client.get(f"/runs/{rid}/metrics", params={"since": 1}).json()
    assert m["next"] == 2 and len(m["items"]) == 1
    ckpts = client.get(f"/runs/{rid}/checkpoints").json()
    assert [c["timestep"] for c in ckpts] == [0, 10]
    matrix = client.get(f"/runs/{rid}/matrix").json()
    assert matrix["opponents"] == ["random"]
    assert [r["timestep"] for r in matrix["rows"]] == [0, 10]
    assert client.get("/runs/nope").status_code == 404
    assert client.get("/runs/..%2F..").status_code in (400, 404)


# --- arena jobs -----------------------------------------------------------------
def _wait(client, job_id, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = client.get(f"/arena/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.3)
    raise AssertionError("job did not finish")


@pytest.fixture()
def jobs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(api.arena_jobs, "JOBS_DIR", tmp_path / "jobs")
    return tmp_path / "jobs"


def test_arena_job_roundtrip(client, jobs_dir):
    players = client.get("/arena/players").json()
    assert {"levels", "builtin", "models", "runs"} <= set(players)
    job = client.post("/arena/jobs", json={"a": "rule", "b": "facil", "games": 40, "seed": 3}).json()
    assert job["status"] in ("queued", "running")
    job = _wait(client, job["id"])
    assert job["status"] == "done", job["error"]
    assert job["result"]["games"] == 40 and job["progress"]["done"] == 40
    assert job["b"] == "random"
    assert any(j["id"] == job["id"] for j in client.get("/arena/jobs").json())
    assert (jobs_dir / f"{job['id']}.json").is_file()


def test_arena_job_validation(client, jobs_dir):
    assert client.post("/arena/jobs", json={"a": "py:os:system", "b": "rule"}).status_code == 400
    assert client.post("/arena/jobs", json={"a": "rule", "b": "rule", "games": 3}).status_code == 400
    assert client.get("/arena/jobs/nope").status_code == 404


# --- training -------------------------------------------------------------------
def test_training_start_status_and_pause(client, runs_dir, monkeypatch):
    monkeypatch.setattr(agent.train, "RUNS_DIR", runs_dir)
    r = client.post("/train/start", json={"total_timesteps": 4096, "eval_every": 2048,
                                          "eval_games": 20, "name": "apitest"})
    assert r.status_code == 200
    assert client.post("/train/start", json={}).status_code == 409
    t0 = time.time()
    while time.time() - t0 < 180:
        st = client.get("/train/status").json()
        if st["latest"] is not None:
            break
        time.sleep(0.5)
    assert st["run_id"] and st["latest_metric"]["timestep"] >= 2048
    assert 0 <= st["latest_metric"]["win_rate"] <= 1
    with client.websocket_connect("/ws/metrics") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "metrics" and msg["full"]
    client.post("/train/pause")
    api.routes.TRAIN.thread.join(timeout=60)
    run = client.get(f"/runs/{st['run_id']}").json()["run"]
    assert run["status"] in ("stopped", "finished")
    assert run["n_metrics"] >= 1
    client.post("/train/reset")
