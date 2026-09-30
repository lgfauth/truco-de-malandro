"""Smoke tests for the live /game/* endpoints with the default checkpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_full_game_against_default_model(client):
    state = client.post("/game/new").json()
    gid = state["game_id"]
    assert state["legal_actions"]
    for _ in range(3000):
        if state["terminated"]:
            break
        if state["hand_ending"]:
            state = client.post("/game/next-hand", json={"game_id": gid}).json()
            continue
        a = state["legal_actions"][0]
        r = client.post("/game/action", json={"game_id": gid, "action": a})
        assert r.status_code == 200, r.text
        state = r.json()
    assert state["terminated"]
    assert state["match_winner"] in (0, 1)
    assert max(state["scores"]["p0"], state["scores"]["p1"]) >= 12
    again = client.get("/game/state", params={"game_id": gid}).json()
    assert again["game_id"] == gid


def test_illegal_action_rejected(client):
    state = client.post("/game/new").json()
    illegal = next(a for a in range(7) if a not in state["legal_actions"])
    r = client.post("/game/action", json={"game_id": state["game_id"], "action": illegal})
    assert r.status_code == 400


def test_unknown_game(client):
    assert client.get("/game/state", params={"game_id": "nope"}).status_code == 404
