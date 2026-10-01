"""Arena: players, match bookkeeping and sanity controls."""

from __future__ import annotations

import json
import random

import pytest

from arena.match import play_match
from arena.players import (
    MODELS_DIR,
    PPOPlayer,
    RandomPlayer,
    RulePlayer,
    make_player,
)
from arena.runner import run_matchup
from arena.stats import wilson
from truco.encoding import decode_action, legal_action_ids
from truco.game import Player, TrucoGame

DEFAULT_CKPT = MODELS_DIR / "truco_ppo_1M.zip"


def test_wilson_interval():
    lo, hi = wilson(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3)
    assert hi == pytest.approx(0.5962, abs=1e-3)
    assert wilson(0, 10)[0] == 0.0
    assert wilson(10, 10)[1] == 1.0
    lo, hi = wilson(500, 1000)
    assert hi - lo < 0.065  # ~±3 points with 1000 games


def test_rule_player_only_returns_legal_actions():
    rule = RulePlayer()
    for seed in range(100):
        g = TrucoGame(seed=seed)
        rng = random.Random(seed)
        while g.state.winner is None:
            seat = g.state.hand.current_player
            a = rule.decide(g, seat)
            assert a in legal_action_ids(g, seat)
            # Mix in random moves so the rule sees varied states.
            if rng.random() < 0.5:
                a = rng.choice(legal_action_ids(g, seat))
            g.step(decode_action(a))


def test_match_bookkeeping_is_consistent():
    log = []
    r = play_match(RulePlayer(), RandomPlayer(), game_seed=2**62 + 5, a_seat=1, log=log)
    assert r.a_seat == 1
    assert max(r.scores.values()) >= 12
    assert r.stats["a"].points == r.scores["a"]
    assert r.stats["b"].points == r.scores["b"]
    assert r.a_won == (r.scores["a"] > r.scores["b"])
    assert r.stats["a"].decisions + r.stats["b"].decisions == len(log)
    assert r.stats["a"].fallbacks == r.stats["b"].fallbacks == 0
    assert all("hand_result" in e for e in log)
    assert {e["hand"] for e in log} == set(range(r.hands))
    json.dumps(log)  # serializable


def test_illegal_and_crashing_players_fall_back():
    r = play_match(make_player("py:helpers:AlwaysRunPlayer"), RandomPlayer(),
                   game_seed=2**62 + 1, a_seat=0)
    st = r.stats["a"]
    assert st.illegal > 0 and st.fallbacks == st.illegal and st.errors == 0
    r = play_match(make_player("py:helpers:CrashingPlayer"), RandomPlayer(),
                   game_seed=2**62 + 1, a_seat=0)
    st = r.stats["a"]
    assert st.errors == st.decisions == st.fallbacks


def test_unknown_spec_is_rejected():
    with pytest.raises(ValueError):
        make_player("nope")
    with pytest.raises(ValueError):
        make_player("ppo")


def test_random_vs_random_is_about_fifty_percent():
    s = run_matchup("random", "random", games=1000, seed=7, workers=1)
    assert s["games"] == 1000
    assert abs(s["win_rate_a"] - 0.5) < 0.05
    assert s["ci95"][0] < 0.5 < s["ci95"][1]
    assert s["by_seat"]["a_as_p0"]["games"] == s["by_seat"]["a_as_p1"]["games"] == 500


def test_rule_vs_itself_is_exactly_fifty_percent():
    # Deterministic and seat-swapped: each deal is the same game mirrored.
    s = run_matchup("rule", "rule", games=200, seed=3, workers=1)
    assert s["win_rate_a"] == 0.5


def test_rule_beats_random():
    s = run_matchup("rule", "random", games=400, seed=11, workers=1)
    assert s["ci95"][0] > 0.5


def test_results_do_not_depend_on_worker_count(tmp_path):
    serial = run_matchup("rule", "random", games=40, seed=5, workers=1,
                         log_path=tmp_path / "a.jsonl", chunk_size=7)
    parallel = run_matchup("rule", "random", games=40, seed=5, workers=2,
                           log_path=tmp_path / "b.jsonl", chunk_size=7)
    for k in ("a_wins", "hands", "a_stats", "b_stats", "by_seat"):
        assert serial[k] == parallel[k]
    assert (tmp_path / "a.jsonl").read_text() == (tmp_path / "b.jsonl").read_text()


@pytest.mark.skipif(not DEFAULT_CKPT.exists(), reason="default checkpoint missing")
def test_ppo_vs_itself_is_fifty_percent():
    s = run_matchup(f"ppo:{DEFAULT_CKPT}", f"ppo:{DEFAULT_CKPT}", games=200, seed=9, workers=1)
    assert s["win_rate_a"] == 0.5
    assert s["a_stats"]["fallbacks"] == s["b_stats"]["fallbacks"] == 0


@pytest.mark.skipif(not DEFAULT_CKPT.exists(), reason="default checkpoint missing")
def test_ppo_player_detects_v1_and_resolves_names():
    p = PPOPlayer("truco_ppo_1M")
    assert p.obs_version == "v1"
    g = TrucoGame(seed=1)
    assert p.decide(g, Player.P0) in legal_action_ids(g, Player.P0)


def test_windows_paths_are_not_mistaken_for_specs():
    from arena.players import as_spec, is_spec

    assert not is_spec(r"E:\models\x.zip")
    assert as_spec(r"E:\models\x.zip") == r"ppo:E:\models\x.zip"
    assert as_spec("rule") == "rule"
    assert as_spec("py:m:C") == "py:m:C"
