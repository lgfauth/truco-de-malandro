"""Run a matchup: every evaluation deal is played twice, once from each seat."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from truco.seeds import eval_game_seeds

from .match import MatchResult, play_match
from .players import ArenaPlayer, make_player
from .stats import summarize

ProgressFn = Callable[[int, int], None]  # (games_done, games_total)

# Per-process player cache so each worker loads a checkpoint only once.
_WORKER_PLAYERS: Dict[str, ArenaPlayer] = {}


def _worker_init() -> None:
    # One torch thread per worker: parallelism comes from processes.
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass


def _get_player(spec: str) -> ArenaPlayer:
    if spec not in _WORKER_PLAYERS:
        _WORKER_PLAYERS[spec] = make_player(spec)
    return _WORKER_PLAYERS[spec]


def _play_chunk(spec_a: str, spec_b: str, jobs: Sequence[Tuple[int, int]],
                with_log: bool) -> Tuple[List[MatchResult], List[dict]]:
    # A and B get separate instances even when the spec is the same.
    a = _get_player(spec_a)
    b = make_player(spec_b) if spec_a == spec_b else _get_player(spec_b)
    results, log = [], ([] if with_log else None)
    for game_seed, a_seat in jobs:
        results.append(play_match(a, b, game_seed, a_seat, log))
    return results, log or []


def _jobs(games: int, seed: int) -> List[Tuple[int, int]]:
    if games < 2 or games % 2:
        raise ValueError("games must be an even number >= 2 (each deal is played from both seats)")
    return [(s, seat) for s in eval_game_seeds(seed, games // 2) for seat in (0, 1)]


def run_matchup(spec_a: str, spec_b: str, games: int = 1000, seed: int = 123,
                workers: Optional[int] = None, log_path: Optional[Path] = None,
                progress: Optional[ProgressFn] = None,
                chunk_size: int = 50) -> Dict:
    """Play ``games`` matches of ``spec_a`` vs ``spec_b`` and summarize.

    Deals come from the evaluation seed space, so they never overlap with
    training. Results are identical for any ``workers`` value.
    """
    jobs = _jobs(games, seed)
    workers = workers if workers is not None else min(8, os.cpu_count() or 1)
    with_log = log_path is not None
    chunks = [jobs[i:i + chunk_size] for i in range(0, len(jobs), chunk_size)]
    by_chunk: Dict[int, Tuple[List[MatchResult], List[dict]]] = {}
    t0 = time.time()
    done = 0

    if workers <= 1:
        for i, chunk in enumerate(chunks):
            by_chunk[i] = _play_chunk(spec_a, spec_b, chunk, with_log)
            done += len(chunk)
            if progress:
                progress(done, len(jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
            futures = {pool.submit(_play_chunk, spec_a, spec_b, chunk, with_log): i
                       for i, chunk in enumerate(chunks)}
            for fut in as_completed(futures):
                i = futures[fut]
                by_chunk[i] = fut.result()
                done += len(chunks[i])
                if progress:
                    progress(done, len(jobs))

    results: List[MatchResult] = []
    for i in range(len(chunks)):
        results.extend(by_chunk[i][0])

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            for i in range(len(chunks)):
                for entry in by_chunk[i][1]:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    summary = summarize(results)
    summary.update({
        "a": spec_a, "b": spec_b, "seed": seed,
        "elapsed_s": round(time.time() - t0, 2),
    })
    return summary


def run_players(a: ArenaPlayer, b: ArenaPlayer, games: int, seed: int) -> Dict:
    """In-process matchup for already-built players (e.g. a model in training)."""
    results = [play_match(a, b, s, seat) for s, seat in _jobs(games, seed)]
    summary = summarize(results)
    summary.update({"a": a.name, "b": b.name, "seed": seed})
    return summary
