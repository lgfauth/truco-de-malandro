"""Arena CLI. From ``backend/``::

    python -m arena.run --a ppo:models/truco_ppo_1M.zip --b random --games 1000 --seed 123

Writes ``summary.json`` and ``decisions.jsonl`` under ``arena_results/``
(or ``--out``) and prints a short report.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from .runner import run_matchup

RESULTS_DIR = Path(os.environ.get(
    "TRUCO_ARENA_DIR", Path(__file__).resolve().parent.parent / "arena_results"))


def _slug(spec: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(spec.split(":", 1)[-1]).stem or spec)[:40]


def format_report(s: dict) -> str:
    a, b = s["a_stats"], s["b_stats"]
    lo, hi = s["ci95"]
    lines = [
        f"{s['a']}  vs  {s['b']}   ({s['games']} partidas, seed={s['seed']}, {s['elapsed_s']}s)",
        f"  vitórias A: {s['a_wins']}/{s['games']} = {s['win_rate_a']:.1%}   IC95 [{lo:.1%}, {hi:.1%}]",
        f"  A como P0: {s['by_seat']['a_as_p0']['win_rate']:.1%}   A como P1: {s['by_seat']['a_as_p1']['win_rate']:.1%}",
        f"  mãos: {s['hands']} ({s['mean_hands_per_game']:.1f}/partida, {s['drawn_hands']} empatadas)",
        f"  {'':14}{'A':>10}{'B':>10}",
    ]
    for key, label, fmt in [
        ("points_per_hand", "pontos/mão", "{:.3f}"),
        ("truco_rate", "truco/mão", "{:.3f}"),
        ("run_rate", "corrida/aposta", "{:.3f}"),
        ("illegal", "ilegais", "{}"),
        ("fallbacks", "fallbacks", "{}"),
        ("errors", "erros", "{}"),
    ]:
        lines.append(f"  {label:14}{fmt.format(a[key]):>10}{fmt.format(b[key]):>10}")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run an arena matchup.")
    p.add_argument("--a", required=True, help="player spec, e.g. ppo:models/x.zip")
    p.add_argument("--b", required=True, help="player spec, e.g. random | rule")
    p.add_argument("--games", type=int, default=1000, help="even number of matches")
    p.add_argument("--seed", type=int, default=123, help="evaluation deal seed")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--out", type=Path, default=None, help="output directory")
    p.add_argument("--no-log", action="store_true", help="skip decisions.jsonl")
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    out = args.out or RESULTS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{_slug(args.a)}-vs-{_slug(args.b)}"
    out.mkdir(parents=True, exist_ok=True)

    def progress(done: int, total: int) -> None:
        print(f"\r  {done}/{total}", end="", file=sys.stderr, flush=True)

    summary = run_matchup(
        args.a, args.b, games=args.games, seed=args.seed, workers=args.workers,
        log_path=None if args.no_log else out / "decisions.jsonl", progress=progress,
    )
    print(file=sys.stderr)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(format_report(summary))
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
