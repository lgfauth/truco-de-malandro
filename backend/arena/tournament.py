"""Every checkpoint against the same opponents, as one table. From ``backend/``::

    python -m arena.tournament --games 500 --seed 123
    python -m arena.tournament --checkpoints models/a.zip runs/x/checkpoints/*.zip \\
        --opponents random rule ppo:truco_ppo_1M

Prints a markdown table (win rate of the checkpoint, 95% CI) and writes
``tournament.json`` / ``tournament.md`` to ``--out``.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from .players import as_spec, list_checkpoints
from .run import RESULTS_DIR
from .runner import run_matchup

DEFAULT_OPPONENTS = ["random", "rule"]


def run_tournament(checkpoints: List[str], opponents: List[str], games: int,
                   seed: int, workers: Optional[int] = None) -> Dict:
    rows = []
    for ckpt in checkpoints:
        spec = as_spec(ckpt)
        cells = {}
        for opp in opponents:
            s = run_matchup(spec, opp, games=games, seed=seed, workers=workers)
            cells[opp] = {"win_rate": s["win_rate_a"], "ci95": s["ci95"],
                          "points_per_hand": s["a_stats"]["points_per_hand"],
                          "fallbacks": s["a_stats"]["fallbacks"]}
            print(f"  {spec} vs {opp}: {s['win_rate_a']:.1%}", file=sys.stderr)
        rows.append({"player": spec, "results": cells})
    return {"games": games, "seed": seed, "opponents": opponents, "rows": rows}


def _label(spec: str) -> str:
    prefix, _, arg = spec.partition(":")
    if not (prefix.startswith("ppo") and arg):
        return spec
    path = Path(arg)
    if path.stem in ("best", "final", "latest") or path.parent.name == "checkpoints":
        run = path.parent.parent if path.parent.name == "checkpoints" else path.parent
        return f"{prefix}:{run.name}/{path.stem}"
    return f"{prefix}:{path.stem}"


def to_markdown(t: Dict) -> str:
    opps = t["opponents"]
    head = "| checkpoint | " + " | ".join(opps) + " |"
    sep = "|---|" + "---|" * len(opps)
    lines = [head, sep]
    for row in t["rows"]:
        cells = []
        for o in opps:
            c = row["results"][o]
            lo, hi = c["ci95"]
            cells.append(f"{c['win_rate']:.1%} [{lo:.0%}–{hi:.0%}]")
        lines.append(f"| {_label(row['player'])} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Checkpoints x opponents table.")
    p.add_argument("--checkpoints", nargs="*", default=None,
                   help="checkpoint paths/globs (default: models/*.zip)")
    p.add_argument("--opponents", nargs="*", default=DEFAULT_OPPONENTS)
    p.add_argument("--games", type=int, default=500)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if args.checkpoints:
        ckpts = [m for pat in args.checkpoints for m in (sorted(glob.glob(pat)) or [pat])]
    else:
        ckpts = [str(c) for c in list_checkpoints()]
    t = run_tournament(ckpts, args.opponents, args.games, args.seed, args.workers)
    md = to_markdown(t)
    out = args.out or RESULTS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-tournament"
    out.mkdir(parents=True, exist_ok=True)
    (out / "tournament.json").write_text(json.dumps(t, indent=2), encoding="utf-8")
    (out / "tournament.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
