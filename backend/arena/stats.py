"""Confidence intervals and matchup summaries."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Dict, Iterable, Tuple

from .match import MatchResult, SideStats


def wilson(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default)."""
    if n <= 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _side_summary(st: SideStats, hands: int) -> Dict[str, float]:
    d = asdict(st)
    d["points_per_hand"] = st.points / hands if hands else 0.0
    d["truco_rate"] = st.truco_calls / hands if hands else 0.0
    d["run_rate"] = st.runs / st.bet_responses if st.bet_responses else 0.0
    return d


def summarize(results: Iterable[MatchResult]) -> Dict:
    results = list(results)
    n = len(results)
    wins = sum(r.a_won for r in results)
    hands = sum(r.hands for r in results)
    a, b = SideStats(), SideStats()
    for r in results:
        a.add(r.stats["a"])
        b.add(r.stats["b"])
    lo, hi = wilson(wins, n)
    by_seat = {}
    for seat in (0, 1):
        rs = [r for r in results if r.a_seat == seat]
        w = sum(r.a_won for r in rs)
        by_seat[f"a_as_p{seat}"] = {
            "games": len(rs), "a_wins": w,
            "win_rate": w / len(rs) if rs else 0.0,
        }
    return {
        "games": n,
        "a_wins": wins,
        "win_rate_a": wins / n if n else 0.0,
        "ci95": [lo, hi],
        "hands": hands,
        "drawn_hands": sum(r.drawn_hands for r in results),
        "mean_hands_per_game": hands / n if n else 0.0,
        "a_stats": _side_summary(a, hands),
        "b_stats": _side_summary(b, hands),
        "by_seat": by_seat,
    }
