"""Read-only endpoints over ``runs/<id>/`` written by :mod:`agent.train`.

    GET /runs                        list runs (newest first)
    GET /runs/{id}                   run.json + config.json
    GET /runs/{id}/metrics?since=N   metrics.jsonl lines from index N
    GET /runs/{id}/checkpoints       best/final and every snapshot
    GET /runs/{id}/matrix            checkpoints x opponents win rates
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from agent.train import RUNS_DIR

from .players import list_run_checkpoints

router = APIRouter(prefix="/runs", tags=["runs"])

_RUN_ID = re.compile(r"^[A-Za-z0-9_.\-]+$")
EVALS_DIR = "evals"


def run_dir(run_id: str) -> Path:
    if not _RUN_ID.match(run_id) or run_id in (".", ".."):
        raise HTTPException(400, "Invalid run id.")
    d = RUNS_DIR / run_id
    if not (d / "run.json").is_file():
        raise HTTPException(404, f"Unknown run: {run_id}")
    return d


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def read_metrics(d: Path) -> List[Dict[str, Any]]:
    path = d / "metrics.jsonl"
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass  # a line being written right now
    return out


def run_summary(d: Path) -> Dict[str, Any]:
    info = _read_json(d / "run.json")
    conf = _read_json(d / "config.json")
    cfg = conf.get("config", {})
    metrics = read_metrics(d)
    last = metrics[-1] if metrics else None
    started, ended = info.get("started_at"), info.get("ended_at")
    ref = ended or info.get("updated_at") or started
    return {
        **info,
        "duration_s": (ref - started) if started and ref else None,
        "config": {
            "obs_version": cfg.get("obs_version"),
            "total_timesteps": cfg.get("total_timesteps"),
            "seed": conf.get("seed"),
            "league": cfg.get("league"),
            "eval_games": cfg.get("eval_games"),
            "eval_opponents": cfg.get("eval_opponents"),
            "hyperparams": cfg.get("hyperparams"),
        },
        "git": conf.get("git"),
        "n_metrics": len(metrics),
        "last_eval": None if last is None else {
            "timestep": last["timestep"], "eval_score": last.get("eval_score"),
            "win_rates": {k: v["win_rate"] for k, v in last.get("eval", {}).items()},
        },
    }


@router.get("")
def list_runs():
    if not RUNS_DIR.is_dir():
        return []
    return [run_summary(d) for d in sorted(RUNS_DIR.iterdir(), reverse=True)
            if (d / "run.json").is_file()]


@router.get("/{run_id}")
def get_run(run_id: str):
    d = run_dir(run_id)
    return {"run": run_summary(d), "config": _read_json(d / "config.json")}


@router.get("/{run_id}/metrics")
def get_metrics(run_id: str, since: int = 0):
    items = read_metrics(run_dir(run_id))
    return {"items": items[max(0, since):], "next": len(items)}


@router.get("/{run_id}/checkpoints")
def get_checkpoints(run_id: str):
    run_dir(run_id)
    return list_run_checkpoints(run_id)


def save_extra_eval(run_id: str, job: Dict[str, Any]) -> None:
    """Store an arena job result so it shows up in the run's matrix."""
    d = run_dir(run_id) / EVALS_DIR
    d.mkdir(exist_ok=True)
    (d / f"{job['id']}.json").write_text(json.dumps(job), encoding="utf-8")


@router.get("/{run_id}/matrix")
def get_matrix(run_id: str):
    """Win rate of every evaluated checkpoint against every opponent.

    Combines the evaluations made during training with arena jobs launched
    for this run. "prev" is left out: its opponent changes on every row.
    """
    d = run_dir(run_id)
    rows: Dict[str, Dict[str, Any]] = {}
    opponents: List[str] = []

    def cell(ckpt: str, timestep, opp: str, res: Dict[str, Any]) -> None:
        row = rows.setdefault(ckpt, {"checkpoint": ckpt, "timestep": timestep, "results": {}})
        row["results"][opp] = {"win_rate": res["win_rate"], "ci95": res["ci95"],
                               "games": res["games"]}
        if opp not in opponents:
            opponents.append(opp)

    for m in read_metrics(d):
        ckpt = f"ppo:runs/{run_id}/{m['checkpoint']}"
        for opp, res in m.get("eval", {}).items():
            if opp != "prev":
                cell(ckpt, m["timestep"], opp, res)
    for path in sorted((d / EVALS_DIR).glob("*.json")) if (d / EVALS_DIR).is_dir() else []:
        job = _read_json(path)
        res = job.get("result")
        if not res:
            continue
        m = re.search(r"step_(\d+)\.zip$", job["a"])
        cell(job["a"], int(m.group(1)) if m else None, job["b"],
             {"win_rate": res["win_rate_a"], "ci95": res["ci95"], "games": res["games"]})
    ordered = sorted(rows.values(), key=lambda r: (r["timestep"] is None, r["timestep"] or 0))
    return {"opponents": opponents, "rows": ordered}
