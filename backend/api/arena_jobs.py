"""Arena jobs over HTTP: queue a matchup, poll its progress, read the result.

    GET  /arena/players        players the API accepts (levels, models, runs)
    POST /arena/jobs           {a, b, games, seed, run_id?} -> job
    GET  /arena/jobs           recent jobs
    GET  /arena/jobs/{id}      one job (status, progress, result)

Jobs run one at a time from a coordinator thread; the matches themselves are
played in a pool of worker processes (``TRUCO_ARENA_WORKERS``, default 2), so
the server process never plays arena games and ``/game/*`` stays responsive.
Results are written to ``<arena dir>/jobs/<id>.json`` and survive restarts.
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from arena.run import RESULTS_DIR
from arena.runner import make_executor, run_matchup

from .players import label_for, list_players, resolve_spec
from .runs import run_dir, save_extra_eval

router = APIRouter(prefix="/arena", tags=["arena"])

ARENA_WORKERS = max(1, int(os.environ.get("TRUCO_ARENA_WORKERS", "2")))
MAX_GAMES = 5000
JOBS_DIR = RESULTS_DIR / "jobs"


class JobManager:
    def __init__(self) -> None:
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._executor = None
        self._loaded = False

    # -- persistence --
    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not JOBS_DIR.is_dir():
            return
        for path in JOBS_DIR.glob("*.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if job.get("status") in ("queued", "running"):
                job["status"] = "failed"
                job["error"] = "Interrupted by a server restart."
            self.jobs.setdefault(job["id"], job)

    def _persist(self, job: Dict[str, Any]) -> None:
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = JOBS_DIR / f"_tmp_{job['id']}.json"
        tmp.write_text(json.dumps(job), encoding="utf-8")
        tmp.replace(JOBS_DIR / f"{job['id']}.json")

    # -- API --
    def submit(self, spec_a: str, ref_a: str, spec_b: str, ref_b: str,
               games: int, seed: int, run_id: Optional[str]) -> Dict[str, Any]:
        job = {
            "id": uuid.uuid4().hex[:12], "status": "queued",
            "a": ref_a, "b": ref_b, "a_label": label_for(ref_a), "b_label": label_for(ref_b),
            "games": games, "seed": seed, "run_id": run_id,
            "progress": {"done": 0, "total": games},
            "result": None, "error": None,
            "created_at": time.time(), "started_at": None, "finished_at": None,
        }
        with self.lock:
            self._load()
            self.jobs[job["id"]] = job
            job["_specs"] = (spec_a, spec_b)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._loop, daemon=True)
                self._thread.start()
        self._queue.put(job["id"])
        return public(job)

    def get(self, job_id: str) -> Dict[str, Any]:
        with self.lock:
            self._load()
            job = self.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job.")
        return public(job)

    def recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self.lock:
            self._load()
            jobs = sorted(self.jobs.values(), key=lambda j: j["created_at"], reverse=True)
        return [public(j) for j in jobs[:limit]]

    # -- worker --
    def _loop(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self.jobs[job_id]
            spec_a, spec_b = job["_specs"]
            job["status"] = "running"
            job["started_at"] = time.time()

            def progress(done: int, total: int) -> None:
                job["progress"] = {"done": done, "total": total}

            try:
                if self._executor is None:
                    self._executor = make_executor(ARENA_WORKERS)
                result = run_matchup(spec_a, spec_b, games=job["games"],
                                     seed=job["seed"], workers=ARENA_WORKERS,
                                     progress=progress, executor=self._executor)
                result["a"], result["b"] = job["a"], job["b"]
                job["result"], status = result, "done"
            except Exception as e:  # noqa: BLE001 — reported to the client
                job["error"], status = f"{type(e).__name__}: {e}", "failed"
            job["finished_at"] = time.time()
            # Persist before publishing the final status, so a client that
            # sees "done" can rely on the stored result.
            clean = {**public(job), "status": status}
            self._persist(clean)
            job["status"] = status
            if job["run_id"] and status == "done":
                try:
                    save_extra_eval(job["run_id"], clean)
                except HTTPException:
                    pass

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)


def public(job: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in job.items() if not k.startswith("_")}


JOBS = JobManager()
atexit.register(JOBS.shutdown)


class ArenaJobReq(BaseModel):
    a: str
    b: str
    games: int = Field(default=500, ge=2, le=MAX_GAMES)
    seed: int = 123
    run_id: Optional[str] = None


@router.get("/players")
def arena_players():
    return list_players()


@router.post("/jobs")
def create_job(req: ArenaJobReq):
    if req.games % 2:
        raise HTTPException(400, "games must be even (each deal is played from both seats).")
    spec_a, ref_a = resolve_spec(req.a)
    spec_b, ref_b = resolve_spec(req.b)
    if req.run_id:
        run_dir(req.run_id)
    return JOBS.submit(spec_a, ref_a, spec_b, ref_b, req.games, req.seed, req.run_id)


@router.get("/jobs")
def list_jobs(limit: int = 50):
    return JOBS.recent(limit)


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    return JOBS.get(job_id)
