"""League training with arena evaluation and per-run persistence.

From ``backend/``::

    python -m agent.train --name liga_v2 --steps 3000000

Each run lives in ``runs/<id>/``:
    config.json     TrainConfig, seed, git commit, library versions
    run.json        status, progress, best checkpoint (rewritten as it goes)
    metrics.jsonl   one line per evaluation (training stats + arena results)
    checkpoints/    step_<timesteps>.zip snapshots (also the league pool)
    latest.zip      most recent weights, used by the "latest" league opponent
    final.zip, best.zip
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import threading
import time
from collections import defaultdict
from concurrent.futures import Executor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from agent.config import TrainConfig, ppo_kwargs
from agent.league import CHECKPOINTS_DIR, LATEST_NAME, LeagueEnv, list_snapshots, save_atomic
from arena.players import as_spec
from arena.runner import make_executor, run_matchup
from arena.stats import wilson

BACKEND_DIR = Path(__file__).resolve().parent.parent
# Point at a persistent volume in production (e.g. Railway) to keep runs.
RUNS_DIR = Path(os.environ.get("TRUCO_RUNS_DIR", BACKEND_DIR / "runs"))

MetricFn = Callable[[Dict[str, Any]], None]


# --- Run directory helpers ----------------------------------------------------
def _git_commit() -> Dict[str, Any]:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BACKEND_DIR,
                             capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "."],
                                    cwd=BACKEND_DIR, capture_output=True, text=True,
                                    timeout=5).stdout.strip())
        return {"commit": sha or None, "dirty": dirty}
    except Exception:
        return {"commit": os.environ.get("RAILWAY_GIT_COMMIT_SHA"), "dirty": None}


def _versions() -> Dict[str, str]:
    import gymnasium
    import sb3_contrib
    import stable_baselines3
    import torch

    return {"python": platform.python_version(), "torch": torch.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "sb3_contrib": sb3_contrib.__version__, "gymnasium": gymnasium.__version__,
            "numpy": np.__version__}


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_name(f"_tmp_{path.name}")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def new_run_dir(name: str, runs_dir: Path) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:40] or "run"
    run_dir = runs_dir / f"{stamp}-{safe}"
    (run_dir / CHECKPOINTS_DIR).mkdir(parents=True, exist_ok=False)
    return run_dir


# --- Callback -----------------------------------------------------------------
class LeagueCallback(BaseCallback):
    """Snapshots, league refresh, arena evaluation and metrics persistence."""

    def __init__(self, cfg: TrainConfig, run_dir: Path, run_info: Dict[str, Any],
                 stop_flag: Optional[threading.Event] = None,
                 on_metric: Optional[MetricFn] = None,
                 executor: Optional[Executor] = None) -> None:
        super().__init__(verbose=0)
        self.cfg = cfg
        self.run_dir = run_dir
        self.run_info = run_info
        self.stop_flag = stop_flag
        self.on_metric = on_metric
        self.executor = executor
        self.metrics_path = run_dir / "metrics.jsonl"
        self._next_snapshot = cfg.snapshot_every
        self._next_latest = cfg.latest_refresh
        self._next_eval = cfg.eval_every
        self._episodes = 0
        self._window: Dict[str, List[int]] = defaultdict(list)  # kind -> wins
        self._ep_rewards: List[float] = []
        self._ep_stats = defaultdict(int)
        self._train_values: Dict[str, float] = {}
        self._t0 = time.time()
        self._last_eval_ckpt: Optional[Path] = None

    @property
    def episodes(self) -> int:
        return self._episodes

    # -- hooks --
    def _on_training_start(self) -> None:
        save_atomic(self.model, self.run_dir / LATEST_NAME)
        self._save_checkpoint()  # step 0 seeds the snapshot pool and "prev"
        self._last_eval_ckpt = self._ckpt_path()

    def _on_rollout_start(self) -> None:
        # Values recorded by the previous train() call (not yet dumped).
        vals = self.model.logger.name_to_value
        for k in ("entropy_loss", "approx_kl", "clip_fraction", "value_loss",
                  "policy_gradient_loss", "explained_variance"):
            if f"train/{k}" in vals:
                self._train_values[k] = float(vals[f"train/{k}"])

    def _on_step(self) -> bool:
        for info, done in zip(self.locals.get("infos", []), self.locals.get("dones", [])):
            if done and "match" in info:
                m = info["match"]
                self._episodes += 1
                self._window[m["opponent"]].append(int(m["won"]))
                self._ep_rewards.append(1.0 if m["won"] else -1.0)
                for k in ("hands", "truco_calls", "runs", "bet_responses"):
                    self._ep_stats[k] += m[k]

        t = self.num_timesteps
        if t >= self._next_latest:
            self._next_latest += self.cfg.latest_refresh
            save_atomic(self.model, self.run_dir / LATEST_NAME)
        if t >= self._next_snapshot:
            self._next_snapshot += self.cfg.snapshot_every
            self._save_checkpoint()
        if t >= self._next_eval:
            self._next_eval += self.cfg.eval_every
            self._evaluate()
        if self.stop_flag is not None and self.stop_flag.is_set():
            return False
        return True

    def _on_training_end(self) -> None:
        if self._last_eval_ckpt != self._ckpt_path() and self.num_timesteps > 0:
            self._evaluate()

    # -- helpers --
    def _ckpt_path(self) -> Path:
        return self.run_dir / CHECKPOINTS_DIR / f"step_{self.num_timesteps:09d}.zip"

    def _save_checkpoint(self) -> Path:
        path = self._ckpt_path()
        if not path.exists():
            save_atomic(self.model, path)
        return path

    def _train_block(self) -> Dict[str, Any]:
        by_opp = {}
        total_w = total_n = 0
        for kind, wins in sorted(self._window.items()):
            n, w = len(wins), sum(wins)
            total_w += w
            total_n += n
            by_opp[kind] = {"games": n, "win_rate": w / n if n else None}
        st = self._ep_stats
        block = {
            "episodes": total_n,
            "win_rate": total_w / total_n if total_n else None,
            "reward_mean": float(np.mean(self._ep_rewards)) if self._ep_rewards else None,
            "by_opponent": by_opp,
            "truco_rate": st["truco_calls"] / st["hands"] if st["hands"] else None,
            "run_rate": st["runs"] / st["bet_responses"] if st["bet_responses"] else None,
            "entropy": -self._train_values["entropy_loss"] if "entropy_loss" in self._train_values else None,
            **{k: v for k, v in self._train_values.items() if k != "entropy_loss"},
        }
        self._window.clear()
        self._ep_rewards.clear()
        self._ep_stats.clear()
        return block

    def _evaluate(self) -> None:
        ckpt = self._save_checkpoint()
        spec = f"ppo:{ckpt}"
        prev = self._last_eval_ckpt if self._last_eval_ckpt and self._last_eval_ckpt != ckpt else None
        results: Dict[str, Any] = {}
        for opp in self.cfg.eval_opponents:
            if opp == "prev":
                if prev is None:
                    continue
                opp_spec, label = f"ppo:{prev}", "prev"
            else:
                opp_spec, label = as_spec(opp), opp
            try:
                s = run_matchup(spec, opp_spec, games=self.cfg.eval_games,
                                seed=self.cfg.eval_seed, workers=self.cfg.eval_workers,
                                executor=self.executor)
            except FileNotFoundError:
                continue  # e.g. reference checkpoint not deployed
            results[label] = {
                "opponent": opp_spec if label == "prev" else opp,
                "games": s["games"], "wins": s["a_wins"], "win_rate": s["win_rate_a"],
                "ci95": s["ci95"], "reward_mean": 2 * s["win_rate_a"] - 1,
                "points_per_hand": s["a_stats"]["points_per_hand"],
                "opp_points_per_hand": s["b_stats"]["points_per_hand"],
                "truco_rate": s["a_stats"]["truco_rate"],
                "run_rate": s["a_stats"]["run_rate"],
                "fallbacks": s["a_stats"]["fallbacks"],
            }
        self._last_eval_ckpt = ckpt

        fixed = [r for k, r in results.items() if k != "prev"]
        wins = sum(r["wins"] for r in fixed)
        games = sum(r["games"] for r in fixed)
        score = wins / games if games else None
        metric = {
            "timestep": int(self.num_timesteps),
            "episode": self._episodes,
            "elapsed_s": round(time.time() - self._t0, 1),
            "checkpoint": str(ckpt.relative_to(self.run_dir)).replace("\\", "/"),
            "train": self._train_block(),
            "eval": results,
            "eval_score": score,
            "eval_score_ci95": list(wilson(wins, games)) if games else None,
        }
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metric) + "\n")

        info = self.run_info
        info["timesteps"] = int(self.num_timesteps)
        info["updated_at"] = time.time()
        best = info.get("best")
        if score is not None and (best is None or score > best["score"]):
            shutil.copyfile(ckpt, self.run_dir / "best.zip")
            info["best"] = {"checkpoint": metric["checkpoint"], "score": score,
                            "timestep": metric["timestep"]}
        _write_json(self.run_dir / "run.json", info)
        if self.on_metric is not None:
            self.on_metric(metric)


# --- Entry point --------------------------------------------------------------
def _make_env_fn(cfg: TrainConfig, run_dir: Path, rank: int):
    def _init():
        return LeagueEnv(seed=cfg.seed * 1000 + rank, obs_version=cfg.obs_version,
                         league=cfg.league, run_dir=str(run_dir),
                         max_snapshots=cfg.max_snapshots_in_pool)
    return _init


def make_vec_env(cfg: TrainConfig, run_dir: Path) -> VecEnv:
    fns = [_make_env_fn(cfg, run_dir, i) for i in range(int(cfg.hyperparams["n_envs"]))]
    if cfg.vec_env == "subproc" and len(fns) > 1:
        return SubprocVecEnv(fns, start_method="spawn")
    return DummyVecEnv(fns)


def train_league(cfg: TrainConfig, runs_dir: Optional[Path] = None,
                 stop_flag: Optional[threading.Event] = None,
                 on_metric: Optional[MetricFn] = None,
                 run_dir: Optional[Path] = None,
                 on_start: Optional[Callable[["LeagueCallback"], None]] = None) -> Path:
    """Run a full league training; returns the run directory.

    ``on_start`` receives the callback before learning starts, so a caller
    (the API) can read live progress from it.
    """
    run_dir = run_dir or new_run_dir(cfg.name, runs_dir or RUNS_DIR)
    started = time.time()
    _write_json(run_dir / "config.json", {
        "config": cfg.to_dict(), "seed": cfg.seed, "git": _git_commit(),
        "versions": _versions(), "started_at": started,
    })
    run_info: Dict[str, Any] = {
        "id": run_dir.name, "name": cfg.name, "status": "running",
        "started_at": started, "ended_at": None, "timesteps": 0,
        "total_timesteps": cfg.total_timesteps, "obs_version": cfg.obs_version,
        "best": None, "error": None,
    }
    _write_json(run_dir / "run.json", run_info)

    env = make_vec_env(cfg, run_dir)
    # Evaluations always run in worker processes, even with a single worker,
    # so an in-server training thread never plays arena matches itself.
    executor = make_executor(max(1, cfg.eval_workers))
    try:
        kwargs = ppo_kwargs(cfg.hyperparams)
        if cfg.init_from:
            model = MaskablePPO.load(cfg.init_from, env=env, device="cpu")
        else:
            model = MaskablePPO("MlpPolicy", env, seed=cfg.seed, verbose=0,
                                device="cpu", **kwargs)
        cb = LeagueCallback(cfg, run_dir, run_info, stop_flag, on_metric, executor)
        if on_start is not None:
            on_start(cb)
        model.learn(total_timesteps=cfg.total_timesteps, callback=cb, use_masking=True)
        save_atomic(model, run_dir / "final.zip")
        stopped = stop_flag is not None and stop_flag.is_set()
        run_info["status"] = "stopped" if stopped else "finished"
    except BaseException as e:
        run_info["status"] = "failed"
        run_info["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        run_info["ended_at"] = time.time()
        _write_json(run_dir / "run.json", run_info)
        env.close()
        executor.shutdown()
    return run_dir


def _parse_league(text: str) -> Dict[str, float]:
    out = {}
    for part in text.split(","):
        k, _, v = part.partition("=")
        out[k.strip()] = float(v)
    return out


def main(argv=None) -> int:
    d = TrainConfig()
    p = argparse.ArgumentParser(description="League training with arena evaluation.")
    p.add_argument("--name", default=d.name)
    p.add_argument("--steps", type=int, default=d.total_timesteps)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--obs", default=d.obs_version, choices=["v1", "v2"])
    p.add_argument("--league", default=None,
                   help="weights, e.g. random=0.1,rule=0.3,snapshot=0.3,latest=0.3")
    p.add_argument("--snapshot-every", type=int, default=d.snapshot_every)
    p.add_argument("--eval-every", type=int, default=d.eval_every)
    p.add_argument("--eval-games", type=int, default=d.eval_games)
    p.add_argument("--eval-opponents", nargs="*", default=d.eval_opponents)
    p.add_argument("--eval-workers", type=int, default=d.eval_workers)
    p.add_argument("--n-envs", type=int, default=None)
    p.add_argument("--vec-env", default=d.vec_env, choices=["subproc", "dummy"])
    p.add_argument("--init-from", default="")
    p.add_argument("--hp", action="append", default=[], metavar="KEY=VALUE",
                   help="override a HYPERPARAMS entry, e.g. --hp ent_coef=0.01 (repeatable)")
    args = p.parse_args(argv)

    cfg = TrainConfig(
        name=args.name, total_timesteps=args.steps, seed=args.seed,
        obs_version=args.obs, snapshot_every=args.snapshot_every,
        eval_every=args.eval_every, eval_games=args.eval_games,
        eval_opponents=args.eval_opponents, eval_workers=args.eval_workers,
        vec_env=args.vec_env, init_from=args.init_from,
    )
    if args.league:
        cfg.league = _parse_league(args.league)
    if args.n_envs:
        cfg.hyperparams["n_envs"] = args.n_envs
    for item in args.hp:
        key, _, value = item.partition("=")
        if key not in cfg.hyperparams:
            p.error(f"unknown hyperparameter: {key}")
        cfg.hyperparams[key] = json.loads(value)

    def report(m: Dict[str, Any]) -> None:
        ev = "  ".join(f"{k}={v['win_rate']:.1%}" for k, v in m["eval"].items())
        tr = m["train"]
        wr = f"{tr['win_rate']:.1%}" if tr["win_rate"] is not None else "-"
        print(f"[{m['elapsed_s']:>7.0f}s] t={m['timestep']:>9,}  treino={wr}  {ev}", flush=True)

    run_dir = train_league(cfg, on_metric=report)
    print(f"-> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
