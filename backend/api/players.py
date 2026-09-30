"""Which players the API may use, and how clients refer to them.

Clients never send file paths or ``py:`` specs (those could import arbitrary
code). They send one of:

    a level name          facil | medio | impossivel
    random | rule
    ppo:models/<file>.zip                 a deployed checkpoint
    ppo:runs/<run_id>/<file>.zip          a run's best/final/latest
    ppo:runs/<run_id>/checkpoints/<file>.zip

``resolve_spec`` validates the reference and returns the absolute arena spec.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import HTTPException

from agent.league import CHECKPOINTS_DIR
from agent.train import RUNS_DIR
from arena.players import MODELS_DIR

# Level name -> (spec, label). "impossivel" is today's default opponent.
LEVELS: Dict[str, Tuple[str, str]] = {
    "facil": ("random", "Fácil — joga ao acaso"),
    "medio": ("rule", "Médio — regras simples"),
    "impossivel": ("ppo:models/truco_ppo_1M.zip", "Impossível — PPO 1M"),
}
DEFAULT_OPPONENT = "impossivel"

_SAFE = re.compile(r"^[A-Za-z0-9_.\-]+$")


def _checkpoint_path(ref: str) -> Path:
    parts = ref.replace("\\", "/").split("/")
    if any(p in ("", ".", "..") or not _SAFE.match(p) for p in parts) or not ref.endswith(".zip"):
        raise HTTPException(400, f"Invalid checkpoint reference: {ref}")
    if parts[0] == "models" and len(parts) == 2:
        path = MODELS_DIR / parts[1]
    elif parts[0] == "runs" and len(parts) == 3:
        path = RUNS_DIR / parts[1] / parts[2]
    elif parts[0] == "runs" and len(parts) == 4 and parts[2] == CHECKPOINTS_DIR:
        path = RUNS_DIR / parts[1] / CHECKPOINTS_DIR / parts[3]
    else:
        raise HTTPException(400, f"Checkpoint must live under models/ or runs/: {ref}")
    if not path.is_file():
        raise HTTPException(404, f"Checkpoint not found: {ref}")
    return path


def resolve_spec(value: Optional[str]) -> Tuple[str, str]:
    """Client reference -> (absolute arena spec, public reference)."""
    value = (value or DEFAULT_OPPONENT).strip()
    if value in LEVELS:
        value = LEVELS[value][0]
    if value in ("random", "rule"):
        return value, value
    prefix, _, ref = value.partition(":")
    if prefix in ("ppo", "ppo-stoch") and ref:
        return f"{prefix}:{_checkpoint_path(ref)}", value
    raise HTTPException(400, f"Unknown player: {value}")


def label_for(ref: str) -> str:
    for name, (spec, label) in LEVELS.items():
        if ref in (name, spec):
            return label
    return ref


def _timestep(name: str) -> Optional[int]:
    m = re.match(r"step_(\d+)\.zip$", name)
    return int(m.group(1)) if m else None


def list_run_checkpoints(run_id: str) -> List[Dict]:
    run_dir = RUNS_DIR / run_id
    out = []
    for name in ("best.zip", "final.zip"):
        if (run_dir / name).is_file():
            out.append({"ref": f"ppo:runs/{run_id}/{name}", "name": name,
                        "timestep": None, "kind": name[:-4]})
    for path in sorted((run_dir / CHECKPOINTS_DIR).glob("step_*.zip")):
        out.append({"ref": f"ppo:runs/{run_id}/{CHECKPOINTS_DIR}/{path.name}",
                    "name": path.name, "timestep": _timestep(path.name),
                    "kind": "snapshot"})
    return out


def list_players() -> Dict:
    levels = [{"ref": name, "spec": spec, "label": label}
              for name, (spec, label) in LEVELS.items()]
    models = [{"ref": f"ppo:models/{p.name}", "label": p.stem}
              for p in sorted(MODELS_DIR.glob("*.zip"))]
    runs = []
    if RUNS_DIR.is_dir():
        for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
            info_path = run_dir / "run.json"
            if not info_path.is_file():
                continue
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            ckpts = [c for c in list_run_checkpoints(run_dir.name) if c["kind"] != "snapshot"]
            if ckpts:
                runs.append({"run_id": run_dir.name, "name": info.get("name"),
                             "checkpoints": ckpts})
    return {"levels": levels, "builtin": ["random", "rule"], "models": models,
            "runs": runs, "default": DEFAULT_OPPONENT}
