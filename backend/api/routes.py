"""FastAPI routes for the Truco RL project.

Endpoints:
    POST /train/start, /train/pause, /train/reset, GET /train/status
    POST /game/new, /game/action, /game/continue, /game/next-hand, GET /game/state
    WS   /ws/metrics
    plus /runs/* (api/runs.py) and /arena/* (api/arena_jobs.py)

Training runs :func:`agent.train.train_league` in a background thread and
persists everything under ``runs/<id>/``; its arena evaluations run in worker
processes. Game state lives in an in-memory dict (``GAMES``).
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from agent.config import TrainConfig
from agent.train import LeagueCallback, train_league
from arena.players import ArenaPlayer, make_player
from truco.encoding import (
    ACT_ACCEPT,
    ACT_CALL_TRUCO,
    ACT_RAISE,
    ACT_RUN,
    actions_equal,
    decode_action,
    encode_action,
    to_legal_action,
)
from truco.env import TrucoEnv
from truco.game import Player

from .arena_jobs import ARENA_WORKERS
from .arena_jobs import router as arena_router
from .players import label_for, resolve_spec
from .runs import router as runs_router
from .runs import run_dir

router = APIRouter()
router.include_router(runs_router)
router.include_router(arena_router)

DEFAULT_TIMESTEPS = 500_000


# --- Training state --------------------------------------------------------
class _TrainState:
    """Singleton holder for the background training job."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.callback: Optional[LeagueCallback] = None
        self.stop_flag = threading.Event()
        self.metrics: List[Dict[str, Any]] = []
        self.run_id: Optional[str] = None
        self.total_timesteps = 0
        self.error: Optional[str] = None
        self.paused = False

    def is_alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive()


TRAIN = _TrainState()


def _train_worker(cfg: TrainConfig) -> None:
    def on_start(cb: LeagueCallback) -> None:
        TRAIN.callback = cb
        TRAIN.run_id = cb.run_dir.name

    try:
        train_league(cfg, stop_flag=TRAIN.stop_flag,
                     on_metric=TRAIN.metrics.append, on_start=on_start)
    except Exception as e:  # noqa: BLE001 — surfaced via /train/status
        TRAIN.error = f"{type(e).__name__}: {e}"


# --- Game state ------------------------------------------------------------
_STAT_KEYS = {ACT_CALL_TRUCO: "truco_calls", ACT_RAISE: "raises",
              ACT_ACCEPT: "accepts", ACT_RUN: "runs"}


class GameSession:
    """A human (P0) against an arena player (P1)."""

    def __init__(self, opponent: Optional[str] = None, seed: Optional[int] = None) -> None:
        self.id = uuid.uuid4().hex
        spec, self.opponent_ref = resolve_spec(opponent)
        self.opponent_label = label_for(self.opponent_ref)
        self.ai: ArenaPlayer = make_player(spec)
        self.ai.reset(self.id)
        self.env = TrucoEnv(seed=seed)
        self.last_obs, self.last_info = self.env.reset(seed=seed)
        self.last_reward: float = 0.0
        self.terminated = False
        self.stats = {side: {k: 0 for k in _STAT_KEYS.values()} for side in ("p0", "p1")}
        # hand_ending: True while frontend is showing the completed hand result.
        # frozen_hand: the Hand object that just ended (game.py already started
        # the next one, so we keep a reference for serialization).
        self.hand_ending = False
        self.frozen_hand = None
        # ai_pending: a round just closed and the AI opens the next one. The
        # AI waits for /game/continue so the frontend can show the finished
        # round on the table first.
        self.ai_pending = False
        # When the AI must act first (P1 leads), play it now.
        self._run_ai_turns()

    def _record(self, side: str, action_id: int) -> None:
        key = _STAT_KEYS.get(action_id)
        if key:
            self.stats[side][key] += 1

    # ---- AI driving --------------------------------------------------------
    def _ai_opens_next_round(self, hand, rounds_before: int) -> bool:
        """A round of ``hand`` just closed and the AI must lead the next one."""
        g = self.env.game
        return (g.state.winner is None and g.state.hand is hand
                and len(hand.rounds) > rounds_before
                and hand.current_player == Player.P1)

    def _run_ai_turns(self) -> None:
        g = self.env.game
        while (
            g.state.winner is None
            and g.state.hand is not None
            and g.state.hand.current_player == Player.P1
        ):
            hand, rounds_before = g.state.hand, len(g.state.hand.rounds)
            try:
                action = int(self.ai.decide(g, Player.P1))
            except Exception:
                action = -1
            # Fall back to the first legal action if the player returns
            # something illegal (e.g. a checkpoint trained without masking).
            pa, _ = to_legal_action(g, action)
            self._record("p1", encode_action(pa))
            g.step(pa)
            if self._ai_opens_next_round(hand, rounds_before):
                self.ai_pending = True
                return

    def _settle(self, current_hand) -> None:
        """Sync match end / hand end flags after moves on ``current_hand``."""
        g = self.env.game
        if g.state.winner is not None:
            self.terminated = True
            self.last_reward = 1.0 if g.state.winner == Player.P0 else -1.0
        elif g.state.hand is not current_hand:
            # Hand ended (by P0's action or by an AI play). Freeze the prior
            # hand so the frontend can render its result before /game/next-hand
            # advances to the new one.
            self.frozen_hand = current_hand
            self.hand_ending = True
        self.last_obs = self.env._observe()
        self.last_info = self.env._info()

    def step_human(self, action: int) -> None:
        if self.terminated:
            raise HTTPException(400, "Game is over.")
        if self.hand_ending:
            raise HTTPException(400, "Hand ended — call /game/next-hand first.")
        if self.ai_pending:
            raise HTTPException(400, "AI to play — call /game/continue first.")
        # Apply P0's action directly on the game, bypassing the env's internal
        # random opponent loop.
        g = self.env.game
        if g.state.hand is None or g.state.hand.current_player != Player.P0:
            raise HTTPException(400, "Not P0's turn.")
        legal = g.legal_actions()
        try:
            pa = decode_action(int(action))
        except ValueError:
            raise HTTPException(400, "Illegal action for current state.")
        if not any(actions_equal(pa, la) for la in legal):
            raise HTTPException(400, "Illegal action for current state.")

        current_hand = g.state.hand
        rounds_before = len(current_hand.rounds)
        self._record("p0", int(action))
        g.step(pa)
        if self._ai_opens_next_round(current_hand, rounds_before):
            # P0's card closed a round the AI now leads: pause for the table.
            self.ai_pending = True
        else:
            # Let the AI advance: its leading card on a new hand started by
            # P0's RUN/last-card step, and any pending P1 turns mid-hand.
            # _run_ai_turns is a no-op when the game is over or it's P0's turn.
            self._run_ai_turns()
        self._settle(current_hand)

    def continue_ai(self) -> None:
        """Called by /game/continue once the frontend has shown the round."""
        if not self.ai_pending:
            raise HTTPException(400, "Nothing to continue.")
        self.ai_pending = False
        current_hand = self.env.game.state.hand
        self._run_ai_turns()
        self._settle(current_hand)

    def advance_to_next_hand(self) -> None:
        """Called by /game/next-hand after the frontend has shown the result."""
        if not self.hand_ending:
            raise HTTPException(400, "No hand ending pending.")
        self.frozen_hand = None
        self.hand_ending = False
        self._run_ai_turns()
        # The AI may have just made its Mão de 11 decision; keep
        # session.terminated in sync in case that ended the match.
        g = self.env.game
        if g.state.winner is not None and not self.terminated:
            self.terminated = True
            self.last_reward = 1.0 if g.state.winner == Player.P0 else -1.0
        self.last_obs = self.env._observe()
        self.last_info = self.env._info()


GAMES: dict[str, GameSession] = {}


# --- Serialization ---------------------------------------------------------
def _card_dict(card) -> dict:
    return {"rank": int(card.rank), "suit": int(card.suit), "label": str(card)}


def _serialize_game(session: GameSession) -> dict:
    g = session.env.game
    s = g.state
    # When a hand just ended, serve the frozen hand data so the frontend
    # can display the final table state before requesting the next hand.
    h = session.frozen_hand if session.hand_ending else s.hand
    legal_ids: list[int] = []
    if h is not None and not session.hand_ending and h.current_player == Player.P0 and s.winner is None:
        legal_ids = session.last_info.get("legal_actions", [])

    payload: dict[str, Any] = {
        "game_id": session.id,
        "opponent": {"ref": session.opponent_ref, "label": session.opponent_label},
        "scores": {"p0": s.scores[0], "p1": s.scores[1]},
        "dealer": int(s.dealer),
        "iron_hand": s.iron_hand,
        "mao11_player": None if s.mao11_player is None else int(s.mao11_player),
        "match_winner": None if s.winner is None else int(s.winner),
        "terminated": session.terminated,
        "hand_ending": session.hand_ending,
        "ai_pending": session.ai_pending,
        "legal_actions": legal_ids,
        "stats": session.stats,
    }
    if h is None:
        return payload
    # The AI's cards are never sent: no rule reveals them to the human.
    payload.update({
        "vira": _card_dict(h.vira),
        "p0_hand": [_card_dict(c) for c in h.hands[Player.P0]],
        "p1_cards_left": len(h.hands[Player.P1]),
        "stake": h.stake,
        "pending_stake": h.pending_stake,
        "truco_caller": None if h.truco_caller is None else int(h.truco_caller),
        "current_player": int(h.current_player),
        "awaiting_mao11_response": h.awaiting_mao11_response,
        "rounds": [
            {
                "plays": [
                    {"player": int(p), "card": _card_dict(c)} for p, c in r.plays
                ],
                "result": None if r.result is None else int(r.result),
            }
            for r in h.rounds
        ],
        "hand_winner": None if h.winner is None else int(h.winner),
        "hand_drawn": h.drawn,
    })
    return payload


def _legacy_metric(m: Dict[str, Any]) -> Dict[str, Any]:
    """Shape the /watch page understood before runs existed (vs random)."""
    ev = m["eval"].get("random") or next(iter(m["eval"].values()), {})
    return {
        "timestep": m["timestep"],
        "episode": m["episode"],
        "win_rate": ev.get("win_rate", 0.0),
        "mean_reward": ev.get("reward_mean", 0.0),
        "mean_steps": 0.0,
        "entropy": m["train"].get("entropy") or 0.0,
        "truco_rate": ev.get("truco_rate", 0.0),
        "run_rate": ev.get("run_rate", 0.0),
    }


# --- Pydantic request models ----------------------------------------------
class TrainStartReq(BaseModel):
    total_timesteps: int = Field(default=DEFAULT_TIMESTEPS, ge=2048)
    name: str = "api"
    obs_version: str = Field(default="v2", pattern="^v[12]$")
    league: Optional[Dict[str, float]] = None
    eval_every: int = Field(default=50_000, ge=2048)
    eval_games: int = Field(default=200, ge=2, le=2000)
    init_from_run: Optional[str] = None


class NewGameReq(BaseModel):
    # Level (facil | medio | dificil | impossivel), "random", "rule" or a
    # checkpoint reference such as "ppo:models/truco_liga_v2.zip".
    # Default: impossivel.
    opponent: Optional[str] = None


class GameActionReq(BaseModel):
    game_id: str
    action: int


class NextHandReq(BaseModel):
    game_id: str


class ContinueReq(BaseModel):
    game_id: str


# --- Training endpoints ----------------------------------------------------
@router.post("/train/start")
def train_start(req: TrainStartReq):
    cfg = TrainConfig(name=req.name, total_timesteps=req.total_timesteps,
                      obs_version=req.obs_version, eval_every=req.eval_every,
                      snapshot_every=req.eval_every, eval_games=req.eval_games,
                      eval_workers=ARENA_WORKERS, vec_env="dummy")
    if req.league:
        cfg.league = dict(req.league)
    if req.init_from_run:
        src = run_dir(req.init_from_run) / "final.zip"
        if not src.is_file():
            raise HTTPException(409, "That run has no final.zip to continue from.")
        cfg.init_from = str(src)
    with TRAIN.lock:
        if TRAIN.is_alive():
            raise HTTPException(409, "Training already running.")
        TRAIN.stop_flag.clear()
        TRAIN.callback = None
        TRAIN.metrics = []
        TRAIN.run_id = None
        TRAIN.error = None
        TRAIN.paused = False
        TRAIN.total_timesteps = req.total_timesteps
        TRAIN.thread = threading.Thread(target=_train_worker, args=(cfg,), daemon=True)
        TRAIN.thread.start()
    return {"status": "started"}


@router.post("/train/pause")
def train_pause():
    """Stop the current run; it is saved as ``stopped`` with final.zip."""
    with TRAIN.lock:
        if not TRAIN.is_alive():
            return {"status": "not_running", **_status_payload()}
        TRAIN.stop_flag.set()
        TRAIN.paused = True
    if TRAIN.thread is not None:
        TRAIN.thread.join(timeout=10.0)
    return {"status": "paused", **_status_payload()}


@router.post("/train/reset")
def train_reset():
    """Stop training and forget the in-memory state. Runs on disk are kept."""
    with TRAIN.lock:
        TRAIN.stop_flag.set()
    if TRAIN.thread is not None:
        TRAIN.thread.join(timeout=10.0)
    with TRAIN.lock:
        TRAIN.thread = None
        TRAIN.callback = None
        TRAIN.metrics = []
        TRAIN.run_id = None
        TRAIN.error = None
        TRAIN.paused = False
        TRAIN.stop_flag.clear()
    return {"status": "reset"}


def _status_payload() -> dict:
    cb = TRAIN.callback
    latest = TRAIN.metrics[-1] if TRAIN.metrics else None
    return {
        "running": TRAIN.is_alive(),
        "paused": TRAIN.paused and not TRAIN.is_alive(),
        "run_id": TRAIN.run_id,
        "timestep": int(cb.num_timesteps) if cb is not None and cb.model is not None else 0,
        "total_timesteps": TRAIN.total_timesteps,
        "episode": cb.episodes if cb is not None else 0,
        "error": TRAIN.error,
        "latest_metric": _legacy_metric(latest) if latest is not None else None,
        "latest": latest,
    }


@router.get("/train/status")
def train_status():
    return _status_payload()


# --- Game endpoints --------------------------------------------------------
@router.post("/game/new")
def game_new(req: Optional[NewGameReq] = None):
    try:
        session = GameSession(opponent=req.opponent if req else None)
    except FileNotFoundError as e:
        raise HTTPException(409, f"Model not available: {e}")
    GAMES[session.id] = session
    return _serialize_game(session)


@router.post("/game/action")
def game_action(req: GameActionReq):
    session = GAMES.get(req.game_id)
    if session is None:
        raise HTTPException(404, "Unknown game_id.")
    session.step_human(req.action)
    return _serialize_game(session)


@router.get("/game/state")
def game_state(game_id: str):
    session = GAMES.get(game_id)
    if session is None:
        raise HTTPException(404, "Unknown game_id.")
    return _serialize_game(session)


@router.post("/game/continue")
def game_continue(req: ContinueReq):
    """Let the AI open the next round after the pause (see ai_pending)."""
    session = GAMES.get(req.game_id)
    if session is None:
        raise HTTPException(404, "Unknown game_id.")
    session.continue_ai()
    return _serialize_game(session)


@router.post("/game/next-hand")
def game_next_hand(req: NextHandReq):
    session = GAMES.get(req.game_id)
    if session is None:
        raise HTTPException(404, "Unknown game_id.")
    session.advance_to_next_hand()
    return _serialize_game(session)


# --- WebSocket: metrics stream --------------------------------------------
@router.websocket("/ws/metrics")
async def ws_metrics(ws: WebSocket):
    await ws.accept()
    last_sent = 0
    run_id = None
    try:
        while True:
            if TRAIN.run_id != run_id:
                run_id, last_sent = TRAIN.run_id, 0
            metrics = TRAIN.metrics
            if len(metrics) > last_sent:
                new = metrics[last_sent:]
                last_sent = len(metrics)
                await ws.send_json({
                    "type": "metrics",
                    "run_id": run_id,
                    "items": [_legacy_metric(m) for m in new],
                    "full": new,
                })
            await ws.send_json({"type": "status", **_status_payload()})
            await asyncio.sleep(2.0)
    except WebSocketDisconnect:
        return
