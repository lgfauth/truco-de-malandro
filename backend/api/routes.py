"""FastAPI routes for the Truco RL project.

Endpoints:
    POST /train/start, /train/pause, /train/reset, GET /train/status
    POST /game/new, /game/action, GET /game/state
    WS   /ws/metrics

Training runs in a background thread; pause/reset signal a stop flag honored
by a control callback. Game state lives in an in-memory dict (``GAMES``); no
database is involved at this stage.
"""

from __future__ import annotations

import asyncio
import shutil
import threading
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from sb3_contrib import MaskablePPO

from agent.trainer import (
    MODELS_DIR,
    DifficultyAgent,
    EvalMetric,
    TrainingCallback,
    get_difficulty_agent,
    train,
)
from truco.encoding import action_mask, actions_equal, decode_action, to_legal_action
from truco.env import TrucoEnv
from truco.game import Player
from truco.obs import observe


router = APIRouter()

DEFAULT_MODEL_PATH = MODELS_DIR / "truco_ppo"
DEFAULT_TIMESTEPS = 500_000


# --- Training state --------------------------------------------------------
class _TrainState:
    """Singleton holder for the background training job."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.callback: Optional[TrainingCallback] = None
        self.stop_flag = threading.Event()
        self.running = False
        self.paused = False

    def is_alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive()


TRAIN = _TrainState()


class _ControlCallback(TrainingCallback):
    """Eval callback that also halts training when the stop flag is set."""

    def __init__(self, stop_flag: threading.Event, **kw: Any) -> None:
        super().__init__(**kw)
        self._stop_flag = stop_flag

    def _on_step(self) -> bool:
        if self._stop_flag.is_set():
            return False
        return super()._on_step()


def _train_worker(total_timesteps: int, save_path: Path) -> None:
    try:
        # Try to continue from the previously saved model so the agent
        # always improves rather than restarting from scratch.
        existing = save_path.with_suffix(".zip")
        loaded_model = None
        if existing.exists():
            try:
                loaded_model = MaskablePPO.load(str(existing))
                # Re-attach a fresh env so the model can keep learning.
                from truco.env import make_env as _make_env
                loaded_model.set_env(_make_env(seed=0))
            except Exception:
                loaded_model = None  # corrupt / incompatible — start fresh
        train(
            total_timesteps=total_timesteps,
            save_path=save_path,
            callback=TRAIN.callback,
            model=loaded_model,
            seed=0,
            verbose=0,
        )
    finally:
        with TRAIN.lock:
            TRAIN.running = False


# --- Game state ------------------------------------------------------------
class GameSession:
    """Holds a TrucoEnv plus the AI configured for P1."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.id = uuid.uuid4().hex
        self.agent: DifficultyAgent = get_difficulty_agent()
        self.env = TrucoEnv(seed=seed)
        self.last_obs, self.last_info = self.env.reset(seed=seed)
        self.last_reward: float = 0.0
        self.terminated = False
        # hand_ending: True while frontend is showing the completed hand result.
        # frozen_hand: the Hand object that just ended (game.py already started
        # the next one, so we keep a reference for serialization).
        self.hand_ending = False
        self.frozen_hand = None
        # When the AI must act first (P1 leads), play it now.
        self._run_ai_turns()

    # ---- AI driving (replaces env's random P1 with our difficulty agent) --
    def _run_ai_turns(self) -> None:
        g = self.env.game
        while (
            g.state.winner is None
            and g.state.hand is not None
            and g.state.hand.current_player == Player.P1
        ):
            action = self.agent.predict(observe(g, Player.P1),
                                        action_mask(g, Player.P1))
            # Fall back to the first legal action if the model returns
            # something illegal (e.g. a checkpoint trained without masking).
            pa, _ = to_legal_action(g, int(action))
            g.step(pa)

    def step_human(self, action: int) -> None:
        if self.terminated:
            raise HTTPException(400, "Game is over.")
        if self.hand_ending:
            raise HTTPException(400, "Hand ended — call /game/next-hand first.")
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
        g.step(pa)
        # Always let the AI advance: it must play its leading card on a new
        # hand started by P0's RUN/last-card step, and any pending P1 turns
        # mid-hand. _run_ai_turns is a no-op when the game is over or it's
        # P0's turn, so this is safe in every branch.
        self._run_ai_turns()

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

    def advance_to_next_hand(self) -> None:
        """Called by /game/next-hand after the frontend has shown the result."""
        if not self.hand_ending:
            raise HTTPException(400, "No hand ending pending.")
        self.frozen_hand = None
        self.hand_ending = False
        self._run_ai_turns()
        # The AI may have just decided the mão de 11 response (accept/run)
        # and that decision might have ended the match (score reached 12).
        # Sync session.terminated so the frontend receives the correct state.
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
        "scores": {"p0": s.scores[0], "p1": s.scores[1]},
        "dealer": int(s.dealer),
        "iron_hand": s.iron_hand,
        "open_hand_for": None if s.open_hand_for is None else int(s.open_hand_for),
        "match_winner": None if s.winner is None else int(s.winner),
        "terminated": session.terminated,
        "hand_ending": session.hand_ending,
        "legal_actions": legal_ids,
    }
    if h is None:
        return payload
    payload.update({
        "vira": _card_dict(h.vira),
        "p0_hand": [_card_dict(c) for c in h.hands[Player.P0]],
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
    })

    # Reveal P1's hand to the player when:
    # - Mão de 11 with P1 at 11 (open_hand_for == P0): rules say P1's hand
    #   is shown to the opponent.
    # - The agent is in cheat mode (impossible difficulty): the player gets
    #   to peek as a UI visualization of the cheat.
    if s.open_hand_for == Player.P0:
        payload["p1_hand"] = [_card_dict(c) for c in h.hands[Player.P1]]
    if session.agent.cheat:
        payload["p1_hand"] = [_card_dict(c) for c in h.hands[Player.P1]]
        payload["cheat_active"] = True
    return payload


def _serialize_metric(m: EvalMetric) -> dict:
    return asdict(m)


# --- Pydantic request models ----------------------------------------------
class TrainStartReq(BaseModel):
    total_timesteps: int = Field(default=DEFAULT_TIMESTEPS, ge=1)
    save_path: Optional[str] = None


class NewGameReq(BaseModel):
    pass


class GameActionReq(BaseModel):
    game_id: str
    action: int


# --- Training endpoints ----------------------------------------------------
@router.post("/train/start")
def train_start(req: TrainStartReq):
    with TRAIN.lock:
        if TRAIN.is_alive():
            raise HTTPException(409, "Training already running.")
        TRAIN.stop_flag.clear()
        TRAIN.callback = _ControlCallback(
            stop_flag=TRAIN.stop_flag,
            eval_freq_episodes=200,
            eval_episodes=20,
        )
        TRAIN.running = True
        TRAIN.paused = False
        save_path = Path(req.save_path) if req.save_path else DEFAULT_MODEL_PATH
        TRAIN.thread = threading.Thread(
            target=_train_worker,
            args=(req.total_timesteps, save_path),
            daemon=True,
        )
        TRAIN.thread.start()
    return {"status": "started"}


@router.post("/train/pause")
def train_pause():
    with TRAIN.lock:
        if not TRAIN.is_alive():
            return {"status": "not_running", **_status_payload()}
        TRAIN.stop_flag.set()
        TRAIN.paused = True
    # Wait briefly for the worker to honor the stop flag.
    if TRAIN.thread is not None:
        TRAIN.thread.join(timeout=5.0)
    return {"status": "paused", **_status_payload()}


@router.post("/train/reset")
def train_reset():
    with TRAIN.lock:
        TRAIN.stop_flag.set()
    if TRAIN.thread is not None:
        TRAIN.thread.join(timeout=5.0)
    with TRAIN.lock:
        TRAIN.thread = None
        TRAIN.callback = None
        TRAIN.running = False
        TRAIN.paused = False
        TRAIN.stop_flag.clear()
    # Remove the saved model artifact (file or directory).
    target = DEFAULT_MODEL_PATH
    for candidate in (target, target.with_suffix(".zip")):
        if candidate.exists():
            if candidate.is_dir():
                shutil.rmtree(candidate, ignore_errors=True)
            else:
                candidate.unlink(missing_ok=True)
    return {"status": "reset"}


def _status_payload() -> dict:
    cb = TRAIN.callback
    latest = cb.metrics[-1] if (cb is not None and cb.metrics) else None
    return {
        "running": TRAIN.is_alive(),
        "paused": TRAIN.paused and not TRAIN.is_alive(),
        "timestep": int(cb.num_timesteps) if cb is not None and cb.model is not None else 0,
        "episode": cb._episode_count if cb is not None else 0,
        "latest_metric": _serialize_metric(latest) if latest is not None else None,
    }


@router.get("/train/status")
def train_status():
    return _status_payload()


# --- Game endpoints --------------------------------------------------------
@router.post("/game/new")
def game_new():
    try:
        session = GameSession()
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


class NextHandReq(BaseModel):
    game_id: str


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
    try:
        while True:
            cb = TRAIN.callback
            if cb is not None:
                metrics = cb.metrics
                if len(metrics) > last_sent:
                    new = metrics[last_sent:]
                    last_sent = len(metrics)
                    await ws.send_json({
                        "type": "metrics",
                        "items": [_serialize_metric(m) for m in new],
                    })
            await ws.send_json({"type": "status", **_status_payload()})
            if not TRAIN.is_alive() and (cb is None or last_sent >= len(cb.metrics)):
                # Training finished and all metrics flushed; keep the socket
                # alive but back off — client may start a new run.
                pass
            await asyncio.sleep(2.0)
    except WebSocketDisconnect:
        return
