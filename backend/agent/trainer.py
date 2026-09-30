"""PPO training loop for the Truco environment.

Uses :class:`sb3_contrib.MaskablePPO` directly with a bare ``TrucoEnv``.
The env exposes ``action_masks()``, which is the interface MaskablePPO
discovers natively — no ``ActionMasker`` wrapper is needed.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from truco.encoding import ACT_CALL_TRUCO, ACT_RUN, action_mask, to_legal_action
from truco.env import TrucoEnv
from truco.game import Player
from truco.obs import observe
from truco.seeds import eval_game_seeds


# --- Env factory -----------------------------------------------------------
def make_env(seed: Optional[int] = None) -> TrucoEnv:
    return TrucoEnv(seed=seed)


# --- Self-play env --------------------------------------------------------
class SelfPlayEnv(TrucoEnv):
    """TrucoEnv where P1 is driven by a frozen snapshot of the training policy.

    Until the first weight swap, ``frozen_model`` is None and P1 falls back
    to the inherited uniform-random opponent. The owning ``SelfPlayCallback``
    creates a frozen MaskablePPO from the training model on the first swap
    and reloads its weights every ``swap_freq`` episodes thereafter.
    """

    def __init__(self, *, swap_freq: int = 200, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.swap_freq = swap_freq
        self.frozen_model: Optional[MaskablePPO] = None

    def set_frozen_model(self, model: MaskablePPO) -> None:
        self.frozen_model = model

    def _play_opponent_until_p0(self) -> None:
        g = self.game
        while (
            g.state.winner is None
            and g.state.hand is not None
            and g.state.hand.current_player == Player.P1
        ):
            legal = g.legal_actions()
            if not legal:
                break

            if self.frozen_model is None:
                pa = self._opp_rng.choice(legal)
            else:
                action_arr, _ = self.frozen_model.predict(
                    observe(g, Player.P1, self.obs_version),
                    action_masks=action_mask(g, Player.P1),
                    deterministic=False,
                )
                pa, _ = to_legal_action(g, int(np.asarray(action_arr).item()))
            g.step(pa)


class SelfPlayCallback(BaseCallback):
    """Refreshes the frozen opponent every ``swap_freq`` episodes."""

    def __init__(self, env: SelfPlayEnv, swap_freq: int = 200, verbose: int = 0):
        super().__init__(verbose)
        self.env = env
        self.swap_freq = swap_freq
        self._episode_count = 0
        self._last_swap = 0
        self._frozen: Optional[MaskablePPO] = None

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        if dones is not None:
            self._episode_count += int(np.sum(dones))
        if self._episode_count - self._last_swap >= self.swap_freq:
            self._last_swap = self._episode_count
            self._swap_weights()
        return True

    def _swap_weights(self) -> None:
        if self._frozen is None:
            buf = io.BytesIO()
            self.model.save(buf)
            buf.seek(0)
            # ``device='cpu'`` keeps the frozen forward pass off-GPU and
            # avoids contention with the training optimizer step.
            self._frozen = MaskablePPO.load(buf, device="cpu")
            self.env.set_frozen_model(self._frozen)
            if self.verbose:
                print(f"[selfplay] frozen opponent created at ep={self._episode_count}")
        else:
            self._frozen.policy.load_state_dict(self.model.policy.state_dict())
            if self.verbose:
                print(f"[selfplay] frozen weights refreshed at ep={self._episode_count}")


# --- Eval / metrics callback ----------------------------------------------
@dataclass
class EvalMetric:
    timestep: int
    episode: int
    win_rate: float
    mean_reward: float
    mean_steps: float
    entropy: float = 0.0
    truco_rate: float = 0.0
    run_rate: float = 0.0


class TrainingCallback(BaseCallback):
    """Periodically evaluates the agent vs the env's random opponent.

    Metrics are appended to :attr:`metrics`, which is meant to be polled by an
    external consumer (e.g. a WebSocket pushing live training status).
    """

    def __init__(
        self,
        eval_freq_episodes: int = 50,
        eval_episodes: int = 100,
        eval_seed: int = 12345,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_freq_episodes = eval_freq_episodes
        self.eval_episodes = eval_episodes
        self.eval_seed = eval_seed
        self.metrics: List[EvalMetric] = []
        # Fixed evaluation deals, disjoint from the training seed space.
        self._eval_game_seeds = eval_game_seeds(eval_seed, eval_episodes)
        self._episode_count = 0
        self._last_eval_episode = 0

    def _on_step(self) -> bool:
        # Track episode boundaries from the vec env.
        dones = self.locals.get("dones")
        if dones is not None:
            self._episode_count += int(np.sum(dones))

        if (self._episode_count - self._last_eval_episode) >= self.eval_freq_episodes:
            self._last_eval_episode = self._episode_count
            self._run_eval()
        return True

    def _on_training_end(self) -> None:
        # Final eval snapshot.
        self._run_eval()

    def _run_eval(self) -> None:
        wins = 0
        rewards: List[float] = []
        steps_list: List[int] = []
        truco_calls = 0
        run_calls = 0
        total_hands = 0
        for i in range(self.eval_episodes):
            env = make_env(seed=self.eval_seed + i)
            obs, info = env.reset(options={"game_seed": self._eval_game_seeds[i]})
            done = False
            total_r = 0.0
            steps = 0
            prev_hand_id = id(env.game.state.hand)
            if env.game.state.hand is not None:
                total_hands += 1
            while not done:
                mask = env.action_masks()
                action, _ = self.model.predict(
                    obs, action_masks=mask, deterministic=True
                )
                action_int = int(action)
                if action_int == ACT_CALL_TRUCO:
                    truco_calls += 1
                if action_int == ACT_RUN:
                    run_calls += 1
                obs, reward, terminated, truncated, info = env.step(action_int)
                total_r += float(reward)
                steps += 1
                done = terminated or truncated
                cur_hand = env.game.state.hand
                if cur_hand is not None and id(cur_hand) != prev_hand_id:
                    total_hands += 1
                    prev_hand_id = id(cur_hand)
            if total_r > 0:
                wins += 1
            rewards.append(total_r)
            steps_list.append(steps)

        entropy = 0.0
        if self.model is not None and self.model.logger is not None:
            entropy = -float(
                self.model.logger.name_to_value.get("train/entropy_loss", 0.0)
            )

        metric = EvalMetric(
            timestep=int(self.num_timesteps),
            episode=self._episode_count,
            win_rate=wins / self.eval_episodes,
            mean_reward=float(np.mean(rewards)),
            mean_steps=float(np.mean(steps_list)),
            entropy=entropy,
            truco_rate=truco_calls / max(total_hands, 1),
            run_rate=run_calls / max(truco_calls, 1),
        )
        self.metrics.append(metric)
        if self.verbose:
            print(
                f"[eval] t={metric.timestep} ep={metric.episode} "
                f"win_rate={metric.win_rate:.3f} "
                f"reward={metric.mean_reward:.3f} steps={metric.mean_steps:.1f}"
            )


# --- Train / load ----------------------------------------------------------
def build_model(
    env: TrucoEnv,
    *,
    seed: Optional[int] = 0,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 128,
    ent_coef: float = 0.05,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    normalize_advantage: bool = True,
    verbose: int = 1,
) -> MaskablePPO:
    return MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        normalize_advantage=normalize_advantage,
        seed=seed,
        verbose=verbose,
    )


def train(
    total_timesteps: int,
    save_path: str | Path,
    callback: Optional[BaseCallback] = None,
    seed: Optional[int] = 0,
    model: Optional[MaskablePPO] = None,
    self_play: bool = False,
    swap_freq: int = 200,
    **build_kwargs: Any,
) -> MaskablePPO:
    """Train a MaskablePPO agent on the Truco env and save to ``save_path``.

    The defaults in :func:`build_model` are tuned to avoid the policy-collapse
    failure we saw with the ActionMasker wrapper (``entropy_loss`` and
    ``clip_fraction`` pinned at zero): bare env with native ``action_masks``
    plus an aggressive ``ent_coef=0.05`` to keep exploration alive,
    ``vf_coef=0.5`` and ``max_grad_norm=0.5`` for value/grad stability, and
    explicit advantage normalization for sparse +/-1 returns.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    sp_env: Optional[SelfPlayEnv] = None
    if model is None:
        if self_play:
            sp_env = SelfPlayEnv(seed=seed, swap_freq=swap_freq)
            model = build_model(sp_env, seed=seed, **build_kwargs)
        else:
            env = make_env(seed=seed)
            model = build_model(env, seed=seed, **build_kwargs)

    callbacks: List[BaseCallback] = []
    if sp_env is not None:
        callbacks.append(
            SelfPlayCallback(sp_env, swap_freq=swap_freq, verbose=1)
        )
    if callback is not None:
        callbacks.append(callback)

    if not callbacks:
        cb: Optional[BaseCallback] = None
    elif len(callbacks) == 1:
        cb = callbacks[0]
    else:
        cb = CallbackList(callbacks)

    model.learn(
        total_timesteps=total_timesteps,
        callback=cb,
        use_masking=True,
        reset_num_timesteps=False,
    )
    model.save(str(save_path))
    return model


def smoke_test(
    timesteps: int = 2000,
    seed: int = 0,
) -> tuple[bool, dict[str, float], MaskablePPO]:
    """Run a tiny training pass and verify the policy is actually updating.

    Returns ``(passed, metrics, model)`` where ``passed`` is ``True`` iff the
    last rollout produced a non-zero ``entropy_loss``. Useful as a guard
    before kicking off a multi-hour run.
    """
    env = make_env(seed=seed)
    model = build_model(env, seed=seed, verbose=0)
    model.learn(total_timesteps=timesteps, use_masking=True)
    values = dict(model.logger.name_to_value)
    metrics = {
        "entropy_loss": float(values.get("train/entropy_loss", 0.0)),
        "clip_fraction": float(values.get("train/clip_fraction", 0.0)),
        "policy_gradient_loss": float(values.get("train/policy_gradient_loss", 0.0)),
        "value_loss": float(values.get("train/value_loss", 0.0)),
    }
    passed = abs(metrics["entropy_loss"]) > 1e-9
    return passed, metrics, model


PredictFn = Callable[[np.ndarray, np.ndarray], int]


def load_agent(path: str | Path) -> PredictFn:
    """Load a saved MaskablePPO model and return a ``predict(obs, mask)`` fn."""
    model = MaskablePPO.load(str(path))

    def predict(obs: np.ndarray, action_mask: np.ndarray) -> int:
        action, _ = model.predict(obs, action_masks=action_mask, deterministic=True)
        return int(action)

    return predict


# --- Difficulty registry ---------------------------------------------------
@dataclass
class DifficultyAgent:
    level: str
    predict: PredictFn
    cheat: bool = False
    meta: dict = field(default_factory=dict)


# On-disk location for the trained checkpoint used by default.
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "truco_ppo_1M"


def get_difficulty_agent() -> DifficultyAgent:
    """Return the default agent (1M-step MaskablePPO, cheat mode enabled)."""
    predict = load_agent(MODEL_PATH)
    return DifficultyAgent(level="impossivel", predict=predict, cheat=True,
                           meta={"path": str(MODEL_PATH)})


if __name__ == "__main__":
    print("[smoke] running 2000-step smoke test...")
    env = make_env(seed=0)
    model = MaskablePPO(
        "MlpPolicy",
        env,
        ent_coef=0.05,
        vf_coef=0.5,
        max_grad_norm=0.5,
        n_steps=512,
        batch_size=32,
        normalize_advantage=True,
        seed=0,
        verbose=1,
    )
    model.learn(total_timesteps=2000, use_masking=True)
    entropy_loss = model.logger.name_to_value.get("train/entropy_loss", 0)
    clip_fraction = model.logger.name_to_value.get("train/clip_fraction", 0)
    print(f"[smoke] entropy_loss={entropy_loss:.6f} clip_fraction={clip_fraction:.6f}")
    if entropy_loss == 0:
        raise SystemExit(
            "[smoke] FAILED: entropy_loss is zero — policy is not updating. "
            "Aborting full run."
        )
    print("[smoke] PASSED. Starting self-play training.")

    cb = TrainingCallback(eval_freq_episodes=50, eval_episodes=50, verbose=1)
    train(
        total_timesteps=1_000_000,
        save_path=MODELS_DIR / "truco_ppo_selfplay",
        callback=cb,
        seed=0,
        self_play=True,
        swap_freq=200,
    )
    if cb.metrics:
        final = cb.metrics[-1]
        print(f"Final win rate vs random: {final.win_rate:.1%}")
    print(f"Collected {len(cb.metrics)} eval snapshots.")
