"""Train and save the default agent model.

Run with::

    python -m train_all

from inside ``backend/``. Produces:
    models/truco_ppo_1M
"""

from __future__ import annotations

from agent.trainer import MODELS_DIR, TrainingCallback, train


def main() -> None:
    name = "truco_ppo_1M"
    timesteps = 1_000_000
    print(f"\n=== Treinando {name} ({timesteps:,} steps) ===")
    cb = TrainingCallback(eval_freq_episodes=50, eval_episodes=50, verbose=1)
    train(
        total_timesteps=timesteps,
        save_path=MODELS_DIR / name,
        callback=cb,
        seed=42,
    )
    final = cb.metrics[-1]
    print(f"\nWin rate final: {final.win_rate:.1%}")


if __name__ == "__main__":
    main()
