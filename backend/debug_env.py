from truco.env import TrucoEnv
import numpy as np

NUM_EPISODES = 50
stuck = []

for ep in range(NUM_EPISODES):
    env = TrucoEnv(seed=ep)
    obs, info = env.reset()
    done = False
    steps = 0
    reward = 0.0
    while not done:
        legal = info['legal_actions']
        if not legal:
            print(f"Ep {ep}: SEM AÇÕES LEGAIS após {steps} steps — bug!")
            stuck.append(ep)
            break
        action = legal[0]  # sempre pega a primeira ação legal
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        done = terminated or truncated
    # Confirm the underlying game has a winner set
    game_winner = env.game.state.winner
    if game_winner is None:
        print(f"Ep {ep}: terminado sem winner setado após {steps} steps — BUG!")
        stuck.append(ep)
    else:
        print(f"Ep {ep:>3}: {steps:>4} steps | reward={reward:+.0f} | winner=P{int(game_winner)}")

print()
if stuck:
    print(f"FALHA: {len(stuck)} episódio(s) com problema: {stuck}")
else:
    print(f"OK: todos os {NUM_EPISODES} episódios terminaram corretamente com winner setado.")