from truco.env import TrucoEnv
import numpy as np

for ep in range(5):
    env = TrucoEnv(seed=ep)
    obs, info = env.reset()
    env.render()
    print(f"Legal actions: {info['legal_actions']}")
    done = False
    steps = 0
    while not done:
        legal = info['legal_actions']
        if not legal:
            print("SEM AÇÕES LEGAIS — bug!")
            break
        action = legal[0]  # sempre pega a primeira ação legal
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        done = terminated or truncated
        if done or steps % 5 == 0:
            env.render()
            print(f"step={steps} reward={reward} legal={info['legal_actions']}")
    print(f"Episódio {ep}: {steps} steps, reward={reward}\n")