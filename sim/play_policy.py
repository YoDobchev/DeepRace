from __future__ import annotations

import argparse
import time
from pathlib import Path

from stable_baselines3 import DQN


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="Path to .zip model (e.g. runs/track_dqn/best_model/best_model.zip)")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    model_path = Path(args.model)
    if model_path.suffix != ".zip":
        if model_path.with_suffix(".zip").exists():
            model_path = model_path.with_suffix(".zip")

    from track_follow_env import TapeLineFollowEnv

    env = TapeLineFollowEnv(render_mode="human", seed=args.seed)

    model = DQN.load(str(model_path), device=args.device)

    try:
        for ep in range(args.episodes):
            obs, info = env.reset()
            terminated = truncated = False
            ep_return = 0.0
            ep_steps = 0

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)  
                obs, reward, terminated, truncated, info = env.step(int(action))  
                ep_return += float(reward)
                ep_steps += 1

            print(f"Episode {ep+1}/{args.episodes} return={ep_return:.3f} steps={ep_steps}")

    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
