# train_dqn.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.utils import set_random_seed


def make_env(seed: int, render: str | None = None):
    from track_follow_env import TapeLineFollowEnv
    # from track_follow_env_real import TapeLineFollowEnvRealistic

    env = TapeLineFollowEnv(render_mode=render, seed=seed)
    return env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=600_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--logdir", type=str, default="runs/track_dqn")
    p.add_argument("--model-name", type=str, default="dqn_track")
    p.add_argument("--eval-freq", type=int, default=10_000)
    p.add_argument("--n-eval-episodes", type=int, default=10)
    p.add_argument("--save-freq", type=int, default=50_000)
    p.add_argument("--device", type=str, default="auto")

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--buffer-size", type=int, default=200_000)
    p.add_argument("--learning-starts", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--train-freq", type=int, default=4)
    p.add_argument("--gradient-steps", type=int, default=1)
    p.add_argument("--target-update-interval", type=int, default=5_000)
    p.add_argument("--exploration-fraction", type=float, default=0.30)
    p.add_argument("--exploration-final-eps", type=float, default=0.05)

    args = p.parse_args()

    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (logdir / "best_model").mkdir(parents=True, exist_ok=True)

    set_random_seed(args.seed)
    np.random.seed(args.seed)

    check_env(make_env(seed=args.seed, render=None),
              warn=True, skip_render_check=True)

    train_env = Monitor(make_env(seed=args.seed, render=None),
                        filename=str(logdir / "monitor_train.csv"))
    eval_env = Monitor(make_env(seed=args.seed + 10_000,
                       render=None), filename=str(logdir / "monitor_eval.csv"))

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(logdir / "best_model"),
        log_path=str(logdir / "eval_logs"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(logdir / "checkpoints"),
        name_prefix=args.model_name,
        save_replay_buffer=True,
        save_vecnormalize=False,
    )

    callbacks = CallbackList([eval_cb, ckpt_cb])

    model = DQN(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=args.lr,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        verbose=1,
        tensorboard_log=str(logdir / "tb"),
        device=args.device,
        seed=args.seed,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        log_interval=10,
        progress_bar=True,
    )

    out_path = logdir / f"{args.model_name}_final"
    model.save(str(out_path))
    print(f"Saved final model to: {out_path}.zip")


if __name__ == "__main__":
    main()
