import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

import env  # noqa: F401 - registers OpenRescueAI-v0


def main():
    parser = argparse.ArgumentParser(description="Train PPO on OpenRescue AI.")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="models/best_model")
    args = parser.parse_args()

    base_env = gym.make("OpenRescueAI-v0")
    train_env = Monitor(FlattenObservation(base_env))

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        seed=args.seed,
        tensorboard_log="runs/openrescue_ppo",
    )
    model.learn(total_timesteps=args.timesteps)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output))
    train_env.close()
    print(f"Saved PPO model to {output}.zip")


if __name__ == "__main__":
    main()
