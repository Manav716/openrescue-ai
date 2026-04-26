import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO

import env  # noqa: F401 - registers OpenRescueAI-v0


def run_episode(model, seed):
    base_env = gym.make("OpenRescueAI-v0")
    wrapped_env = FlattenObservation(base_env)
    obs, _ = wrapped_env.reset(seed=seed)
    terminated = False
    truncated = False
    episode_return = 0.0
    steps = 0

    while not terminated and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        episode_return += float(reward)
        steps += 1

    wrapped_env.close()
    return {
        "seed": seed,
        "return": episode_return,
        "steps": steps,
        "civilians_saved": info.get("civilians_saved", 0),
        "messages": info.get("messages", []),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained OpenRescue PPO policy.")
    parser.add_argument("--model", default="models/best_model.zip", help="Path to a Stable-Baselines3 PPO zip.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="plots/evaluation.json")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = PPO.load(str(model_path))
    episodes = [run_episode(model, args.seed + idx) for idx in range(args.episodes)]
    returns = np.array([episode["return"] for episode in episodes], dtype=np.float32)

    result = {
        "model": str(model_path),
        "episodes": episodes,
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "min_return": float(returns.min()),
        "max_return": float(returns.max()),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
