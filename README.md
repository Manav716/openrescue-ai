---
title: OpenRescue AI
emoji: 🚁
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# OpenRescue AI

OpenRescue AI is an OpenEnv-compatible disaster response environment for the Meta + Hugging Face + PyTorch OpenEnv Hackathon. It combines a Gymnasium grid world, PPO training, autonomous drone scouting, partial observability, dynamic fire spread, shared memory, and Hugging Face transformer-based command interpretation.

The project is intentionally submission-focused: the original PPO and NLP behavior is preserved, while the repository now includes packaging, an OpenEnv manifest, HTTP endpoints, reproducible scripts, and Hugging Face Spaces deployment files.

## What The Agent Does

The learned PPO policy controls a rescue agent on a 10x10 disaster grid. A separate autonomous drone explores the map, discovers fires and civilians, and emits semantic messages such as `Civilian detected at (5, 5)`. The demo path classifies those messages with `facebook/bart-large-mnli`; when a civilian report is classified as a critical rescue target, a symbolic coordinate override temporarily routes the rescue agent to the target.

## Architecture

| Layer | File | Responsibility |
| --- | --- | --- |
| Environment API | `env/disaster_env.py` | Gymnasium reset/step loop, rewards, partial observations, shared memory updates |
| World model | `env/world.py` | Grid state, static scenario layout, dynamic fire spread |
| Agents | `env/agents.py` | Rescue and drone movement, drone message generation |
| Constants | `env/constants.py` | Tile IDs, actions, defaults |
| OpenEnv server | `app.py` | `/health`, `/reset`, `/step`, `/state`, `/tasks`, `/web` |
| Pygame demo | `main.py` | PPO model loading, renderer, Hugging Face NLP command override |
| Training | `scripts/train_ppo.py` | Reproducible PPO training entrypoint |
| Evaluation | `scripts/evaluate.py` | Deterministic model evaluation |
| Plotting | `scripts/plot_rewards.py` | Reward progression chart generation |

## Environment Specification

Action space: `Discrete(4)`

| Action | Meaning |
| --- | --- |
| `0` | Move up |
| `1` | Move down |
| `2` | Move left |
| `3` | Move right |

Observation space: `Box(low=-1, high=4, shape=(10, 10), dtype=int32)`

| Value | Meaning |
| --- | --- |
| `-1` | Unknown / fog of war |
| `0` | Empty |
| `1` | Fire |
| `2` | Civilian |
| `3` | Rescue agent |
| `4` | Drone |

Reward shaping:

| Event | Reward |
| --- | ---: |
| Step cost | `-1` |
| New rescue-agent cell explored | `+5` |
| Drone discovers civilian | `+20` |
| Civilian rescued | `+100` |
| All civilians rescued | `+200` |
| Rescue agent enters fire | `-100` |

## Results

The hackathon reward progression is preserved in `openenv.yaml` and visualized by `plots/reward_curve.png`.

| System | Reward |
| --- | ---: |
| Random Agent | `-5000` |
| Early PPO | `-3000` |
| Improved PPO | `-347` |
| Reward-Shaped PPO | `-135` |
| PPO + NLP System | `-15` |

Generate the chart:

```bash
python scripts/plot_rewards.py
```

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Quick Gymnasium smoke test:

```bash
python - <<'PY'
import gymnasium as gym
import env

e = gym.make("OpenRescueAI-v0")
obs, info = e.reset(seed=42)
obs, reward, terminated, truncated, info = e.step(1)
print(obs.shape, reward, terminated, truncated, info["step"])
PY
```

## Run The Demo

The interactive Pygame demo loads the trained PPO model from `models/best_model.zip` and the Hugging Face zero-shot classifier.

```bash
python main.py
```

## Train PPO

```bash
python scripts/train_ppo.py --timesteps 200000 --seed 42 --output models/best_model
```

The script uses `gym.make("OpenRescueAI-v0")`, `FlattenObservation`, `Monitor`, and Stable-Baselines3 PPO.

## Evaluate PPO

```bash
python scripts/evaluate.py --model models/best_model.zip --episodes 10 --seed 42
```

This writes `plots/evaluation.json` with per-episode returns, saved civilians, and discovered messages.

## OpenEnv API

Run the OpenEnv/Hugging Face Space server:

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

Endpoints:

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | `GET` | Health check |
| `/reset` | `POST` | Reset the environment |
| `/step` | `POST` | Step with JSON body `{"action": 1}` |
| `/state` | `GET` | Current observation, info, termination flags, return |
| `/tasks` | `GET` | OpenEnv task metadata |
| `/docs` | `GET` | FastAPI docs |
| `/web` | `GET` | Minimal browser UI |

## Hugging Face Spaces Deployment

This repository is deployment-ready as a Docker Space.

```bash
huggingface-cli login
openenv push --repo-id <your-username>/openrescue-ai
```

Manual Docker run:

```bash
docker build -t openrescue-ai .
docker run -p 7860:7860 openrescue-ai
```

The `openenv.yaml` manifest declares the environment ID, task, action space, observation space, reward function, API endpoints, artifacts, and reward results.

## Submission Checklist

- OpenEnv manifest: `openenv.yaml`
- Gymnasium environment registration: `OpenRescueAI-v0`
- HF Space entrypoint: `app.py`
- Docker deployment: `Dockerfile`
- PPO training script: `scripts/train_ppo.py`
- Evaluation script: `scripts/evaluate.py`
- Reward plot utility: `scripts/plot_rewards.py`
- Reward evidence: `plots/reward_curve.png`
- Trained model artifact: `models/best_model.zip`
- Demo path with Hugging Face NLP override: `main.py`
- Mini-blog: `BLOG.md`
