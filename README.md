# 🚁 OpenRescue AI: Neuro-Symbolic Multi-Agent Disaster Response

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Reinforcement Learning](https://img.shields.io/badge/RL-Stable_Baselines3-orange)
![NLP](https://img.shields.io/badge/NLP-Hugging_Face-yellow)
![Environment](https://img.shields.io/badge/Sim-Gymnasium-lightgrey)

## 🚀 Overview
OpenRescue AI is an advanced Multi-Agent Reinforcement Learning (MARL) simulation designed for autonomous disaster response. Instead of relying purely on spatial pathfinding, this project utilizes **Neuro-Symbolic AI**—combining the spatial decision-making of Reinforcement Learning (PPO) with the semantic reasoning of Natural Language Processing (Transformers).

The simulation features two entities operating simultaneously in a Partially Observable Markov Decision Process (POMDP) environment:
1. **Autonomous Scout Drone:** Explores the "Fog of War," maps dynamically spreading hazards (fires), and generates natural language alerts based on its sensor data.
2. **Rescue Unit (PPO Agent):** An RL agent trained via Proximal Policy Optimization with dense reward shaping to navigate the shared memory map, avoid hazards, and extract civilians.

## 🧠 Core Architecture

* **The Physics Engine:** A custom grid-world built on `Gymnasium` featuring persistent memory, fog of war, and dynamic cellular automata (fire spreading). Modularly designed with separated `world.py` and `agents.py` logic.
* **The Spatial Brain:** A `stable-baselines3` PPO algorithm trained for 200,000 timesteps to optimize exploration and hazard avoidance.
* **The Semantic Brain (Neuro-Symbolic Override):** A local Hugging Face Zero-Shot Classification Transformer (`facebook/bart-large-mnli`) intercepts the drone's text streams in real-time. If it classifies a message as a *Critical Rescue Target*, it uses Regular Expressions to extract coordinates and **symbolically overrides the RL agent**, forcing an immediate, autonomous rescue execution.

## 📊 Training Results
By implementing strategic reward shaping (exploration bonuses, time-completion bonuses, and hazard penalties), the agent successfully learned to map the environment rather than just survive.

| Model | Setup | Final Score |
| :--- | :--- | :--- |
| Random Agent | No Training | -5000+ |
| PPO (Base) | 10k Steps | -347 |
| **PPO + NLP (Final)** | **200k Steps** | **-15 (Mission Success)** |

## 💻 Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/openrescue-ai.git](https://github.com/yourusername/openrescue-ai.git)
cd openrescue-ai