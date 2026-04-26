# OpenRescue AI: Building A Neuro-Symbolic Disaster Response Environment

OpenRescue AI started as a compact question: can a reinforcement learning agent coordinate with a semantic scout in a dangerous, partially observed world?

The environment is a 10x10 disaster grid with civilians, spreading fire, and fog of war. The rescue unit is trained with PPO, while an autonomous drone scouts nearby cells and writes natural-language alerts into a shared communication stream. Those messages are interpreted by a Hugging Face transformer classifier in the demo. When the system identifies a civilian report as a critical rescue target, a symbolic override extracts the coordinates and temporarily routes the rescue agent toward the target.

This produces a deliberately hybrid system. PPO handles spatial decision-making under uncertainty. The drone provides broader situational awareness. The transformer layer turns raw map discoveries into semantic instructions. The override mechanism gives the system an interpretable emergency path when language says a rescue target matters.

The most important improvement was reward shaping. A purely random agent collapsed around `-5000`, early PPO reached around `-3000`, and the improved reward-shaped policy reached `-135`. With the communication and NLP override path, the final system reached approximately `-15`, turning the environment from survival-only behavior into a credible rescue workflow.

For the OpenEnv submission, the project was hardened rather than redesigned. The working Gymnasium environment is still intact, but it now has a proper manifest, environment registration, HTTP endpoints, reproducible train/evaluate scripts, Docker packaging, and Hugging Face Spaces readiness.

