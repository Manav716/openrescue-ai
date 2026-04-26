from gymnasium.envs.registration import register

from env.disaster_env import DisasterEnv

ENV_ID = "OpenRescueAI-v0"

try:
    register(
        id=ENV_ID,
        entry_point="env.disaster_env:DisasterEnv",
        max_episode_steps=200,
    )
except Exception:
    # Gymnasium raises when modules are reloaded in notebooks.
    pass

__all__ = ["DisasterEnv", "ENV_ID"]
