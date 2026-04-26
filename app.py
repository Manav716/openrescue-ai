from typing import Any

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from env import DisasterEnv
from env.constants import ACTION_NAMES


class StepRequest(BaseModel):
    action: int = Field(..., ge=0, le=3, description="Discrete rescue-agent action: 0 up, 1 down, 2 left, 3 right.")


class ResetRequest(BaseModel):
    seed: int | None = Field(default=None, description="Optional Gymnasium RNG seed.")


app = FastAPI(
    title="OpenRescue AI",
    description="OpenEnv-compatible multi-agent disaster response environment.",
    version="1.0.0",
)

_env = DisasterEnv()
_last_observation: np.ndarray | None = None
_last_info: dict[str, Any] = {}
_terminated = False
_truncated = False
_episode_return = 0.0


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    return value


def _state_payload() -> dict[str, Any]:
    return {
        "observation": _jsonify(_last_observation),
        "info": _jsonify(_last_info),
        "terminated": _terminated,
        "truncated": _truncated,
        "episode_return": _episode_return,
        "action_space": {"type": "discrete", "n": 4, "actions": ACTION_NAMES},
    }


@app.on_event("startup")
def startup() -> None:
    reset_environment(ResetRequest())


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "name": "OpenRescue AI",
        "status": "ok",
        "endpoints": ["/health", "/reset", "/step", "/state", "/tasks", "/docs", "/web"],
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/reset")
def reset_environment(request: ResetRequest) -> dict[str, Any]:
    global _last_observation, _last_info, _terminated, _truncated, _episode_return

    _last_observation, _last_info = _env.reset(seed=request.seed)
    _terminated = False
    _truncated = False
    _episode_return = 0.0
    return _state_payload()


@app.post("/step")
def step_environment(request: StepRequest) -> dict[str, Any]:
    global _last_observation, _last_info, _terminated, _truncated, _episode_return

    if _terminated or _truncated:
        _last_observation, _last_info = _env.reset()
        _terminated = False
        _truncated = False
        _episode_return = 0.0

    _last_observation, reward, _terminated, _truncated, _last_info = _env.step(request.action)
    _episode_return += float(reward)
    payload = _state_payload()
    payload["reward"] = float(reward)
    return payload


@app.get("/state")
def get_state() -> dict[str, Any]:
    return _state_payload()


@app.get("/tasks")
def get_tasks() -> list[dict[str, Any]]:
    return [
        {
            "id": "openrescue-disaster-response",
            "name": "Neuro-symbolic disaster rescue",
            "description": "Navigate partial observations, avoid spreading fire, and rescue civilians using shared drone memory.",
            "difficulty": "medium",
            "max_steps": _env.max_steps,
            "reward_range": [-200.0, 500.0],
        }
    ]


@app.get("/web", response_class=HTMLResponse)
def web() -> str:
    return """
    <!doctype html>
    <html>
      <head>
        <title>OpenRescue AI</title>
        <style>
          body { font-family: system-ui, sans-serif; margin: 2rem; background: #f7f7f4; color: #171717; }
          main { max-width: 880px; margin: auto; }
          button { margin-right: .5rem; padding: .55rem .8rem; border: 1px solid #444; background: white; cursor: pointer; }
          pre { background: #111; color: #f5f5f5; padding: 1rem; overflow: auto; min-height: 320px; }
        </style>
      </head>
      <body>
        <main>
          <h1>OpenRescue AI</h1>
          <p>Use the controls to step the OpenEnv-compatible disaster-response environment.</p>
          <button onclick="resetEnv()">Reset</button>
          <button onclick="stepEnv(0)">Up</button>
          <button onclick="stepEnv(1)">Down</button>
          <button onclick="stepEnv(2)">Left</button>
          <button onclick="stepEnv(3)">Right</button>
          <pre id="out">Loading...</pre>
        </main>
        <script>
          async function show(response) {
            document.getElementById('out').textContent = JSON.stringify(await response.json(), null, 2);
          }
          async function resetEnv() { show(await fetch('/reset', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'})); }
          async function stepEnv(action) { show(await fetch('/step', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({action})})); }
          resetEnv();
        </script>
      </body>
    </html>
    """

