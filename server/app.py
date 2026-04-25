from fastapi import FastAPI
from .env import VogenEnv
from .schemas import Observation, Action, StepResult
import yaml
import logging

logger = logging.getLogger(__name__)

app = FastAPI()

env = VogenEnv()

@app.post("/reset")
async def reset(task_spec: dict) -> Observation:
    return await env.reset(task_spec)

@app.post("/step")
async def step(action: Action) -> StepResult:
    return await env.step(action)

@app.get("/state")
async def state() -> dict:
    return await env.state()

@app.post("/close")
async def close() -> None:
    await env.close()

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/openenv/manifest")
async def manifest():
    with open("openenv.yaml") as f:
        return yaml.safe_load(f)