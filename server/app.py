from fastapi import FastAPI, Query
from .env import VogenEnv
from .schemas import Observation, Action, StepResult, Reward
import yaml
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

app = FastAPI()

env = VogenEnv()

@app.post("/reset")
async def reset(task_spec: dict, session_id: str = Query(...)) -> Dict[str, Any]:
    obs = await env.reset(task_spec, session_id)
    return {"observation": obs.model_dump(), "session_id": session_id}

@app.post("/step")
async def step(action: Action, session_id: str = Query(...)) -> Dict[str, Any]:
    result = await env.step(action, session_id)
    return result.model_dump()

@app.get("/state")
async def state(session_id: str = Query(...)) -> dict:
    return await env.state(session_id)

@app.post("/close")
async def close(session_id: str = Query(...)) -> None:
    await env.close(session_id)

@app.post("/score")
async def score(traj: List[Dict[str, Any]], session_id: str = Query(...)) -> Dict[str, Any]:
    reward = await env.score(traj, session_id)
    return {"reward": reward.model_dump()}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/openenv/manifest")
async def manifest():
    with open("openenv.yaml") as f:
        return yaml.safe_load(f)

@app.post("/tools/vogen.style")
async def vogen_style(params: Dict[str, Any], session_id: str = Query(...)) -> Dict[str, Any]:
    # Stub tool
    return {"result": "styled", "session_id": session_id}

@app.post("/tools/vogen.negotiate")
async def vogen_negotiate(params: Dict[str, Any], session_id: str = Query(...)) -> Dict[str, Any]:
    # Stub tool
    return {"result": "negotiated", "session_id": session_id}

@app.post("/tools/vogen.predict")
async def vogen_predict(params: Dict[str, Any], session_id: str = Query(...)) -> Dict[str, Any]:
    # Stub tool
    return {"result": "predicted", "session_id": session_id}

@app.post("/tools/vogen.evolve")
async def vogen_evolve(params: Dict[str, Any], session_id: str = Query(...)) -> Dict[str, Any]:
    # Stub tool, long running
    return {"result": "evolved", "session_id": session_id}