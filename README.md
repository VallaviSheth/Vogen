---
title: vogen
emoji: "👗"
colorFrom: "#6D28D9"
colorTo: "#EC4899"
sdk: fastapi
app_port: 8000
tags: ["openenv", "fashion", "rl"]
---

# VOGEN

OpenEnv fashion styling environment with tiered briefs, critic-based reward components, and a FastAPI-backed app.

## Motivation

VOGEN models subjective styling decisions as a constrained fashion environment. Agents select garment IDs, justify a look, and self-predict their score while managing budget and cultural-context constraints. The codebase is built around staged briefs, critic scoring, and rule-based validation for reward-safe training.

## Environment Overview

| Domain | Observation type | Action type | Reward shape | Number of tasks | Grader type |
|---|---|---|---|---|---|
| Fashion styling | `server.schemas.Observation` | `server.schemas.Action` | Dict of rubric values | Tiered briefs (1–6) | `server.critics.Critic` / deterministic fallback |

## Tasks

The environment uses a unified styling task in `server/task.py`.

### Tier 1 — Foundations
- `brief.text`: "Style a simple outfit for a casual occasion."
- Constraints: none
- Budget: `100.0`
- Validation: garment IDs must exist, outfit size 1–5, self-predicted score in [0, 1]

### Tier 2 — Constrained Styling
- `brief.text`: "Style an outfit under budget constraints."
- Constraints: `{"budget_strict": True}`
- Budget: `50.0`
- Validation: budget check uses 20 per garment stub cost

### Tier 3 — Trend-Aware Styling
- `brief.text`: "Style an outfit that trends with current culture."
- Constraints: `{"trend_aware": True}`
- Budget: `150.0`
- Validation: justification must cite a `garment_<i>` and `dim_<i>` token

### Tier 4 — Adversarial Styling
- `brief.text`: "Style an outfit with adversarial constraints."
- Constraints: `{"adversarial": True}`
- Budget: `200.0`
- Validation: same action checks, plus adversarial challenge placeholder

### Tier 5 — Iterative Refinement
- Terminal after 3 actions (`Task.is_terminal`)
- Target: sequence-based refinement on repeated briefs

### Tier 6 — Open Atelier
- Terminal after 5 actions (`Task.is_terminal`)
- Target: sustained styling under longer episodes and drift

## Action Space

| Field | Type | Range / Meaning |
|---|---|---|
| `garment_ids` | `list[str]` | Selected wardrobe IDs from `wardrobe_handle` |
| `justification` | `str` | Text explanation of the outfit |
| `self_predicted_score` | `float` | Prediction of critic score in [0.0, 1.0] |

## Observation Space

```python
class Observation(BaseModel):
    brief: Brief
    wardrobe_handle: List[str]
    context_vector: List[float]
    history: List[Dict[str, Any]]
```

`Brief` includes `text`, `constraints`, `occasion`, `budget`, and `tier`.

## Reward Function

| Component | Sign | Trigger | Source |
|---|---|---|---|
| `critic_score` | positive | critic evaluation output | `training/reward_aggregator.py` |
| `justification` | positive | justification quality model | `training/reward_aggregator.py` |
| `novelty` | positive | action novelty score | `training/reward_aggregator.py` |
| `difficulty` | positive | tier multiplier | `training/reward_aggregator.py` |
| `calibration` | positive | self prediction vs critic | `training/reward_aggregator.py` |

The training reward is dense: the aggregator computes a weighted sum of rubric scores each step.

## Setup

```bash
python -m pip install -r requirements.txt
```

`pyproject.toml` contains the same pinned dependencies.

TODO: No Dockerfile exists in this repository.

TODO: No Hugging Face Spaces deploy script exists in this repository.

## Usage

### Run the FastAPI app

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### HTTP API

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/reset -H 'Content-Type: application/json' -d '{"seed": 42}'
curl -X POST http://localhost:8000/step -H 'Content-Type: application/json' -d '{"garment_ids": ["garment_0"], "justification": "garment_0", "self_predicted_score": 0.5}'
```

### Python client

```python
from client.vogen_client import VogenClient
client = VogenClient.from_url('http://localhost:8000')
obs = client.reset({'seed': 42})
step = client.step({'garment_ids': ['garment_0'], 'justification': 'garment_0', 'self_predicted_score': 0.5})
```

### Training CLI

```bash
python -m training.train_vogen --config training/configs/grpo_default.yaml --smoke
```

## Baseline scores

| Task | Baseline | Score |
|---|---|---|
| Tiered styling | Random / zero-shot | TBD |

## Project Structure

```
README.md
openenv.yaml
pyproject.toml
requirements.txt
LICENSE
REFERENCE_README.md
server/
  app.py
  env.py
  schemas.py
  task.py
  critics.py
  runway.py
  curator.py
  safety/
    anticheat.py
    sandbox.py
  rubrics/
    critic_score.py
    justification.py
    novelty.py
    difficulty.py
    calibration.py
client/
  vogen_client.py
training/
  train_vogen.py
  reward_aggregator.py
  rollout.py
  configs/
    grpo_default.yaml
    dpo_default.yaml
eval/
  __init__.py
tests/
  test_env_contract.py
  test_critics.py
  test_curriculum.py
  test_rubrics.py
  test_anticheat.py
  test_runway.py
```

## OpenEnv spec compliance

- [x] `reset` implemented in `server.env.VogenEnv`
- [x] `step` implemented in `server.env.VogenEnv`
- [x] `state` implemented in `server.env.VogenEnv`
- [x] `close` implemented in `server.env.VogenEnv`
- [ ] `task.grade` not implemented; `Task.score_action` is present instead
- [x] Typed Pydantic models in `server.schemas`
- [x] `openenv.yaml` present with entrypoint and rubric IDs

## License

Licensed under the Apache License, Version 2.0.
