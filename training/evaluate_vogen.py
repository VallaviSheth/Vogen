import argparse
import os
import yaml
import asyncio
from typing import Any, Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from client.vogen_client import VogenClient
from client.models import Outfit, Reward
from server.env import VogenEnv

DEFAULT_CONFIG = "training/configs/grpo_default.yaml"


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model(model_name: str, smoke: bool):
    if smoke or model_name is None:
        model_name = model_name or "Qwen/Qwen2.5-0.5B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(model_name, load_in_4bit=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def build_prompt(obs: Dict[str, Any]) -> str:
    brief = obs["brief"]["text"] if isinstance(obs, dict) else obs.brief.text
    wardrobe = obs["wardrobe_handle"] if isinstance(obs, dict) else obs.wardrobe_handle
    return f"Task: {brief}\nWardrobe: {wardrobe}\nAction:"


def sample_action(model, tokenizer, obs: Dict[str, Any]) -> Outfit:
    prompt = build_prompt(obs)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return Outfit(garment_ids=["garment_0"], justification=response, self_predicted_score=0.5)


def compute_accuracy_like(reward: Reward) -> float:
    values = reward.model_dump().values()
    return sum(values) / len(values)


async def evaluate_local(model, tokenizer, episodes: int, max_steps: int) -> None:
    env = VogenEnv()
    totals = {"critic": 0.0, "novelty": 0.0, "calibration": 0.0, "teaching": 0.0, "difficulty": 0.0}
    accuracy_like = 0.0

    for episode in range(episodes):
        session_id = f"local-{episode}"
        obs = await env.reset({"seed": episode}, session_id)
        traj = []
        for _step in range(max_steps):
            action = sample_action(model, tokenizer, obs.model_dump())
            result = await env.step(action, session_id)
            traj.append({
                "obs": result.observation.model_dump(),
                "action": action.model_dump(),
                "reward": result.reward.model_dump(),
                "done": result.done,
                "info": result.info,
            })
            if result.done:
                break
        reward = await env.score(traj, session_id)
        for field, value in reward.model_dump().items():
            totals[field] += value
        accuracy_like += compute_accuracy_like(reward)

    avg_reward = {field: totals[field] / episodes for field in totals}
    print("=== VOGEN Local Evaluation ===")
    print(f"episodes={episodes} max_steps={max_steps}")
    print("average reward:")
    for field, value in avg_reward.items():
        print(f"  {field}: {value:.4f}")
    print(f"accuracy-like score: {accuracy_like / episodes:.4f}")


def evaluate_remote(model, tokenizer, url: str, episodes: int, max_steps: int) -> None:
    client = VogenClient.from_url(url)
    totals = {"critic": 0.0, "novelty": 0.0, "calibration": 0.0, "teaching": 0.0, "difficulty": 0.0}
    accuracy_like = 0.0

    for episode in range(episodes):
        obs = client.reset({"seed": episode})
        traj = []
        for _step in range(max_steps):
            action = sample_action(model, tokenizer, obs.model_dump() if hasattr(obs, 'model_dump') else obs)
            obs, reward, done, info = client.step(action)
            traj.append({
                "obs": obs.model_dump() if hasattr(obs, 'model_dump') else obs,
                "action": action.model_dump(),
                "reward": reward.model_dump(),
                "done": done,
                "info": info,
            })
            if done:
                break
        score = client.score(traj)
        for field, value in score.model_dump().items():
            totals[field] += value
        accuracy_like += compute_accuracy_like(score)

    avg_reward = {field: totals[field] / episodes for field in totals}
    print("=== VOGEN Remote Evaluation ===")
    print(f"url={url} episodes={episodes} max_steps={max_steps}")
    print("average reward:")
    for field, value in avg_reward.items():
        print(f"  {field}: {value:.4f}")
    print(f"accuracy-like score: {accuracy_like / episodes:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VOGEN on a held-out validation trajectory.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--local", action="store_true", help="Evaluate against a local VogenEnv instance instead of a running server.")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    model_name = config.get("model_name") if config else None
    model, tokenizer = load_model(model_name, args.smoke)

    if args.local:
        asyncio.run(evaluate_local(model, tokenizer, args.episodes, args.max_steps))
    else:
        evaluate_remote(model, tokenizer, args.url, args.episodes, args.max_steps)


if __name__ == "__main__":
    main()
