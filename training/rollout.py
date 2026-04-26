from typing import List, Tuple, Dict, Any

def rollout(model, tokenizer, env_client, n_episodes: int, max_steps: int) -> List[Tuple[str, str, Dict[str, float]]]:
    """Generate rollouts for training.

    Args:
        model: The model.
        tokenizer: Tokenizer.
        env_client: Environment client.
        n_episodes: Number of episodes.
        max_steps: Max steps per episode.

    Returns:
        List of (prompt, response, reward_dict).
    """
    trajectories = []
    for _ in range(n_episodes):
        obs = env_client.reset({"seed": 42})
        traj = []
        for _ in range(max_steps):
            prompt = f"Task: {obs.brief.text}\nWardrobe: {obs.wardrobe_handle}\nAction:"
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(**inputs, max_length=100, do_sample=True)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Parse action (stub)
            from client.models import Outfit
            action = Outfit(garment_ids=["garment_0"], justification=response, self_predicted_score=0.5)
            obs, reward, done, info = env_client.step(action)
            traj.append({"obs": obs.model_dump(), "action": action.model_dump(), "reward": reward.model_dump(), "done": done, "info": info})
            if done:
                break
        # Score the trajectory
        reward_dict = env_client.score(traj).model_dump()
        trajectories.append((prompt, response, reward_dict))
    return trajectories