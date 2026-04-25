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
        history = []
        for _ in range(max_steps):
            prompt = f"Task: {obs['brief']['text']}\nWardrobe: {obs['wardrobe_handle']}\nAction:"
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(**inputs, max_length=100, do_sample=True)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Parse action (stub)
            action = {"garment_ids": ["garment_0"], "justification": response, "self_predicted_score": 0.5}
            result = env_client.step(action)
            reward_dict = result['reward_dict']
            trajectories.append((prompt, response, reward_dict))
            if result['done']:
                break
            obs = result['observation']
    return trajectories