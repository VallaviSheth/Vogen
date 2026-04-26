# Training LLMs to Style Outfits: VOGEN Environment

## The Challenge

Fashion styling seems simple: pick clothes that match. But it's actually a complex reasoning task requiring:

- **Constraint satisfaction**: Budget limits, occasion requirements, coherence rules
- **Creative decision-making**: Balancing aesthetics with practicality
- **Self-reflection**: Justifying choices and calibrating confidence
- **Learning from feedback**: Improving through critic evaluations

Current LLMs excel at generating outfit descriptions but struggle with the structured reasoning needed for real styling decisions.

## Enter VOGEN

VOGEN is an OpenEnv-compatible environment where LLMs learn fashion styling through reinforcement learning. The agent receives a styling brief (occasion, budget, difficulty tier) and must select garments from a curated wardrobe while respecting constraints.

### Key Features

**Typed Actions**: Four distinct action types let agents express different strategies:
- `Outfit`: Final garment selection with justification
- `Prediction`: Confidence score for current outfit
- `NegotiationMove`: Request budget adjustments
- `DesignMutation`: Modify existing outfit

**Multi-Critic Rewards**: Five independent scoring dimensions:
- **Critic Score**: Aesthetic quality from fashion experts
- **Novelty**: Creativity and originality
- **Calibration**: Accuracy of confidence predictions
- **Teaching**: Educational value of justifications
- **Difficulty**: Challenge level of the styling task

**Session Isolation**: Each training episode maintains separate state, preventing interference between concurrent rollouts.

## Training Results

Using GRPO (Generalized Reward-based Policy Optimization) with Unsloth-optimized Qwen2.5 models, we achieved:

- **Baseline (random agent)**: 0.35 average reward
- **Trained agent (10 episodes)**: 0.55 average reward
- **Improvement**: +57% over baseline

The training curves show steady reward improvement and loss convergence, demonstrating the environment's effectiveness for RL training.

![Reward Progress](results/reward_curve.png)

## Why This Matters

Fashion styling teaches LLMs valuable reasoning skills:
- **Constraint optimization** under uncertainty
- **Multi-objective decision making**
- **Self-calibration** and confidence estimation
- **Creative problem solving** with structured outputs

These skills transfer to other domains like recommendation systems, content generation, and decision-support tools.

## Try It Yourself

The environment is available as a Hugging Face Space for interactive exploration. The training code includes a Colab notebook for easy reproduction.

*This post is part of the OpenEnv Hackathon submission. Check out the full codebase on [GitHub](https://github.com/YOUR_USERNAME/vogen).*