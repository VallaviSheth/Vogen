import argparse
import yaml
import os
from transformers import TrainingArguments
from .rollout import rollout
from .reward_aggregator import RewardAggregator

try:
    import wandb
except ImportError:
    wandb = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.smoke:
        config['model_name'] = "Qwen/Qwen2.5-0.5B-Instruct"
        config['max_steps'] = 10

    if wandb is not None:
        wandb_mode = "disabled" if args.smoke or os.getenv("WANDB_MODE") == "disabled" else os.getenv("WANDB_MODE", "online")
        wandb.init(project="vogen-hackathon", mode=wandb_mode)

    if args.smoke:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(config['model_name'])
        tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Stub training for smoke
        print("Smoke mode: loaded model, skipping training")
        return
    else:
        from unsloth import FastLanguageModel
        from trl import GRPOTrainer
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config['model_name'],
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(model, r=config['lora_r'], target_modules=config['lora_target_modules'])

    aggregator = RewardAggregator(config['reward_weights'])

    def reward_fn(trajectory):
        return aggregator.aggregate(trajectory)[0]

    # Stub dataset
    train_dataset = [{"dummy": "data"}]

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        reward_funcs=[reward_fn],
        args=TrainingArguments(
            output_dir="./results",
            num_train_epochs=1,
            per_device_train_batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            max_steps=config['max_steps'],
            logging_steps=1,
            save_steps=50,
        ),
    )

    trainer.train()

    # Save
    model.save_pretrained_merged("merged_model", save_method="merged_4bit")

if __name__ == "__main__":
    main()