#!/usr/bin/env python3
"""
Train activation oracle on reasoning step activations.
Adapted from activation_oracles for single-GPU training on RTX 3090.
"""

import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
import wandb

# Paths
DATA_DIR = Path("/workspace/activation-oracles-reasoning/experiments/3_oracle_training/results")
OUTPUT_DIR = DATA_DIR / "oracle_model"


@dataclass
class OracleTrainingConfig:
    """Training configuration for reasoning oracle."""
    model_name: str = "deepseek-ai/deepseek-r1-distill-llama-8b"

    # LoRA config
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"

    # Training
    num_epochs: int = 1
    batch_size: int = 4  # Small for RTX 3090 24GB
    gradient_accumulation_steps: int = 4  # Effective batch = 16
    lr: float = 1e-5
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1

    # Memory optimization
    gradient_checkpointing: bool = True
    torch_dtype: str = "bfloat16"

    # Evaluation
    eval_steps: int = 100

    # Logging
    wandb_project: str = "reasoning-oracle"
    wandb_run_name: str = "deepseek-r1-distill-llama-8b-reasoning"

    # Hooks
    hook_onto_layer: int = 1  # Layer for steering


def get_hf_submodule(model, layer_idx: int):
    """Get the submodule to hook onto."""
    # For most models, we hook at embedding output
    if hasattr(model, "model"):
        return model.model.embed_tokens
    elif hasattr(model, "transformer"):
        return model.transformer.wte
    else:
        raise ValueError(f"Unknown model architecture: {type(model)}")


def load_training_data():
    """Load prepared training data."""
    train_path = DATA_DIR / "oracle_train_data.pt"
    eval_path = DATA_DIR / "oracle_eval_data.pt"

    print(f"Loading training data from {train_path}")
    train_data = torch.load(train_path, weights_only=False)
    print(f"Loading eval data from {eval_path}")
    eval_data = torch.load(eval_path, weights_only=False)

    return train_data["data"], eval_data["data"]


def construct_batch(batch_data: list[dict], tokenizer, device):
    """Construct a training batch."""
    max_length = max(len(dp["input_ids"]) for dp in batch_data)

    batch_tokens = []
    batch_labels = []
    batch_attn_masks = []
    batch_steering_vectors = []
    batch_positions = []

    pad_token_id = tokenizer.pad_token_id

    for dp in batch_data:
        input_ids = dp["input_ids"]
        labels = dp["labels"]
        steering_vectors = dp["steering_vectors"]
        positions = dp["positions"]

        # Pad to max_length
        padding_length = max_length - len(input_ids)
        padded_input_ids = input_ids + [pad_token_id] * padding_length
        padded_labels = labels + [-100] * padding_length
        attn_mask = [1] * len(input_ids) + [0] * padding_length

        batch_tokens.append(padded_input_ids)
        batch_labels.append(padded_labels)
        batch_attn_masks.append(attn_mask)
        batch_steering_vectors.append(steering_vectors)
        batch_positions.append(positions)

    return {
        "input_ids": torch.tensor(batch_tokens, device=device),
        "labels": torch.tensor(batch_labels, device=device),
        "attention_mask": torch.tensor(batch_attn_masks, device=device),
        "steering_vectors": batch_steering_vectors,
        "positions": batch_positions,
    }


class SteeringHook:
    """Hook to inject activations at specified positions."""

    def __init__(self, positions: list[list[int]], steering_vectors: list[torch.Tensor], device, dtype):
        self.positions = positions
        self.steering_vectors = [sv.to(device=device, dtype=dtype) for sv in steering_vectors]
        self.handle = None

    def hook_fn(self, module, input, output):
        # output shape: [batch, seq_len, hidden_dim]
        # Clone to avoid in-place modification of leaf variable
        output = output.clone()
        for batch_idx, (pos_list, sv) in enumerate(zip(self.positions, self.steering_vectors)):
            for pos_idx, pos in enumerate(pos_list):
                if pos < output.shape[1]:
                    output[batch_idx, pos] = sv[pos_idx]
        return output

    def register(self, submodule):
        self.handle = submodule.register_forward_hook(self.hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()


def train_batch(batch, model, submodule, device, dtype):
    """Train on a single batch."""
    # Create steering hook
    hook = SteeringHook(
        positions=batch["positions"],
        steering_vectors=batch["steering_vectors"],
        device=device,
        dtype=dtype
    )
    hook.register(submodule)

    try:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
    finally:
        hook.remove()

    return loss


def evaluate(eval_data, model, submodule, tokenizer, device, dtype, cfg, max_samples=100):
    """Evaluate on held-out data."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    eval_subset = eval_data[:max_samples]

    with torch.no_grad():
        for start in range(0, len(eval_subset), cfg.batch_size):
            batch_data = eval_subset[start:start + cfg.batch_size]
            batch = construct_batch(batch_data, tokenizer, device)
            loss = train_batch(batch, model, submodule, device, dtype)
            total_loss += loss.item()
            n_batches += 1

    model.train()
    return total_loss / max(n_batches, 1)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Support for timing test - set MAX_STEPS env var to limit training
    max_steps = int(os.environ.get("MAX_STEPS", 0))  # 0 = no limit

    print("=" * 60)
    print("ORACLE TRAINING")
    if max_steps > 0:
        print(f"TIMING TEST MODE: Will stop after {max_steps} steps")
    print("=" * 60)

    cfg = OracleTrainingConfig()
    print(f"\nConfig: {asdict(cfg)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if cfg.torch_dtype == "bfloat16" else torch.float16

    # Load data
    print("\n1. Loading data...")
    train_data, eval_data = load_training_data()
    print(f"   Train: {len(train_data)}, Eval: {len(eval_data)}")

    # Load model
    print("\n2. Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Enable gradient checkpointing
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # Get submodule for hooks
    submodule = get_hf_submodule(model, cfg.hook_onto_layer)

    # Apply LoRA
    print("\n3. Applying LoRA...")
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model.enable_input_require_grads()

    # Setup optimizer and scheduler
    steps_per_epoch = len(train_data) // (cfg.batch_size * cfg.gradient_accumulation_steps)
    total_steps = steps_per_epoch * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"\n4. Training setup:")
    print(f"   Steps per epoch: {steps_per_epoch}")
    print(f"   Total steps: {total_steps}")
    print(f"   Warmup steps: {warmup_steps}")

    # Initialize wandb
    wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=asdict(cfg))

    # Training loop
    print("\n5. Training...")
    model.train()
    global_step = 0
    accumulated_loss = 0.0

    import time
    training_start = time.time()
    step_times = []

    for epoch in range(cfg.num_epochs):
        optimizer.zero_grad()

        pbar = tqdm(range(0, len(train_data), cfg.batch_size), desc=f"Epoch {epoch + 1}")
        for step_idx, start in enumerate(pbar):
            batch_data = train_data[start:start + cfg.batch_size]
            batch = construct_batch(batch_data, tokenizer, device)

            loss = train_batch(batch, model, submodule, device, dtype)
            loss = loss / cfg.gradient_accumulation_steps
            loss.backward()
            accumulated_loss += loss.item()

            is_update_step = (step_idx + 1) % cfg.gradient_accumulation_steps == 0

            if is_update_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                wandb.log({
                    "train/loss": accumulated_loss,
                    "train/lr": scheduler.get_last_lr()[0],
                }, step=global_step)

                pbar.set_postfix({"loss": f"{accumulated_loss:.4f}"})

                # Evaluation
                if global_step > 0 and global_step % cfg.eval_steps == 0:
                    eval_loss = evaluate(eval_data, model, submodule, tokenizer, device, dtype, cfg)
                    wandb.log({"eval/loss": eval_loss}, step=global_step)
                    print(f"\n   Step {global_step}: eval_loss = {eval_loss:.4f}")

                # Track timing
                step_time = time.time() - training_start
                step_times.append(step_time)
                training_start = time.time()

                global_step += 1
                accumulated_loss = 0.0

                # Timing test: stop early if MAX_STEPS set
                if max_steps > 0 and global_step >= max_steps:
                    avg_step_time = sum(step_times) / len(step_times)
                    total_steps = len(train_data) // (cfg.batch_size * cfg.gradient_accumulation_steps)
                    eta_minutes = (avg_step_time * total_steps) / 60
                    print(f"\n   TIMING TEST: Stopping after {max_steps} steps")
                    print(f"   Average time per step: {avg_step_time:.2f} seconds")
                    print(f"   Total steps for full epoch: {total_steps}")
                    print(f"   Estimated full training time: {eta_minutes:.1f} minutes ({eta_minutes/60:.1f} hours)")
                    break

        # Break outer loop too if timing test
        if max_steps > 0 and global_step >= max_steps:
            break

    # Final evaluation
    print("\n6. Final evaluation...")
    eval_loss = evaluate(eval_data, model, submodule, tokenizer, device, dtype, cfg)
    print(f"   Final eval loss: {eval_loss:.4f}")
    wandb.log({"eval/final_loss": eval_loss}, step=global_step)

    # Save model
    print("\n7. Saving model...")
    model.save_pretrained(OUTPUT_DIR / "final")
    tokenizer.save_pretrained(OUTPUT_DIR / "final")

    # Save config
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    wandb.finish()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Model saved to: {OUTPUT_DIR / 'final'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
