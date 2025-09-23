#!/usr/bin/env python3
"""
Realistic GPU Memory Usage Simulation
Simulates training with gradient_accumulation_steps=16, seq_len=640, without gradient checkpointing.
Generates a clean summary at the end for reporting.
"""

import os
import gc
import time
import torch
import random
from typing import List, Tuple
from modeling_lexicon_compressor import LexiconCompressorModel

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def create_sample_dictionary(size: int = 500) -> List[List[int]]:
    """Create a dummy dictionary with random token sequences."""
    dictionary = []
    for i in range(size):
        length = random.randint(3, 15)
        tokens = [random.randint(1, 50000) for _ in range(length)]
        dictionary.append(tokens)
    return dictionary


def pad_row_indices(row_indices_per_sample: List[List[int]]) -> torch.Tensor:
    """Pad row indices to the same length for batching."""
    if not row_indices_per_sample:
        return torch.empty(0, 0, dtype=torch.long)

    max_len = max(len(indices) for indices in row_indices_per_sample)
    batch_size = len(row_indices_per_sample)

    padded = torch.full((batch_size, max_len), -1, dtype=torch.long)
    for i, indices in enumerate(row_indices_per_sample):
        if len(indices) > 0:
            padded[i, :len(indices)] = torch.tensor(indices, dtype=torch.long)

    return padded


def get_gpu_memory_usage() -> Tuple[float, float]:
    """Return current allocated and reserved GPU memory (GB)."""
    if not torch.cuda.is_available():
        return 0.0, 0.0

    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    return allocated, reserved


def log_peak_memory(stage: str, last_peak: List[float], peaks: List[Tuple[str, float]]):
    """Record memory peak updates for later summary."""
    current_peak = torch.cuda.max_memory_allocated() / 1024**3
    if current_peak > last_peak[0]:
        print(f"  [Peak Updated] {stage}: {current_peak:.2f} GB (prev {last_peak[0]:.2f} GB)")
        last_peak[0] = current_peak
        peaks.append((stage, current_peak))


def simulate_real_training_step(
    model: LexiconCompressorModel,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    row_indices_padded: torch.Tensor,
    step: int,
    accumulation_steps: int,
    last_peak: List[float],
    peaks: List[Tuple[str, float]]
) -> float:
    """Simulate one training step with gradient accumulation."""
    # Forward
    outputs = model(
        qwen_input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        row_indices_per_sample=row_indices_padded,
    )
    log_peak_memory(f"Step {step+1} Forward", last_peak, peaks)

    loss = outputs.loss / accumulation_steps
    print(f"  Step {step + 1}/{accumulation_steps}: Loss = {loss.item() * accumulation_steps:.4f}")

    # Backward
    loss.backward()
    log_peak_memory(f"Step {step+1} Backward", last_peak, peaks)

    allocated, reserved = get_gpu_memory_usage()
    print(f"  Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

    return loss.item() * accumulation_steps


def test_realistic_memory_usage(
    model_name: str,
    per_device_batch_size: int = 1,
    accumulation_steps: int = 16,
    base_seq_length: int = 512,
    dictionary_size: int = 500,
    compress_tokens: int = 4,
    rows_per_sample: int = 32
):
    """Test realistic GPU memory usage with the given config."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not torch.cuda.is_available():
        print("⚠️ CUDA not available, running on CPU.")
        return

    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {torch.cuda.get_device_name(0)} ({total_memory:.1f} GB total)")

    last_peak = [0.0]
    peaks: List[Tuple[str, float]] = []

    try:
        # Step 1: Clear memory
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        log_peak_memory("Initialization", last_peak, peaks)

        # Step 2: Load model
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        log_peak_memory("Base Model Loaded", last_peak, peaks)

        dictionary = create_sample_dictionary(dictionary_size)
        model = LexiconCompressorModel(
            qwen_model=base_model,
            full_dict=dictionary,
            dict_encoder_num_compress_tokens=compress_tokens,
            dict_encoder_learned_tokens_prepend=True,
        ).to(device)

        model.qwen.gradient_checkpointing_enable()
        log_peak_memory("Full Model Ready", last_peak, peaks)

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

        # Step 3: Prepare data
        sample_text = "This is a sample text for memory testing. " * (base_seq_length // 10)
        inputs = tokenizer(
            [sample_text] * per_device_batch_size,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=base_seq_length
        )
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        labels = input_ids.clone()

        row_indices_per_sample = [random.sample(range(len(dictionary)), rows_per_sample)
                                  for _ in range(per_device_batch_size)]
        row_indices_padded = pad_row_indices(row_indices_per_sample).to(device)
        log_peak_memory("Data Prepared", last_peak, peaks)

        # Step 4: Training simulation
        model.train()
        total_loss = 0.0
        for step in range(accumulation_steps):
            loss = simulate_real_training_step(
                model, optimizer, input_ids, attention_mask, labels,
                row_indices_padded, step, accumulation_steps, last_peak, peaks
            )
            total_loss += loss

        # Step 5: Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        log_peak_memory("Optimizer Step", last_peak, peaks)

        # Final stats
        allocated, reserved = get_gpu_memory_usage()
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3

        # === Final Summary ===
        print("\n" + "="*80)
        print("📊 GPU Memory Usage Summary")
        print(f"- Model: {model_name}")
        print(f"- Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"- Max Peak Memory: {peak_memory:.2f} GB")
        print(f"- Final Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
        print("\nKey Peak Events:")
        for stage, mem in peaks:
            print(f"  • {stage}: {mem:.2f} GB")
        print("="*80 + "\n")

    except torch.cuda.OutOfMemoryError as e:
        print(f"\n❌ Out of Memory: {e}")
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    configs = [
        {
            "model_name": "Qwen/Qwen3-0.6B",
            "per_device_batch_size": 1,
            "accumulation_steps": 16,
            "base_seq_length": 512,
            "dictionary_size": 500,
            "compress_tokens": 4,
            "rows_per_sample": 32,
        }
    ]

    for config in configs:
        test_realistic_memory_usage(**config)


if __name__ == "__main__":
    main()
