import os
import json
import csv
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    Qwen3ForCausalLM,
    Trainer,
    TrainingArguments,
    PreTrainedModel,
)
from configuration_lexicon_compressor import LexiconCompressorConfig
from modeling_lexicon_compressor import LexiconCompressorModel, LexiconCompressorModelOutput
from tokenization_lexicon_compressor import LexiconCompressorTokenizor


# ==============================
# 1) HF wrapper (inject RCA weights once)
# ==============================
class LexiconCompressorHFModel(PreTrainedModel):
    """HuggingFace wrapper that:
      • owns the vectorized LexiconCompressorModel
      • loads Row/Column Attention weights once (from Qwen layers) on the first call
    """

    config_class = LexiconCompressorConfig

    def __init__(self, config: LexiconCompressorConfig, qwen_model: Qwen3ForCausalLM, full_dict):
        super().__init__(config)
        self.qwen = qwen_model
        self.model = LexiconCompressorModel(
            qwen_model=qwen_model,
            full_dict=full_dict,
            dict_encoder_num_compress_tokens=config.num_compress_tokens,
            dict_encoder_learned_tokens_prepend=config.learned_tokens_prepend,
            compressor_config=config,
        )
        self._rca_loaded: bool = False

    def _build_attention_weights(self):
        """Create per-layer (row_weights, col_weights) from Qwen's own decoder layers.
        This mirrors the Qwen3DecoderLayer state dict into both row/col processors.
        """
        layer_weights = []
        for layer in self.qwen.model.layers:
            sd = layer.state_dict()
            # Use the same weights for row and column paths by default
            layer_weights.append((sd, sd))
        return layer_weights

    def forward(self, **kwargs):
        # Inject RCA weights once
        if not self._rca_loaded:
            attn_w = self._build_attention_weights()
            out = self.model(attention_weights=attn_w, **kwargs)
            self._rca_loaded = True
            return out
        return self.model(**kwargs)


# ==============================
# 2) Dataset
# ==============================
class LexiconDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=512):
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item.get('instruction', '')
        source_text = item.get('source_text', '')
        target_text = item.get('target_text', '')
        full_text = f"{instruction}{source_text}\n{target_text}"

        enc = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_attention_mask=True,
        )
        input_ids = enc['input_ids'].squeeze(0)
        attention_mask = enc['attention_mask'].squeeze(0)

        labels = input_ids.clone()
        # (Option) mask label parts here if needed

        return {
            "qwen_input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "row_indices_per_sample": item['dict_indices'],
        }


# ==============================
# 3) DataCollator
# ==============================
class LexiconDataCollator:
    def __call__(self, features):
        batch = {}
        # Vectorize row indices so DataParallel/DDP can scatter along batch dim
        rows = [torch.as_tensor(f["row_indices_per_sample"], dtype=torch.long) for f in features]
        if len(rows) > 0:
            R_sel = max(int(r.numel()) for r in rows)
            rows = [F.pad(r, (0, R_sel - int(r.numel())), value=-1) if int(r.numel()) < R_sel else r for r in rows]
            batch["row_indices_per_sample"] = torch.stack(rows, dim=0)  # (B, R_sel)
        else:
            batch["row_indices_per_sample"] = torch.empty(0, 0, dtype=torch.long)

        # Stack the rest
        for key in features[0]:
            if key == "row_indices_per_sample":
                continue
            batch[key] = torch.stack([f[key] for f in features], dim=0)
        return batch


# ==============================
# 4) Trainer
# ==============================
class LexiconCompressorTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        row_indices_per_sample = inputs.pop("row_indices_per_sample", None)
        outputs = model(**inputs, row_indices_per_sample=row_indices_per_sample)
        loss = outputs.loss if isinstance(outputs, LexiconCompressorModelOutput) else outputs[0]
        return (loss, outputs) if return_outputs else loss


# ==============================
# 5) Save fallback
# ==============================
def save_model_safe(model, tokenizer, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    try:
        model.save_pretrained(output_dir, safe_serialization=False)
        tokenizer.save_pretrained(output_dir)
        print(f"Model saved successfully to {output_dir}")
    except Exception as e:
        print(f"Standard save failed: {e}\nUsing alternative save method...")
        try:
            torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
            model.config.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"Model saved with alternative method to {output_dir}")
        except Exception as e2:
            print(f"Alternative save also failed: {e2}")


# ==============================
# 6) Dictionary loader (consistency)
# ==============================
def load_dictionary_consistent(csv_path: str):
    entries = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lexical_unit = row.get('lexical_unit', '').strip()
            if lexical_unit:
                entries.append({
                    'lexical_unit': lexical_unit,
                    'variant': row.get('variant', '').strip(),
                    'pos': row.get('pos', '').strip(),
                    'gloss': row.get('gloss', '').strip(),
                })
    return entries


# ==============================
# 7) Main
# ==============================
def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    CSV_PATH = "./data/cleaned_lexicon_tiny.csv"
    MODEL_NAME = "Qwen/Qwen3-0.6B"
    TRAIN_DATA_PATH = "./data/train.json"
    OUTPUT_DIR = "./results"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    qwen_model = Qwen3ForCausalLM.from_pretrained(MODEL_NAME)

    print("Tokenizing dictionary...")
    dict_tokenizer = LexiconCompressorTokenizor(
        csv_path=CSV_PATH,
        tokenizer=tokenizer,
        columns=['lexical_unit', 'pos', 'gloss'],
        sep='; ',
        strip=True,
        lowercase=True,
    )
    full_dict = dict_tokenizer.tokenize()
    print(f"Loaded dictionary rows: {len(full_dict)}")

    _ = load_dictionary_consistent(CSV_PATH)  # optional check

    if not os.path.exists(TRAIN_DATA_PATH):
        raise FileNotFoundError(f"Training data file not found: {TRAIN_DATA_PATH}")

    train_dataset = LexiconDataset(TRAIN_DATA_PATH, tokenizer, max_length=512)

    # Config (num_layers used by RCA config; RCA stack depth in model equals Qwen decoder depth)
    config = LexiconCompressorConfig(
        qwen_config=qwen_model.config,
        num_layers=2,                # kept for config compatibility
        num_compress_tokens=5,
        learned_tokens_prepend=False,
    )

    model = LexiconCompressorHFModel(config=config, qwen_model=qwen_model, full_dict=full_dict)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_train_epochs=30,
        logging_dir="./logs",
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 8e9,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        report_to=[],  # set to ['wandb'] if W&B is available
        logging_first_step=True,
        seed=42,
        max_grad_norm=1.0,
        learning_rate=1e-5,
        warmup_steps=10,
        optim="adamw_torch",
        lr_scheduler_type="linear",
        weight_decay=0.01,
    )

    trainer = LexiconCompressorTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=LexiconDataCollator(),
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    try:
        trainer.save_model(OUTPUT_DIR)
    except Exception as e:
        print(f"Warning: Could not save model normally: {e}\nUsing alternative save method...")
        save_model_safe(model, tokenizer, OUTPUT_DIR)

    print("Done.")


if __name__ == "__main__":
    main()
