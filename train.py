# train.py
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from modeling_lexicon_compressor import LexiconCompressorModel
from tokenization_lexicon_compressor import LexiconCompressorTokenizor


# ===== Custom collator for handling row_indices_per_sample =====
class CustomDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features, return_tensors="pt"):
        row_indices_list = []
        cleaned_features = []

        for feature in features:
            row_indices = feature.pop("row_indices_per_sample", [])
            row_indices_list.append(row_indices)

            if "input_ids" in feature and len(feature["input_ids"]) > 0:
                cleaned_features.append(feature)
            else:
                print(f"⚠️ Warning: Skipping feature with empty input_ids")
                row_indices_list[-1] = []

        if not cleaned_features:
            print("❌ Error: All features in batch are empty")
            dummy_ids = [self.tokenizer.eos_token_id]
            cleaned_features = [{
                "input_ids": dummy_ids,
                "attention_mask": [1],
            
                "labels": dummy_ids
            }]
            row_indices_list = [[]]

        # ==== padding input_ids / attention_mask / labels ====
        max_length = max(len(f["input_ids"]) for f in cleaned_features)
        batch_input_ids, batch_attention_mask, batch_labels = [], [], []

        for f in cleaned_features:
            input_ids = f["input_ids"]
            labels = f.get("labels", input_ids.copy())
            pad_len = max_length - len(input_ids)

            padded_input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            padded_labels = labels + [-100] * pad_len
            attention_mask = [1] * len(input_ids) + [0] * pad_len

            batch_input_ids.append(padded_input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(padded_labels)

        batch = {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }

        # ==== padding row_indices ====
        if row_indices_list and any(len(r) > 0 for r in row_indices_list):
            max_len = max(len(r) for r in row_indices_list if len(r) > 0)
            padded_indices = torch.full((len(row_indices_list), max_len), -1, dtype=torch.long)
            for i, indices in enumerate(row_indices_list):
                if len(indices) > 0:
                    valid_len = min(len(indices), max_len)
                    padded_indices[i, :valid_len] = torch.tensor(indices[:valid_len], dtype=torch.long)
            batch["row_indices_per_sample"] = padded_indices
        else:
            batch["row_indices_per_sample"] = torch.full((len(cleaned_features), 1), -1, dtype=torch.long)

        return batch


# ===== Custom Trainer (remap input_ids → qwen_input_ids) =====
class LexiconTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels", None)
        row_indices = inputs.pop("row_indices_per_sample", None)

        # === 映射字段名 ===
        if "input_ids" in inputs:
            inputs["qwen_input_ids"] = inputs.pop("input_ids")
        if "attention_mask" in inputs:
            inputs["qwen_attention_mask"] = inputs.pop("attention_mask")

        # print("++++")
        # print(sum(p.numel() for p in model.qwen.parameters()))

        outputs = model(
            **inputs,
            labels=labels,
            row_indices_per_sample=row_indices,
            use_cache=False,
        )
        loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, requires_grad=True)
        return (loss, outputs) if return_outputs else loss


def main():
    # ==== Tokenizer ====
    model_name = "Qwen/Qwen3-1.7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ==== Base Qwen ====
    base_qwen = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

# # 应该是几亿到几十亿，不会是 0
#     print(list(base_qwen.model.layers[0].named_parameters())[:5])


    # ==== Dictionary ====
    lex_tokenizer = LexiconCompressorTokenizor(
        csv_path="data/cleaned_lexicon_tiny.csv",
        tokenizer=tokenizer,
        columns=["lexical_unit", "pos", "gloss"],
    )
    full_dict = lex_tokenizer.tokenize()
    print(f"📖 Loaded dictionary with {len(full_dict)} entries")

    # ==== Custom Model ====
    model = LexiconCompressorModel(
        qwen_model=base_qwen,
        full_dict=full_dict,
        dict_encoder_num_compress_tokens=4,
        dict_encoder_learned_tokens_prepend=True,
    )
    model.qwen.gradient_checkpointing_enable()
    # print("====")
    # print(sum(p.numel() for p in model.qwen.parameters()))

    # ==== Dataset ====
    dataset = load_dataset(
        "json",
        data_files={"train": "data/train.json", "validation": "data/test.json"},
    )
    print(f"📊 Dataset loaded: train={len(dataset['train'])}, val={len(dataset['validation'])}")

    # ==== Preprocess ====
    def preprocess(example):
        instruction = example.get("instruction", "")
        source = example.get("source_text", "")
        target = example.get("target_text", "")

        full_text = f"{instruction}{source} -> {target}"
        if not full_text.strip():
            full_text = "Empty example."

        enc = tokenizer(full_text, truncation=True, max_length=512, padding=False)
        enc["labels"] = enc["input_ids"].copy()

        dict_indices = example.get("dict_indices", [])
        max_dict_size = len(full_dict)
        enc["row_indices_per_sample"] = [i for i in dict_indices if 0 <= i < max_dict_size][:32]

        return enc

    tokenized = dataset.map(
        preprocess,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )
    tokenized = tokenized.filter(lambda e: len(e["input_ids"]) > 0)

    # ==== Collator & Args ====
    data_collator = CustomDataCollator(tokenizer)
    training_args = TrainingArguments(
        output_dir="./outputs/1.7B/2",
        report_to="none",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=5e-5,
        num_train_epochs=100,
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        save_total_limit=10,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        # deepspeed="deepspeed/ds_config.json"
    )

    # ==== Trainer ====
    trainer = LexiconTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # ==== Run training ====
    print("🚀 Starting training...")
    trainer.train()
    trainer.save_model("./outputs/final_model")
    print("✅ Training completed!")


if __name__ == "__main__":
    main()