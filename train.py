# train.py
import os
import json
import csv
import torch
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
# 1. HF 模型包装类 - 简化版本
# ==============================
class LexiconCompressorHFModel(PreTrainedModel):
    config_class = LexiconCompressorConfig

    def __init__(self, config: LexiconCompressorConfig, qwen_model, full_dict):
        super().__init__(config)
        self.model = LexiconCompressorModel(
            qwen_model=qwen_model,
            full_dict=full_dict,
            dict_encoder_num_layers=config.num_layers,
            dict_encoder_num_compress_tokens=config.num_compress_tokens,
            dict_encoder_learned_tokens_prepend=config.learned_tokens_prepend,
            compressor_config=config
        )

    def forward(self, **kwargs):
        return self.model(**kwargs)


# ==============================
# 2. 自定义 Dataset
# ==============================
class LexiconDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=512):
        """
        data_file: JSON文件路径
        tokenizer: Qwen tokenizer
        """
        print(f"Loading training data from {data_file}...")
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} training samples")
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 获取文本内容
        instruction = item.get('instruction', '')
        source_text = item.get('source_text', '')
        target_text = item.get('target_text', '')
        
        # 构造完整的输入文本
        full_text = f"{instruction}{source_text}\n{target_text}"
        
        # 编码文本
        enc = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_attention_mask=True
        )
        
        input_ids = enc['input_ids'].squeeze()
        attention_mask = enc['attention_mask'].squeeze()
        
        # 创建labels（可以mask掉部分区域）
        labels = input_ids.clone()
        
        # 如果需要，可以mask掉instruction部分
        # 这里简单处理，你可能需要根据具体需求调整
        
        return {
            "qwen_input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "row_indices_per_sample": item['dict_indices']  # 使用预计算的词典索引
        }


# ==============================
# 3. 自定义 DataCollator
# ==============================
class LexiconDataCollator:
    def __call__(self, features):
        batch = {}
        for key in features[0]:
            if key == "row_indices_per_sample":
                batch[key] = [f[key] for f in features]
            else:
                batch[key] = torch.stack([f[key] for f in features])
        return batch


# ==============================
# 4. 自定义 Trainer
# ==============================
class LexiconCompressorTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        row_indices_per_sample = inputs.pop("row_indices_per_sample", None)
        outputs = model(**inputs, row_indices_per_sample=row_indices_per_sample)
        loss = outputs.loss if isinstance(outputs, LexiconCompressorModelOutput) else outputs[0]
        return (loss, outputs) if return_outputs else loss


# ==============================
# 5. 备选模型保存函数
# ==============================
def save_model_safe(model, tokenizer, output_dir):
    """安全的模型保存方法"""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 尝试标准保存
        model.save_pretrained(output_dir, safe_serialization=False)
        tokenizer.save_pretrained(output_dir)
        print(f"Model saved successfully to {output_dir}")
    except Exception as e:
        print(f"Standard save failed: {e}")
        print("Using alternative save method...")
        try:
            # 备选方法1: 直接保存状态字典
            torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
            model.config.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"Model saved with alternative method to {output_dir}")
        except Exception as e2:
            print(f"Alternative save also failed: {e2}")


# ==============================
# 6. 一致性检查函数
# ==============================
def load_dictionary_consistent(csv_path: str):
    """与数据生成时使用相同的词典加载方式"""
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
                    'gloss': row.get('gloss', '').strip()
                })
    return entries


# ==============================
# 7. 主训练逻辑
# ==============================
def main():
    # 设置环境变量
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # 路径和参数
    CSV_PATH = "./data/cleaned_lexicon_tiny.csv"
    MODEL_NAME = "Qwen/Qwen3-0.6B"
    TRAIN_DATA_PATH = "./data/train.json"  # 使用你的JSON文件
    OUTPUT_DIR = "./results"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载 tokenizer 和模型
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    qwen_model = Qwen3ForCausalLM.from_pretrained(MODEL_NAME)

    # 构造词典数据
    print("Loading dictionary data...")
    dict_tokenizer = LexiconCompressorTokenizor(
        csv_path=CSV_PATH,
        tokenizer=tokenizer,
        columns=['lexical_unit', 'pos', 'gloss'],
        sep='; ',
        strip=True,
        lowercase=True
    )
    full_dict = dict_tokenizer.tokenize()
    print(f"Loaded dictionary with {len(full_dict)} entries")

    # 验证词典一致性
    dict_entries = load_dictionary_consistent(CSV_PATH)
    print(f"Dictionary consistency check: {len(dict_entries)} entries loaded")
    print(f"Max dictionary index: {len(full_dict) - 1}")

    # 构造训练数据
    print("Loading training data...")
    if not os.path.exists(TRAIN_DATA_PATH):
        raise FileNotFoundError(f"Training data file not found: {TRAIN_DATA_PATH}")
    
    train_dataset = LexiconDataset(
        data_file=TRAIN_DATA_PATH,
        tokenizer=tokenizer,
        max_length=512
    )

    # 构造模型配置
    config = LexiconCompressorConfig(
        qwen_config=qwen_model.config,
        num_layers=2,  # 减少层数用于测试
        num_compress_tokens=5,
        learned_tokens_prepend=False
    )

    # 构造 HF 模型包装类
    model = LexiconCompressorHFModel(
        config=config,
        qwen_model=qwen_model,
        full_dict=full_dict
    )

    # 设置训练参数
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
        report_to=['wandb'],
        logging_first_step=True,
        seed=42,
        # 训练稳定性参数
        max_grad_norm=1.0,           # 梯度裁剪
        learning_rate=1e-5,          # 学习率
        warmup_steps=10,             # 预热步数
        optim="adamw_torch",
        lr_scheduler_type="linear",
        weight_decay=0.01,
    )

    # 初始化自定义 Trainer
    trainer = LexiconCompressorTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=LexiconDataCollator(),
    )

    # 开始训练
    print("Starting training...")
    trainer.train()

    # 保存模型
    print("Saving model...")
    try:
        trainer.save_model(OUTPUT_DIR)
    except Exception as e:
        print(f"Warning: Could not save model normally: {e}")
        print("Using alternative save method...")
        save_model_safe(model, tokenizer, OUTPUT_DIR)
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()