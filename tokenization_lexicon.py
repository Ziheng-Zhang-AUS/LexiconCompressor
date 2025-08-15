import csv
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional

class LexiconTokenizer:
    def __init__(self,
                 csv_file_path: str,
                 tokenizer: Any,
                 embedding_layer: nn.Module,
                 column_names: List[str],
                 compress_token_id: int,
                 delimiter: Optional[str] = None):
        self.csv_file_path = csv_file_path
        self.tokenizer = tokenizer
        self.embedding_layer = embedding_layer
        self.column_names = column_names
        self.compress_token_id = compress_token_id
        self.delimiter = delimiter if delimiter is not None else ','
        
        self.lexicon_data = self._load_lexicon()

    def _load_lexicon(self):
        data = []
        with open(self.csv_file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
        return data
    
    def _extract_text_from_row(self, row: Dict[str, str]):
        selected_texts = []
        for col in self.column_names:
            if col in row and row[col]:
                selected_texts.append(str(row[col]))
        
        if not selected_texts:
            return ""
        
        result_parts = [selected_texts[0] + ':'] + selected_texts[1:]
        return self.delimiter.join(result_parts)
    
    def process_lexicon(self):
        result_tensors = []
        
        for row in self.lexicon_data:
            text = self._extract_text_from_row(row)
            
            if text.strip(): 
                tokens = self.tokenizer.encode(text, add_special_tokens=True)
                final_tokens = [self.compress_token_id] + tokens

                token_tensor = torch.tensor(final_tokens, dtype=torch.long)
                
                embedded = self.embedding_layer(token_tensor)
                
                result_tensors.append(embedded)
        
        return result_tensors



def main():
    from transformers import AutoTokenizer, AutoModel
    # 加载Qwen3-0.6B的tokenizer和model
    model_name = "Qwen/Qwen3-0.6B"  # 或者本地路径
    print("Loading tokenizer and model...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please make sure you have installed the model or have internet connection")
        return
    
    # 添加新的compress token
    new_tokens = ['[COMP]']
    num_added_toks = tokenizer.add_tokens(new_tokens)
    print(f"Added {num_added_toks} new tokens")
    
    # 扩展embedding层
    model.resize_token_embeddings(len(tokenizer))
    
    # 获取compress token的ID
    compress_token_id = tokenizer.convert_tokens_to_ids('[COMP]')
    print(f"Compress token ID: {compress_token_id}")
    
    # 创建LexiconTokenizer实例
    lexicon_tokenizer = LexiconTokenizer(
        csv_file_path="lexicon_demo.csv",
        tokenizer=tokenizer,
        embedding_layer=model.embed_tokens,
        column_names=["lexical_unit", "pos", "gloss", "variant"],
        compress_token_id=compress_token_id,
        delimiter=","
    )
    
    # 处理lexicon
    print("\nProcessing lexicon...")
    try:
        result_tensors = lexicon_tokenizer.process_lexicon()
        print(f"Processed {len(result_tensors)} lexicon entries")
        
        # 显示所有结果的详细信息
        print("\n" + "="*80)
        print("DETAILED RESULTS:")
        print("="*80)
        
        for i, (row, tensor) in enumerate(zip(lexicon_tokenizer.lexicon_data, result_tensors)):
            print(f"\nEntry {i+1}:")
            print(f"  Original row: {row}")
            
            # 提取并显示原始文本
            original_text = lexicon_tokenizer._extract_text_from_row(row)
            print(f"  Extracted text: '{original_text}'")
            
            # 显示原始tokenization结果
            original_tokens = tokenizer.encode(original_text, add_special_tokens=True)
            print(f"  Original tokens: {original_tokens}")
            print(f"  Original decoded (skip_special_tokens=True): '{tokenizer.decode(original_tokens, skip_special_tokens=True)}'")
            print(f"  Original decoded (skip_special_tokens=False): '{tokenizer.decode(original_tokens, skip_special_tokens=False)}'")
            
            # 显示加上compress token后的结果
            final_tokens = [compress_token_id] + original_tokens
            print(f"  Final tokens (with [COMP]): {final_tokens}")
            print(f"  Final decoded (skip_special_tokens=True): '{tokenizer.decode(final_tokens, skip_special_tokens=True)}'")
            print(f"  Final decoded (skip_special_tokens=False): '{tokenizer.decode(final_tokens, skip_special_tokens=False)}'")
            
            # 显示tensor信息
            print(f"  Embedded tensor shape: {tensor.shape}")
            print(f"  Embedded tensor dtype: {tensor.dtype}")
            
            print("-" * 60)
            
    except Exception as e:
        print(f"Error processing lexicon: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()