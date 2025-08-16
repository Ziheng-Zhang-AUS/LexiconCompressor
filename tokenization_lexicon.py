import csv
from typing import List, Dict, Any, Optional

class LexiconTokenizer:
    def __init__(self,
                 csv_file_path: str,
                 tokenizer: Any,
                 column_names: List[str],
                 compress_token_id: int,
                 delimiter: Optional[str] = None,
                 add_special_tokens: bool = True):
        self.csv_file_path = csv_file_path
        self.tokenizer = tokenizer
        self.column_names = column_names
        self.compress_token_id = compress_token_id
        self.delimiter = delimiter if delimiter is not None else ','
        self.add_special_tokens = add_special_tokens
        
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
        
        result_parts = selected_texts[1:]
        return selected_texts[0] + ':' + self.delimiter.join(result_parts)

    def process_lexicon(self):
        result_tensors = []
        
        for row in self.lexicon_data:
            text = self._extract_text_from_row(row)
            
            if text.strip(): 
                ids = self.tokenizer(text, add_special_tokens=self.add_special_tokens)['input_ids']
                ids = ids[0] if ids and isinstance(ids[0], list) else ids
                final_tokens = [self.compress_token_id] + ids

                result_tensors.append(final_tokens)

        return result_tensors


def main():
    from transformers import AutoTokenizer

    # 固定配置
    MODEL_NAME = "Qwen/Qwen3-0.6B"
    CSV_PATH   = "lexicon_demo.csv"
    COLUMNS    = ["lexical_unit", "pos", "gloss", "variant"]

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # 添加 [COMP]（作为普通 token，便于解码可见）
    tokenizer.add_tokens(["[COMP]"], special_tokens=False)
    comp_id = tokenizer.convert_tokens_to_ids("[COMP]")
    print(f"[COMP] id = {comp_id}")

    # 实例化你的 LexiconTokenizer（词典行一般不需要自动 special tokens）
    lt = LexiconTokenizer(
        csv_file_path=CSV_PATH,
        tokenizer=tokenizer,
        column_names=COLUMNS,
        compress_token_id=comp_id,
        delimiter=",",
        add_special_tokens=False
    )

    # 处理并拿到“带 [COMP]”的 id 序列
    all_ids_with_comp = lt.process_lexicon()
    total = len(all_ids_with_comp)
    print(f"Processed {total} entries. Showing first 5...\n")

    # 打印前 5 行：抽取文本、无/有 [COMP] 的 ids 及解码
    show_n = min(5, total)
    for i in range(show_n):
        row = lt.lexicon_data[i]
        text = lt._extract_text_from_row(row).strip()
        ids_no_comp = tokenizer.encode(text, add_special_tokens=False)
        ids_with_comp = all_ids_with_comp[i]

        print(f"Entry {i+1}")
        print(f"  Extracted text: '{text}'")
        print(f"  IDs (no [COMP]): {ids_no_comp}")
        print(f"  Decode(no [COMP], skip_special=True):  '{tokenizer.decode(ids_no_comp, skip_special_tokens=True)}'")
        print(f"  Decode(no [COMP], skip_special=False): '{tokenizer.decode(ids_no_comp, skip_special_tokens=False)}'")
        print(f"  IDs (with [COMP] first): {ids_with_comp}")
        print(f"  Decode(with [COMP], skip_special=True):  '{tokenizer.decode(ids_with_comp, skip_special_tokens=True)}'")
        print(f"  Decode(with [COMP], skip_special=False): '{tokenizer.decode(ids_with_comp, skip_special_tokens=False)}'")
        print("-" * 80)

if __name__ == "__main__":
    main()