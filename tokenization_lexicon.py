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
        """
        Initialize lexicon tokenizer

        Args:
            csv_file_path: a csv file for dictoinary. The first line should be column names.

            tokenizer: Qwen3 tokenizer, which normally should be added a new token as compress token.

            colulmns_names: the columns that are used to input.

            compress_token_id: the token id which is used as compress token.

            delimiter: delimiter used to form the input. 
                Example: delimiter=',', row: yajinga, Verb, to go
                    Output: yajinga:Verb,to go
                    The lexicon unit will always be followed by ':', and the other component will be joined by delimiter.

            add_special_tokens: args for Qwen3 tokenizer.
        """
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

    # Fixed configuration
    MODEL_NAME = "Qwen/Qwen3-0.6B"
    CSV_PATH   = "lexicon_demo.csv"
    COLUMNS    = ["lexical_unit", "pos", "gloss", "variant"]

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Add [COMP] as a regular token for visibility during decoding
    tokenizer.add_tokens(["[COMP]"], special_tokens=False)
    comp_id = tokenizer.convert_tokens_to_ids("[COMP]")
    print(f"[COMP] id = {comp_id}")

    # Instantiate your LexiconTokenizer 
    lt = LexiconTokenizer(
        csv_file_path=CSV_PATH,
        tokenizer=tokenizer,
        column_names=COLUMNS,
        compress_token_id=comp_id,
        delimiter=",",
        add_special_tokens=False
    )

    # Process and get ID sequences with [COMP] tokens
    all_ids_with_comp = lt.process_lexicon()
    total = len(all_ids_with_comp)
    print(f"Processed {total} entries. Showing first 5...\n")

    # Display first 5 entries: extract text, encode, then decode
    show_n = min(5, total)
    for i in range(show_n):
        row = lt.lexicon_data[i]
        text = lt._extract_text_from_row(row).strip()
        ids_with_comp = all_ids_with_comp[i]

        print(f"Entry {i+1}")
        print(f"  Original text: '{text}'")
        print(f"  Encoded IDs: {ids_with_comp}")
        print(f"  Decoded:  '{tokenizer.decode(ids_with_comp, skip_special_tokens=True)}'")
        print("-" * 80)


if __name__ == "__main__":
    main()