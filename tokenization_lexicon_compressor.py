# tokenization_lexicon_compressor.py
import csv
from typing import List, Optional


class LexiconCompressorTokenizor:
    def __init__(
        self,
        csv_path: str,
        tokenizer,
        columns: List[str],
        sep: str = "; ",
        strip: bool = True,
        lowercase: bool = True
    ):
        self.csv_path = csv_path
        self.tokenizer = tokenizer
        self.columns = columns
        self.sep = sep
        self.strip = strip
        self.lowercase = lowercase

    def tokenize(self) -> List[List[int]]:
        """Tokenize dictionary entries into list of token IDs."""
        entries = []
        
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Combine specified columns
                parts = []
                for col in self.columns:
                    if col in row and row[col]:
                        value = row[col]
                        if self.lowercase:
                            value = value.lower()
                        if self.strip:
                            value = value.strip()
                        parts.append(value)
                
                # Join parts and tokenize
                text = self.sep.join(parts)
                if text:
                    tokens = self.tokenizer.encode(text, add_special_tokens=False)
                    entries.append(tokens)
                else:
                    # Empty entry
                    entries.append([])
                    
        return entries