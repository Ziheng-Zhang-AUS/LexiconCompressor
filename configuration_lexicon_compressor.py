# configuration_lexicon_compressor
from typing import Optional
from transformers import PretrainedConfig


class LexiconCompressorConfig(PretrainedConfig):
    model_type = "lexicon_compressor"
    
    def __init__(
        self,
        qwen_config=None,
        num_layers: int = 2,
        num_compress_tokens: int = 5,
        learned_tokens_prepend: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.qwen_config = qwen_config
        self.num_layers = num_layers
        self.num_compress_tokens = num_compress_tokens
        self.learned_tokens_prepend = learned_tokens_prepend