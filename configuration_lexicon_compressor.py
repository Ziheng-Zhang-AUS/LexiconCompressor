from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

class LexiconCompressorConfig:
    model_type = "lexicon_compressor"
    
    def __init__(self, 
                 qwen_config: Qwen3Config = None,
                 num_layers: int = 4,
                 num_compress_tokens: int = 5,
                 learned_tokens_prepend: bool = True
                 ):
        self.qwen_config = qwen_config
        self.num_layers = num_layers
        self.num_compress_tokens = num_compress_tokens
        self.learned_tokens_prepend = learned_tokens_prepend
    

