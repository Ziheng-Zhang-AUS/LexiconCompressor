from transformers import PretrainedConfig

class LexiconCompressorConfig(PretrainedConfig):
    model_type = "lexicon_compressor"

    def __init__(
        self,
        qwen_config=None,
        num_layers=4,
        num_compress_tokens=5,
        learned_tokens_prepend=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.num_compress_tokens = num_compress_tokens
        self.learned_tokens_prepend = learned_tokens_prepend
        self.qwen_config = qwen_config