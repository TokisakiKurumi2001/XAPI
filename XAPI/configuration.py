from transformers.configuration_utils import PretrainedConfig

class PADConfig(PretrainedConfig):
    model_type = "pad"

    def __init__(
        self,
        mlm_vocab_size=250057,
        hidden_size=1024,
        cls_out=2,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        drop_prob=0.1,
        **kwargs,
    ):
        super().__init__(pad_token_id=1, bos_token_id=0, eos_token_id=2, **kwargs)
        self.mlm_vocab_size = mlm_vocab_size
        self.hidden_size = hidden_size
        self.cls_out = cls_out
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.drop_prob = drop_prob