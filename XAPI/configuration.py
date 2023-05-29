from transformers.configuration_utils import PretrainedConfig

class XAPIConfig(PretrainedConfig):
    model_type = "xapi"

    def __init__(
        self,
        model_hidden_dim=768,
        attend_hidden_dim=256,
        drop_prob=0.2,
        attend_act_fn="relu",
        compare_hidden_dim=256,
        compare_act_fn="relu",
        aggregate_hidden_dim=256,
        aggregate_act_fn="relu",
        num_classes=2,
        initializer_range=0.2,
        **kwargs,
    ):
        super().__init__(pad_token_id=1, bos_token_id=0, eos_token_id=2, **kwargs)
        self.model_hidden_dim = model_hidden_dim
        self.attend_hidden_dim = attend_hidden_dim
        self.drop_prob = drop_prob
        self.attend_act_fn = attend_act_fn
        self.compare_input_dim = self.model_hidden_dim * 2
        self.compare_hidden_dim = compare_hidden_dim
        self.compare_act_fn = compare_act_fn
        self.aggregate_input_dim = self.compare_hidden_dim * 2
        self.aggregate_hidden_dim = aggregate_hidden_dim
        self.aggregate_act_fn = aggregate_act_fn
        self.num_classes = num_classes
        self.initializer_range = initializer_range