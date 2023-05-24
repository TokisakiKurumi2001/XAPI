from transformers import PreTrainedModel, MBartForConditionalGeneration
import torch
from torch import Tensor
import torch.nn as nn
from typing import List, Union, Optional, Tuple, Dict
from PAD import PADConfig

class PADPreTrainedModel(PreTrainedModel):
    config_class = PADConfig
    base_model_prefix = "pad"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

class PADMapperModel(PADPreTrainedModel):
    def __init__(self, config: PADConfig):
        super().__init__(config)
        self.config = config

        # Entailment classification
        self.cls_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.cls_act_fn = nn.Tanh()
        self.cls_dropout = nn.Dropout(config.drop_prob)
        self.cls_out = nn.Linear(config.hidden_size, config.cls_out)

        self.post_init()

    def forward(self, input_embeddings: torch.Tensor) -> Tensor:
        # take only the label hint
        x = input_embeddings[:, 1, :]
        c = self.cls_act_fn(self.cls_dense(x))
        c = self.cls_out(self.cls_dropout(c))
        return c

class PADModel(nn.Module):
    def __init__(self, ckpt: str, mapper_ckpt: str = '', mode: str="train"):
        super(PADModel, self).__init__()
        if mode == "train":
            config = PADConfig()
            self.mapper = PADMapperModel(config)
        else:
            self.mapper = PADMapperModel.from_pretrained(mapper_ckpt)

        self.generator = MBartForConditionalGeneration.from_pretrained(ckpt)
        self.generator.resize_token_embeddings(self.mapper.config.mlm_vocab_size)

    def save_pretrained(self, path):
        self.mapper.save_pretrained(path + "/mapper")
        self.generator.save_pretrained(path + "/generator")

    def forward(self, inputs):
        outputs = self.generator.model(
            input_ids=inputs['encoder_input_ids'],
            attention_mask=inputs['encoder_attention_mask'],
            decoder_input_ids=inputs['decoder_input_ids'],
            return_dict=True,
        )
        mask_pred = self.generator.lm_head(outputs[0]) + self.generator.final_logits_bias
        cls_out = self.mapper(outputs[0])
        return mask_pred, cls_out
