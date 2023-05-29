from transformers import PreTrainedModel, XLMRobertaModel
import torch
from torch import Tensor
import torch.nn as nn
from typing import List, Union, Optional, Tuple, Dict
from XAPI import XAPIConfig
import torch.nn.functional as F

class XAPIPreTrainedModel(PreTrainedModel):
    config_class = XAPIConfig
    base_model_prefix = "xapi"
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

class XAPIAttend(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, drop_prob: float, act_fn: str, **kwargs):
        super(XAPIAttend, self).__init__(**kwargs)
        self.dropout1 = nn.Dropout(drop_prob)
        self.fc1 = nn.Linear(input_size, hidden_size)
        if act_fn == "relu":
            self.act_fn = nn.ReLU()
        elif act_fn == "gelu":
            self.act_fn = nn.GELU()
        elif act_fn == "tanh":
            self.act_fn = nn.Tanh()
        else:
            assert "Activation function can be: relu, gelu or tanh"
        self.dropout2 = nn.Dropout(drop_prob)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def _compute(self, input):
        out = self.act_fn(self.fc1(self.dropout1(input)))
        out = self.act_fn(self.fc2(self.dropout2(out)))
        return out

    def forward(self, input_1, input_2):
        f_1 = self._compute(input_1)
        f_2 = self._compute(input_2)
        # Shape of `e`: (`batch_size`, no. of tokens in sequence A,
        # no. of tokens in sequence B)
        e = torch.bmm(f_1, f_2.permute(0, 2, 1))
        # Shape of `beta`: (`batch_size`, no. of tokens in sequence A,
        # `embed_size`), where sequence B is softly aligned with each token
        # (axis 1 of `beta`) in sequence A
        beta = torch.bmm(F.softmax(e, dim=-1), input_2)
        # Shape of `alpha`: (`batch_size`, no. of tokens in sequence B,
        # `embed_size`), where sequence A is softly aligned with each token
        # (axis 1 of `alpha`) in sequence B
        alpha = torch.bmm(F.softmax(e.permute(0, 2, 1), dim=-1), input_1)
        return beta, alpha

class XAPICompare(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, drop_prob: float, act_fn: str, **kwargs):
        super(XAPICompare, self).__init__(**kwargs)
        self.dropout1 = nn.Dropout(drop_prob)
        self.fc1 = nn.Linear(input_size, hidden_size)
        if act_fn == "relu":
            self.act_fn = nn.ReLU()
        elif act_fn == "gelu":
            self.act_fn = nn.GELU()
        elif act_fn == "tanh":
            self.act_fn = nn.Tanh()
        else:
            assert "Activation function can be: relu, gelu or tanh"
        self.dropout2 = nn.Dropout(drop_prob)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def _compute(self, input):
        out = self.act_fn(self.fc1(self.dropout1(input)))
        out = self.act_fn(self.fc2(self.dropout2(out)))
        return out

    def forward(self, A, B, beta, alpha):
        V_A = self._compute(torch.cat([A, beta], dim=2))
        V_B = self._compute(torch.cat([B, alpha], dim=2))
        return V_A, V_B

class XAPIAggregate(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, drop_prob: float, act_fn: str, **kwargs):
        super(XAPIAggregate, self).__init__(**kwargs)
        self.dropout1 = nn.Dropout(drop_prob)
        self.fc1 = nn.Linear(input_size, hidden_size)
        if act_fn == "relu":
            self.act_fn = nn.ReLU()
        elif act_fn == "gelu":
            self.act_fn = nn.GELU()
        elif act_fn == "tanh":
            self.act_fn = nn.Tanh()
        else:
            assert "Activation function can be: relu, gelu or tanh"
        self.dropout2 = nn.Dropout(drop_prob)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.flatten = nn.Flatten(start_dim=1)

    def _compute(self, input):
        out = self.flatten(self.act_fn(self.fc1(self.dropout1(input))))
        out = self.flatten(self.act_fn(self.fc2(self.dropout2(out))))
        return out

    def forward(self, V_A, V_B):
        # Sum up both sets of comparison vectors
        V_A = V_A.sum(dim=1)
        V_B = V_B.sum(dim=1)
        # Feed the concatenation of both summarization results into an MLP
        out = self._compute(torch.cat([V_A, V_B], dim=1))
        return out

class XAPIDeAttModel(XAPIPreTrainedModel):
    def __init__(self, config: XAPIConfig):
        super().__init__(config)
        self.config = config

        self.attend = XAPIAttend(
            input_size=self.config.model_hidden_dim,
            hidden_size=self.config.attend_hidden_dim,
            drop_prob=self.config.drop_prob,
            act_fn=self.config.attend_act_fn,
        )
        self.compare = XAPICompare(
            input_size=self.config.compare_input_dim,
            hidden_size=self.config.compare_hidden_dim,
            drop_prob=self.config.drop_prob,
            act_fn=self.config.compare_act_fn,
        )
        self.aggregate = XAPIAggregate(
            input_size=self.config.aggregate_input_dim,
            hidden_size=self.config.aggregate_hidden_dim,
            drop_prob=self.config.drop_prob,
            act_fn=self.config.aggregate_act_fn,
        )

        self.post_init()

    def forward(self, A, B):
        beta, alpha = self.attend(A, B)
        V_A, V_B = self.compare(A, B, beta, alpha)
        out = self.aggregate(V_A, V_B)
        return out

class XAPIModel(nn.Module):
    def __init__(self, ckpt: str, mapper_ckpt: str = '', mode: str="train"):
        super(XAPIModel, self).__init__()
        if mode == "train":
            self.config = XAPIConfig()
            self.deatt = XAPIDeAttModel(self.config)
        else:
            self.deatt = XAPIDeAttModel.from_pretrained(mapper_ckpt)
            self.config = self.deatt.config

        self.encoder = XLMRobertaModel.from_pretrained(ckpt)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.out = nn.Linear(self.config.aggregate_hidden_dim, self.config.num_classes)

    def save_pretrained(self, path):
        self.deatt.save_pretrained(path + "/mapper")
        self.encoder.save_pretrained(path + "/encoder")

    def forward(self, inputs):
        with torch.no_grad():
            input_1 = self.encoder(
                input_ids=inputs['input_ids_1'],
                attention_mask=inputs['attention_mask_1']
            )
            input_2 = self.encoder(
                input_ids=inputs['input_ids_2'],
                attention_mask=inputs['attention_mask_2']
            )
        embedding_1 = input_1.last_hidden_state
        embedding_2 = input_2.last_hidden_state
        out = self.deatt(embedding_1, embedding_2)
        logits = self.out(out)
        return logits