from datasets import load_dataset
from torch.utils.data import DataLoader
from collections.abc import Mapping
import torch
from torch import Tensor
from typing import List, Tuple, Dict
from transformers import XLMRobertaTokenizerFast
import random
from copy import deepcopy

class XAPIDataLoader:
    def __init__(self, ckpt: str, max_length: int):
        data_dict = {'train': 'data/pawsx.train.csv', 'test': 'data/pawsx.test.csv', 'validation': 'data/pawsx.validation.csv'}
        dataset = load_dataset('csv', data_files=data_dict)
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(ckpt)
        self.max_length = max_length
        random.seed(42)

        # self.dataset = dataset

        self.dataset = dataset.shuffle(seed=42).map(
            self.__tokenize,
            batched=True,
            remove_columns=dataset["train"].column_names
        )

    def __tokenize(self, examples):
        rt_dict = {}
        toks_1 = self.tokenizer(examples['sentence1'], padding="max_length", max_length=self.max_length)
        toks_2 = self.tokenizer(examples['sentence2'], padding="max_length", max_length=self.max_length)
        for k, v in toks_1.items():
            rt_dict[f'{k}_1'] = v
        for k, v in toks_2.items():
            rt_dict[f'{k}_2'] = v
        rt_dict['label'] = examples['label']
        return rt_dict

    def __collate_fn(self, examples):
        if isinstance(examples, (list, tuple)) and isinstance(examples[0], Mapping):
            encoded_inputs = {key: [example[key] for example in examples] for key in examples[0].keys()}
        else:
            encoded_inputs = examples
        
        # convert list to tensor
        batch = {k: torch.tensor(v, dtype=torch.int32) for k, v in encoded_inputs.items()}

        return batch

    def get_dataloader(self, batch_size:int=16, types: List[str] = ["train", "test"]):
        res = []
        for type in types:
            res.append(
                DataLoader(self.dataset[type], batch_size=batch_size, collate_fn=self.__collate_fn, num_workers=32)
            )
        return res
