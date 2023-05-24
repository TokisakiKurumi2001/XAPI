from datasets import load_dataset
from torch.utils.data import DataLoader
from collections.abc import Mapping
import torch
from torch import Tensor
from typing import List, Tuple, Dict
from transformers import MBart50TokenizerFast
import random
from copy import deepcopy

class PADDataLoader:
    def __init__(self, ckpt: str, max_length: int):
        data_dict = {'train': 'data/pawsx-extend.train.csv', 'test': 'data/pawsx-extend.test.csv', 'validation': 'data/pawsx-extend.validation.csv'}
        # data_dict = {'train': 'data/pawsx-extend.validation.csv'}
        dataset = load_dataset('csv', data_files=data_dict)
        self.tokenizer = MBart50TokenizerFast.from_pretrained(ckpt)
        self.tokenizer.add_tokens(['<LABEL_MASK>', '<LABEL_0>', '<LABEL_1>'])
        self.max_length = max_length
        random.seed(42)
        self.lang_mappings = {
            "en": "en_XX",
            "de": "de_DE",
            "es": "es_XX",
            "fr": "fr_XX",
            "ja": "ja_XX",
            "ko": "ko_KR",
            "zh": "zh_CN",
        }

        # self.dataset = dataset

        self.dataset = dataset.shuffle(seed=42).map(
            self.__tokenize,
            remove_columns=dataset["train"].column_names
        )

    def __tokenize_mlm(self, words: List[str], blk_words: List[str], mask_token_id: int) -> Dict[str, List[int]]:
        # mask whole word in block keywords
        masks = []
        if len(blk_words) > 0:
            for word in words:
                if word in blk_words or word.lower() in blk_words:
                    masks.append(1)
                else:
                    masks.append(0)
        else:
            masks = [0] * len(words)

        inps = self.tokenizer(words, is_split_into_words=True, padding="max_length", max_length=self.max_length, truncation=True)
        labels = []
        for i, idx in enumerate(inps.word_ids(0)):
            if idx is None:
                labels.append(-100)
            else:
                if masks[idx] == 1:
                    # mask
                    labels.append(inps['input_ids'][i])
                    inps['input_ids'][i] = mask_token_id
                else:
                    labels.append(-100)

        inps['labels'] = labels
        return inps

    def __tokenize(self, examples):
        rt_dict = {}
        self.tokenizer.src_lang = self.lang_mappings[examples['lang']]
        encoder_inps = self.tokenizer(examples['sentence1'], padding="max_length", max_length=self.max_length, truncation=True)

        if examples['task'] == "MASK":
            label_hint = f"<LABEL_{examples['label']}>"
            blk_words = examples['blk_kws'].split("|")
        else:
            label_hint = "<LABEL_MASK>"
            blk_words = []
        words = [label_hint] + examples['sentence2'].split(" ")

        decoder_inps = self.__tokenize_mlm(words, blk_words, self.tokenizer.mask_token_id)
        for k, v in encoder_inps.items():
            rt_dict[f"encoder_{k}"] = v
        for k, v in decoder_inps.items():
            rt_dict[f"decoder_{k}"] = v
        rt_dict['cls_label'] = examples['label']
        return rt_dict

    def __collate_fn(self, examples):
        if isinstance(examples, (list, tuple)) and isinstance(examples[0], Mapping):
            encoded_inputs = {key: [example[key] for example in examples] for key in examples[0].keys()}
        else:
            encoded_inputs = examples
        
        # print(examples)
        # batch = self.__tokenize(examples[0])
        # batch = {k: torch.tensor(v, dtype=torch.int32) for k, v in batch.items()}
        # convert list to tensor
        batch = {k: torch.tensor(v, dtype=torch.int32) for k, v in encoded_inputs.items()}

        return batch

    def get_dataloader(self, batch_size:int=16, types: List[str] = ["train", "test"]):
        res = []
        for type in types:
            res.append(
                DataLoader(self.dataset[type], batch_size=batch_size, collate_fn=self.__collate_fn, num_workers=1)
            )
        return res
