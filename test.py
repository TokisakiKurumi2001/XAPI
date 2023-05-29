import torch.nn as nn
import torch
from XAPI import XAPIModel, XAPIDataLoader
import evaluate
from transformers import XLMRobertaModel

ckpt = 'xlm-roberta-base'

dataloader = XAPIDataLoader(ckpt, 128)
[train_dataloader] = dataloader.get_dataloader(batch_size=2, types=['train'])

for batch in train_dataloader:
    # print(batch)
    break

model = XAPIModel(ckpt)
cls_class_num = 2
cls_loss = nn.CrossEntropyLoss()
metric = evaluate.load("metrics/paraid.py")
cls_labels = batch.pop('label')

with torch.no_grad():
    logits = model(batch)

loss = cls_loss(logits.view(-1, cls_class_num), cls_labels.view(-1).long())
print(loss)
cls_preds = logits.argmax(dim=-1)
metric.add_batch(predictions=cls_preds, references=cls_labels)
results = metric.compute()
for k, v in results.items():
    print(f'valid/{k}', v)