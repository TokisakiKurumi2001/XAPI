import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import List
from XAPI import XAPIModel
import evaluate
import numpy as np
import re
from transformers import get_constant_schedule_with_warmup

class LitXAPI(pl.LightningModule):
    def __init__(self, ckpt: str, lr: float):
        super(LitXAPI, self).__init__()
        self.model = XAPIModel(ckpt)
        self.cls_class_num = self.model.config.num_classes
        self.cls_loss = nn.CrossEntropyLoss()
        self.lr = lr
        self.cls_valid_metric = evaluate.load("metrics/paraid.py")
        self.cls_test_metric = evaluate.load("metrics/paraid.py")
        self.save_hyperparameters()

    def export_model(self, path):
        self.model.save_pretrained(path)

    def training_step(self, batch, batch_idx):
        cls_labels = batch.pop('label')
        logits = self.model(batch)

        cls_loss = self.cls_loss(logits.view(-1, self.cls_class_num), cls_labels.view(-1).long())

        self.log("train/loss", cls_loss, sync_dist=True)
        return cls_loss

    def validation_step(self, batch, batch_idx):
        cls_labels = batch.pop('label')
        logits = self.model(batch)

        cls_loss = self.cls_loss(logits.view(-1, self.cls_class_num), cls_labels.view(-1).long())

        self.log("valid/loss", cls_loss, sync_dist=True)

        cls_preds = logits.argmax(dim=-1)
        self.cls_valid_metric.add_batch(predictions=cls_preds, references=cls_labels)

    def validation_epoch_end(self, outputs):
        results = self.cls_valid_metric.compute()
        for k, v in results.items():
            self.log(f'valid/{k}', v, on_epoch=True, on_step=False, sync_dist=True)

    def test_step(self, batch, batch_idx):
        cls_labels = batch.pop('label')
        logits = self.model(batch)

        cls_loss = self.cls_loss(logits.view(-1, self.cls_class_num), cls_labels.view(-1).long())

        self.log("test/loss", cls_loss, sync_dist=True)

        cls_preds = logits.argmax(dim=-1)
        self.cls_test_metric.add_batch(predictions=cls_preds, references=cls_labels)

    def test_epoch_end(self, outputs):
        results = self.cls_valid_metric.compute()
        for k, v in results.items():
            self.log(f'test/{k}', v, on_epoch=True, on_step=False, sync_dist=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
