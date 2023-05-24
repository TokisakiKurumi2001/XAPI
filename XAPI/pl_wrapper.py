import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import List
from PAD import PADModel
import evaluate
import numpy as np
import re
from transformers import get_constant_schedule_with_warmup

class LitPAD(pl.LightningModule):
    def __init__(self, ckpt: str, lr: float):
        super(LitPAD, self).__init__()
        self.model = PADModel(ckpt)
        self.mlm_vocab_size = self.model.mapper.config.mlm_vocab_size
        self.cls_class_num = self.model.mapper.config.cls_out
        self.mlm_loss = nn.CrossEntropyLoss()
        self.cls_loss = nn.CrossEntropyLoss()
        self.lr = lr
        self.cls_valid_metric = evaluate.load("metrics/accuracy.py")
        self.cls_test_metric = evaluate.load("metrics/accuracy.py")
        self.save_hyperparameters()

    def export_model(self, path):
        self.model.save_pretrained(path)

    def training_step(self, batch, batch_idx):
        mlm_labels = batch.pop('decoder_labels')
        cls_labels = batch.pop('cls_label')
        m, c = self.model(batch)

        mlm_loss = self.mlm_loss(m.view(-1, self.mlm_vocab_size), mlm_labels.view(-1).long())
        cls_loss = self.cls_loss(c.view(-1, self.cls_class_num), cls_labels.view(-1).long())

        loss = torch.nan_to_num(mlm_loss) + cls_loss
        self.log("train/mlm_loss", mlm_loss, sync_dist=True)
        self.log("train/cls_loss", cls_loss, sync_dist=True)
        self.log("train/loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        mlm_labels = batch.pop('decoder_labels')
        cls_labels = batch.pop('cls_label')
        m, c = self.model(batch)

        mlm_loss = self.mlm_loss(m.view(-1, self.mlm_vocab_size), mlm_labels.view(-1).long())
        cls_loss = self.cls_loss(c.view(-1, self.cls_class_num), cls_labels.view(-1).long())

        loss = torch.nan_to_num(mlm_loss) + cls_loss
        self.log("valid/mlm_loss", mlm_loss, sync_dist=True)
        self.log("valid/cls_loss", cls_loss, sync_dist=True)
        self.log("valid/loss", loss, sync_dist=True)

        cls_preds = c.argmax(dim=-1)
        self.cls_valid_metric.add_batch(predictions=cls_preds, references=cls_labels)

    def validation_epoch_end(self, outputs):
        results = self.cls_valid_metric.compute()
        self.log('valid/accuracy', results['accuracy'], on_epoch=True, on_step=False, sync_dist=True)

    def test_step(self, batch, batch_idx):
        mlm_labels = batch.pop('decoder_labels')
        cls_labels = batch.pop('cls_label')
        m, c = self.model(batch)

        mlm_loss = self.mlm_loss(m.view(-1, self.mlm_vocab_size), mlm_labels.view(-1).long())
        cls_loss = self.cls_loss(c.view(-1, self.cls_class_num), cls_labels.view(-1).long())

        loss = torch.nan_to_num(mlm_loss) + cls_loss
        self.log("test/mlm_loss", mlm_loss, sync_dist=True)
        self.log("test/cls_loss", cls_loss, sync_dist=True)
        self.log("test/loss", loss, sync_dist=True)

        cls_preds = c.argmax(dim=-1)
        self.cls_test_metric.add_batch(predictions=cls_preds, references=cls_labels)

    def test_epoch_end(self, outputs):
        results = self.cls_valid_metric.compute()
        self.log('test/accuracy', results['accuracy'], on_epoch=True, on_step=False, sync_dist=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        lr_scheduler = get_constant_schedule_with_warmup(optimizer, 1500)
        return [optimizer], [lr_scheduler]
