# -*- coding: utf-8 -*-
# @Time    : 11/22/22 11:28 AM
# @Author  : LIANYONGXING
# @FileName: text_classification_task.py
# @Software: PyCharm
# @Repo    : https://github.com/lianyongxing/

import pytorch_lightning as pl
from transformers import get_cosine_schedule_with_warmup, AdamW
import torch
import torch.nn.functional as F
from torch.nn.modules import CrossEntropyLoss
import torchmetrics
from models.chinesebert.modeling_glycebert import GlyceBertForSequenceClassification
from transformers import BertConfig
from datasets.chinesebert_datasets import build_dataloader as build_chinesebert_dataloader


class ChineseBertTextClassificationTask(pl.LightningModule):

    def __init__(self, data_path, bert_path='/Users/user/Desktop/git_projects/ChineseBERT-base'):
        super().__init__()

        self.data_path = data_path
        self.bert_path = bert_path
        self.bert_config = BertConfig.from_pretrained(self.bert_path, output_hidden_states=False, num_labels=2)
        self.model = GlyceBertForSequenceClassification.from_pretrained(self.bert_path, config=self.bert_config)
        self.criterion = CrossEntropyLoss()
        self.acc = torchmetrics.Accuracy(num_classes=2, task='binary')
        self.train_dl, self.valid_dl = self.get_dataloader()

    def get_dataloader(self):
        train_loader, valid_loader = build_chinesebert_dataloader(self.data_path, self.bert_path)
        return train_loader, valid_loader

    def training_step(self, batch, batch_idx):
        loss, acc = self.compute_loss_and_acc(batch)
        tf_board_logs = {
            "train_loss": loss,
            "train_acc": acc,
            "lr": self.trainer.optimizers[0].param_groups[0]['lr']
        }
        return {'loss': loss, 'log': tf_board_logs}

    def compute_loss_and_acc(self, batch):
        ids, pinyins, lab = batch
        batch_size, length = ids.shape
        pinyin_ids = pinyins.view(batch_size, length, 8)

        y = lab.long().view(-1)

        y_hat = self.forward(ids, pinyin_ids)
        # compute loss
        loss = self.criterion(y_hat, y)
        # compute acc
        predict_scores = F.softmax(y_hat, dim=1)
        predict_labels = torch.argmax(predict_scores, dim=-1)
        acc = self.acc(predict_labels, y)
        return loss, acc

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        optimizer = AdamW(self.model.parameters(), lr=2e-5, weight_decay=1e-4)  # AdamW优化器
        # num_gpus = self.num_gpus
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(self.train_dl),
                                                    num_training_steps=1 * len(self.train_dl))

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_ids, pinyin_ids):
        attention_mask = (input_ids != 0).long()
        return self.model(input_ids, pinyin_ids, attention_mask=attention_mask)[0]

    def validation_step(self, batch, idx):
        loss, acc = self.compute_loss_and_acc(batch)
        tf_board_logs = {
            "valid_loss": loss,
            "valid_acc": acc
        }
        return {'loss': loss, 'log': tf_board_logs}

    def predict_step(self, batch, batch_idx, dataloader_idx = None):
        ids, pys, lab = batch
        y_hat = self.forward(ids, pys)
        predict_scores = F.softmax(y_hat, dim=1)
        predict_labels = torch.argmax(predict_scores, dim=-1)
        return predict_labels, predict_scores

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.valid_dl



