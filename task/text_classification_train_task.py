# -*- coding: utf-8 -*-
# @Time    : 11/22/22 11:28 AM
# @Author  : LIANYONGXING
# @FileName: text_classification_task.py
# @Software: PyCharm

import pytorch_lightning as pl
from transformers import get_cosine_schedule_with_warmup, AdamW
import torch
import torch.nn.functional as F
from torch.nn.modules import CrossEntropyLoss
import torchmetrics
from models.bert import Bert


class BertTextClassificationTask(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.model = Bert('/Users/user/Desktop/git_projects/text-classification-nlp-pytorch/resources/chinese_bert')
        self.criterion = CrossEntropyLoss()
        self.acc = torchmetrics.Accuracy(num_classes=2, task='binary')

    def training_step(self, batch, idx):
        loss, acc = self.compute_loss_and_acc(batch)
        tf_board_logs = {
            "train_loss": loss,
            "train_acc": acc,
            "lr": self.trainer.optimizers[0].param_groups[0]['lr']
        }
        return {'loss': loss, 'log': tf_board_logs}

    def compute_loss_and_acc(self, batch):
        ids, att, tpe, lab = batch['input_ids'], batch['attention_mask'], batch['token_type_ids'], batch['label']
        y = lab.long()
        logits = self.model(ids, tpe, att)
        # compute loss
        loss = self.criterion(logits, y)
        # compute acc
        predict_scores = F.softmax(logits, dim=1)
        predict_labels = torch.argmax(predict_scores, dim=-1)
        acc = self.acc(predict_labels, y)
        return loss, acc

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        optimizer = AdamW(self.model.parameters(), lr=2e-5, weight_decay=1e-4)  # AdamW优化器
        # num_gpus = self.num_gpus
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(self.train_dataloader()),
                                                    num_training_steps=1 * len(self.train_dataloader()))
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def validation_step(self, batch, idx):
        loss, acc = self.compute_loss_and_acc(batch)
        tf_board_logs = {
            "valid_loss": loss,
            "valid_acc": acc
        }
        return {'loss': loss, 'log': tf_board_logs}

    def predict_step(self, batch, batch_idx, dataloader_idx = None):
        ids, att, tpe, lab = batch['input_ids'], batch['attention_mask'], batch['token_type_ids'], batch['label']
        y_hat = self.model(ids, tpe, att)
        predict_scores = F.softmax(y_hat, dim=1)
        predict_labels = torch.argmax(predict_scores, dim=-1)
        return predict_labels, predict_scores

    def forward(self, text):
        input_ids, input_masks, input_types = [], [], []  # input char ids, segment type ids, attention mask  # 标签
        encode_dict = self.model.tokenizer.encode_plus(text, max_length=128, padding='max_length', truncation=True)
        input_ids.append(encode_dict['input_ids'])
        input_types.append(encode_dict['token_type_ids'])
        input_masks.append(encode_dict['attention_mask'])
        logits = self.model(torch.LongTensor(input_ids), torch.LongTensor(input_masks),
                             torch.LongTensor(input_types))
        y_pred_res = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()[0]
        y_pred_prob = F.softmax(logits, dim=1).detach().cpu().numpy()[0][1]
        return y_pred_res, y_pred_prob