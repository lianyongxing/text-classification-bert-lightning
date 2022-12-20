# -*- coding: utf-8 -*-
# @Time    : 11/22/22 4:24 PM
# @Author  : LIANYONGXING
# @FileName: text_multi_task_learning_task.py
# @Software: PyCharm
# @Repo    : https://github.com/lianyongxing/


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
from torch.nn.modules import CrossEntropyLoss, BCEWithLogitsLoss
import torchmetrics

class BertMultiClassificationTask(pl.LightningModule):

    def __init__(self, base_model):
        super().__init__()

        self.model = base_model
        self.criterion1 = BCEWithLogitsLoss()
        self.criterion2 = CrossEntropyLoss()
        self.acc = torchmetrics.Accuracy(num_classes=2)

    def sub_task_criterion(self, logits, y):
        logits = logits.t()
        loss0 = self.criterion1(logits[0], y[0].float())
        loss1 = self.criterion1(logits[1], y[1].float())
        loss2 = self.criterion1(logits[2], y[2].float())
        loss3 = self.criterion1(logits[3], y[3].float())
        loss4 = self.criterion1(logits[4], y[4].float())
        loss5 = self.criterion1(logits[5], y[5].float())
        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5
        return loss / 6

    def new_criterion(self, y_pred_sz, y_pred_labs, y_sz, y):
        loss0 = self.criterion2(y_pred_sz, y_sz)
        loss1 = self.sub_task_criterion(y_pred_labs, y)
        return (loss1 + loss0) / 2

    def training_step(self, batch, batch_idx):
        loss, acc = self.compute_loss_and_acc(batch)
        tf_board_logs = {
            "train_loss": loss,
            "train_acc": acc,
            "lr": self.trainer.optimizers[0].param_groups[0]['lr']
        }
        return {'loss': loss, 'log': tf_board_logs}

    def compute_loss_and_acc(self, batch):
        ids, att, tpe, lab, sub_lab = batch['input_ids'], batch['attention_mask'], batch['token_type_ids'], batch['label'], batch['sub_label']
        y = lab.long()
        y_sub = sub_lab.long().t()
        y_mtask, y_stask = self.forward(ids, tpe, att)
        # compute loss
        loss = self.new_criterion(y_mtask, y_stask, y, y_sub)
        return loss, 0

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)  # AdamW优化器
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(self.train_dataloader()),
                                                    num_training_steps=1 * len(self.train_dataloader()))

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_ids, input_types, input_masks):
        return self.model(input_ids=input_ids, token_type_ids=input_types, attention_mask=input_masks)


    # def predict_step(self, batch, batch_idx, dataloader_idx = None):
    #     ids, att, tpe, lab = batch['input_ids'], batch['attention_mask'], batch['token_type_ids'], batch['label']
    #     y_hat = self.forward(ids, tpe, att)
    #
    #     predict_scores = F.softmax(y_hat, dim=1)
    #     predict_labels = torch.argmax(predict_scores, dim=-1)
    #     return predict_labels, predict_scores
