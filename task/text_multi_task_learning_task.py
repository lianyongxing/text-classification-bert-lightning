# -*- coding: utf-8 -*-
# @Time    : 11/22/22 4:24 PM
# @Author  : LIANYONGXING
# @FileName: text_multi_task_learning_task.py
# @Software: PyCharm
# @Repo    : https://github.com/lianyongxing/

import pytorch_lightning as pl
from transformers import get_cosine_schedule_with_warmup, AdamW
import torch
from torch.nn.modules import CrossEntropyLoss, BCEWithLogitsLoss
import torchmetrics
from models.multi_task_bert import MultiTaskBert
from utils.utils import text_filtering
from datasets.mlt_datasets import build_dataloader


class BertMultiClassificationTask(pl.LightningModule):

    def __init__(self, train_filepath, bert_path='/Users/user/Desktop/git_projects/ChineseBERT-base'):
        super().__init__()

        self.train_filepath = train_filepath
        self.train_dl, self.valid_dl = self.get_dataloader()

        self.model = MultiTaskBert(bert_path)
        self.criterion1 = BCEWithLogitsLoss()
        self.criterion2 = CrossEntropyLoss()
        self.acc = torchmetrics.Accuracy(num_classes=2, task='binary')

    def sub_task_criterion(self, logits, y):

        loss0 = self.criterion1(logits[:,0], y[:,0])
        loss1 = self.criterion1(logits[:,1], y[:,1])
        loss2 = self.criterion1(logits[:,2], y[:,2])
        loss3 = self.criterion1(logits[:,3], y[:,3])
        loss4 = self.criterion1(logits[:,4], y[:,4])
        loss5 = self.criterion1(logits[:,5], y[:,5])
        loss = loss0.item() + loss1.item() + loss2.item() + loss3.item() + loss4.item() + loss5.item()
        return loss/6

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
        y = lab
        y_sub = sub_lab
        y_mtask, y_stask = self.model(ids, tpe, att)

        predict_scores = torch.nn.functional.softmax(y_mtask, dim=-1)
        # predict_labels = torch.argmax(predict_scores, dim=-1)
        # compute loss
        loss = self.new_criterion(y_mtask, y_stask, y, y_sub)

        acc = self.acc(predict_scores, y)

        return loss, acc

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        optimizer = AdamW(self.model.parameters(), lr=2e-5, weight_decay=1e-4)  # AdamW优化器
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(self.train_dl),
                                                    num_training_steps=1 * len(self.train_dl))

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def encoding(self, raw_text):
        text = text_filtering(raw_text)
        if len(text) <= 1:
            return False
        input_ids, input_masks, input_types = [], [], []  # input char ids, segment type ids, attention mask  # 标签
        encode_dict = self.model.tokenizer.encode_plus(text, max_length=self.model.max_len, padding='max_length', truncation=True)
        input_ids.append(encode_dict['input_ids'])
        input_types.append(encode_dict['token_type_ids'])
        input_masks.append(encode_dict['attention_mask'])
        logits = self.model(torch.LongTensor(input_ids),
                              torch.LongTensor(input_masks),
                              torch.LongTensor(input_types))
        return logits

    def compute_score(self, logits):
        if logits == False:
            return 0, 0, [0] * (self.model.classes_num - 1), [0] * (self.model.classes_num - 1)
        logs_sz, logs_lab = logits
        sz_score = torch.softmax(logs_sz, dim=-1)[0].cpu().tolist()[1]
        lab_scores = torch.sigmoid(logs_lab)[0]
        # 阈值
        if sz_score >= 0.5:
            sz_lab = 1
        else:
            sz_lab = 0

        # 标签阈值
        labs = lab_scores > 0.5
        labs = labs.to(torch.float).detach().cpu().tolist()

        lab_scores = lab_scores.detach().cpu().tolist()

        return sz_score, sz_lab, lab_scores, labs

    def forward(self, text):
        logs = self.encoding(text)
        sz_score, sz_lab, lab_scores, labs = self.compute_score(logs)
        print('涉政分:\t%.8f\n是否涉政: %s\n各标签分数: %s\n各标签命中: %s\n' % (sz_score, sz_lab, lab_scores, labs))
        return sz_score, sz_lab, lab_scores, labs

    def get_dataloader(self):
        train_dataloader, valid_dataloader = build_dataloader(self.train_filepath, batch_size=4, max_len=256)
        return train_dataloader, valid_dataloader

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.valid_dl