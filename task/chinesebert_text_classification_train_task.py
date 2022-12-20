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

class ChineseBertTextClassificationTask(pl.LightningModule):

    def __init__(self, base_model):
        super().__init__()

        self.model = base_model
        self.batch_size = 16
        self.criterion = CrossEntropyLoss()
        self.acc = torchmetrics.Accuracy(num_classes=2)


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
        model = self.model
        optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)  # AdamW优化器
        # num_gpus = self.num_gpus
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(self.trainer._data_connector._train_dataloader_source.dataloader()),
                                                    num_training_steps=1 * len(self.trainer._data_connector._train_dataloader_source.dataloader()))

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_ids, pinyin_ids):
        attention_mask = (input_ids != 0).long()
        return self.model(input_ids, pinyin_ids, attention_mask=attention_mask)[0]


    def predict_step(self, batch, batch_idx, dataloader_idx = None):
        ids, pys, lab = batch

        y_hat = self.forward(ids, pys)

        predict_scores = F.softmax(y_hat, dim=1)
        predict_labels = torch.argmax(predict_scores, dim=-1)
        return predict_labels, predict_scores

    # def train_dataloader(self) -> DataLoader:
    #     return self.get_dataloader("train")

    # def get_dataloader(self, mode="train") -> DataLoader:
    #     """get training dataloader"""
    #     encodings, labs = self.get_encodings(self.train_data_filepath, mode)
    #
    #     dataset = XHSDataset(encodings, labs)
    #
    #     if mode == "train":
    #         # define data_generator will help experiment reproducibility.
    #         data_generator = torch.Generator()
    #         data_generator.manual_seed(2333)
    #         data_sampler = RandomSampler(dataset, generator=data_generator)
    #     else:
    #         data_sampler = SequentialSampler(dataset)
    #
    #     dataloader = DataLoader(dataset,
    #                             batch_size=self.batch_size,
    #                             sampler=data_sampler,
    #                             num_workers=self.workers)
    #
    #     return dataloader

    # def get_encodings(self, path, mode='train'):
    #     datas = pd.read_csv(path)[:100]
    #     texts = datas['content_filter'].tolist()
    #     if mode == 'train':
    #         labs = datas['lab'].tolist()
    #     elif mode == 'test':
    #         labs = [1] * len(texts)
    #     else:
    #         raise Exception('mode 选择错误！！')
    #     encodings = self.model.tokenizer(texts, max_length=self.max_length, padding='max_length', truncation=True)
    #     return encodings, labs


