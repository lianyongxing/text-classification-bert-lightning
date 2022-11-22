# -*- coding: utf-8 -*-
# @Time    : 11/22/22 3:21 PM
# @Author  : LIANYONGXING
# @FileName: model.py
# @Software: PyCharm
# @Repo    : https://github.com/lianyongxing/text-classification-bert-lightning

from models.bert import Bert
from task.text_classification_train_task import BertTextClassificationTask
from datasets.basic_datasets import build_dataloader
import pytorch_lightning as pl


def experiment1():
    bert_path = '/Users/user/Desktop/git_projects/text-classification-nlp-pytorch/resources/chinese_bert'
    data_path = '/Users/user/Downloads/final_train_v1.csv'

    base_model = Bert(bert_path)
    model = BertTextClassificationTask(base_model)

    train_loader = build_dataloader(data_path, 128)

    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model=model, train_dataloader=train_loader)


if __name__ == '__main__':
    experiment1()