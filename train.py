# -*- coding: utf-8 -*-
# @Time    : 11/22/22 3:21 PM
# @Author  : LIANYONGXING
# @FileName: model.py
# @Software: PyCharm
# @Repo    : https://github.com/lianyongxing/text-classification-bert-lightning

from task.text_classification_train_task import BertTextClassificationTask
from task.chinesebert_text_classification_train_task import ChineseBertTextClassificationTask
from task.text_multi_task_learning_task import BertMultiClassificationTask
from datasets.basic_datasets import build_dataloader
from datasets.chinesebert_datasets import build_dataloader as build_chinesebert_dataloader
import pytorch_lightning as pl
from models.chinesebert.modeling_glycebert import GlyceBertForSequenceClassification
from transformers import BertConfig

def experiment1():
    data_path = '/Users/user/Downloads/final_train_v1.csv'

    model = BertTextClassificationTask(data_path)

    trainer = pl.Trainer(max_epochs=2)
    trainer.fit(model=model)


def experiment2():

    bert_dir = '/Users/user/Desktop/git_projects/ChineseBERT-base'
    bert_config = BertConfig.from_pretrained(bert_dir, output_hidden_states=False, num_labels=2)
    base_model = GlyceBertForSequenceClassification.from_pretrained(bert_dir, config=bert_config)
    model = ChineseBertTextClassificationTask(base_model)

    data_path = '/Users/user/Downloads/final_train_v1.csv'
    train_loader = build_chinesebert_dataloader(data_path)

    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model=model, train_dataloaders=train_loader)


def experiment3():

    data_path = '/Users/user/Downloads/final_train_v1.csv'

    model = BertMultiClassificationTask(data_path)

    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model=model)



if __name__ == '__main__':
    experiment1()
    # experiment2()