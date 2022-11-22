# -*- coding: utf-8 -*-
# @Time    : 11/22/22 2:35 PM
# @Author  : LIANYONGXING
# @FileName: predict.py
# @Software: PyCharm
# @Repo    : https://github.com/lianyongxing/text-classification-bert-lightning
from task.text_classification_train_task import BertTextClassificationTask
from models.bert import Bert
from datasets.basic_datasets import build_dataloader

bert_path = '/Users/user/Desktop/git_projects/text-classification-nlp-pytorch/resources/chinese_bert'
data_path = '/Users/user/Downloads/final_train_v1.csv'


model = BertTextClassificationTask.load_from_checkpoint(
    checkpoint_path="lightning_logs/version_0/checkpoints/epoch=0.ckpt",
    base_model=Bert(bert_path),
    kwargs=dict(param_name='base_model')
)
model.eval()

test_loader = build_dataloader(data_path)

import pytorch_lightning as pl

trainer = pl.Trainer()
res = trainer.predict(model, test_loader)
a = res
print(res)
