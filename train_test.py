# -*- coding: utf-8 -*-
# @Time    : 11/22/22 3:21 PM
# @Author  : LIANYONGXING
# @FileName: model.py
# @Software: PyCharm

# from task.text_classification_train_task import BertTextClassificationTask
# from task.chinesebert_text_classification_train_task import ChineseBertTextClassificationTask
# from task.text_multi_task_learning_task import BertMultiClassificationTask
# from datasets.chinesebert_datasets import build_dataloader as build_chinesebert_dataloader
# import pytorch_lightning as pl
from task.text_classification_train_task import BertTextClassificationTask
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

def experiment1():
    data_path = '/Users/user/Desktop/badcase_analyse/leader_notes/leader_note_train_datas.csv'
    bert_path = '/data/yxlian/tianran_data/chinese_bert'
    log_path = 'lightning_logs/v2_2_5/'

    checkpoint_callback = ModelCheckpoint(dirpath=log_path, every_n_epochs=1,
                                          save_on_train_epoch_end=True, save_top_k=-1)

    model = BertTextClassificationTask(data_path, bert_path)

    trainer = pl.Trainer(max_epochs=5, gpus=2, callbacks=[checkpoint_callback])
    trainer.fit(model=model)


def experiment2():

    # bert_dir = '/Users/user/Desktop/git_projects/ChineseBERT-base'
    # bert_config = BertConfig.from_pretrained(bert_dir, output_hidden_states=False, num_labels=2)
    # base_model = GlyceBertForSequenceClassification.from_pretrained(bert_dir, config=bert_config)
    data_path = '/Users/user/Downloads/final_train_v1.csv'
    model = ChineseBertTextClassificationTask(data_path)

    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model=model)


def experiment3():

    data_path = 'testtest.csv'

    model = BertMultiClassificationTask(data_path)

    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model=model)



if __name__ == '__main__':
    # experiment1()
    experiment2()