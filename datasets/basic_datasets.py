# -*- coding: utf-8 -*-
# @Time    : 11/16/22 3:36 PM
# @Author  : LIANYONGXING
# @FileName: xhs_datasets.py
# @Software: PyCharm
# @Repo    : https://github.com/lianyongxing/text-classification-bert-lightning
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torch.utils.data.dataloader import DataLoader
import pandas as pd


class BasicDataset(Dataset):

    def __init__(self, encodings, labs):
        self.encodings = encodings
        self.labs = labs

    def __getitem__(self, idx):
        item = {key: torch.LongTensor(val[idx]) for key, val in self.encodings.items()}
        item['label'] = torch.LongTensor([float(i) for i in self.labs])[idx]
        return item

    def __len__(self):
        return len(self.labs)

    @classmethod
    def get_labels(cls, ):
        return [0, 1]


def build_dataloader(fp, batch_size=128):
    datas = pd.read_csv(fp)
    texts = datas['content_filter'].tolist()
    labs = datas['lab'].tolist()

    tokenizer = BertTokenizer.from_pretrained(
        '/Users/user/Desktop/git_projects/text-classification-nlp-pytorch/resources/chinese_bert')
    train_encodings = tokenizer(texts, max_length=30, padding='max_length', truncation=True)
    train_dataset = BasicDataset(train_encodings, labs)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader

