# -*- coding: utf-8 -*-
# @Time    : 11/24/22 5:29 PM
# @Author  : LIANYONGXING
# @FileName: mlt_datasets.py
# @Software: PyCharm
# @Repo    : https://github.com/lianyongxing/
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torch.utils.data.dataloader import DataLoader
import pandas as pd


class BasicDataset(Dataset):

    def __init__(self, encodings, labs, sublabs):
        self.encodings = encodings
        self.labs = labs
        self.sublabs = sublabs

    def __getitem__(self, idx):
        item = {key: torch.LongTensor(val[idx]) for key, val in self.encodings.items()}
        item['label'] = torch.LongTensor([float(i) for i in self.labs])[idx]
        item['sub_label'] = torch.LongTensor([float(i) for i in self.sublabs])[idx]
        return item

    def __len__(self):
        return len(self.labs)


def build_dataloader(fp, batch_size=128, max_len=256):
    datas = pd.read_csv(fp)
    texts = datas['content_filter'].tolist()
    labs = datas['lab'].tolist()
    sub_labs = datas['sublab'].tolist()

    tokenizer = BertTokenizer.from_pretrained(
        '/Users/user/Desktop/git_projects/text-classification-nlp-pytorch/resources/chinese_bert')
    train_encodings = tokenizer(texts, max_length=max_len, padding='max_length', truncation=True)
    train_dataset = BasicDataset(train_encodings, labs, sub_labs)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader

