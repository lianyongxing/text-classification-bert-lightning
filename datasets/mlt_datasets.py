# -*- coding: utf-8 -*-
# @Time    : 11/24/22 5:29 PM
# @Author  : LIANYONGXING
# @FileName: mlt_datasets.py
# @Software: PyCharm
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torch.utils.data.dataloader import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split


class BasicDataset(Dataset):

    def __init__(self, encodings, labs, sublabs):
        self.encodings = encodings
        self.labs = labs
        self.sublabs = sublabs
        self.item_labs = torch.LongTensor([float(i) for i in self.labs])
        self.item_sublabs = torch.LongTensor([float(i) for i in self.sublabs])

    def __getitem__(self, idx):
        item = {key: torch.LongTensor(val[idx]) for key, val in self.encodings.items()}
        item['label'] = torch.FloatTensor([0.0, 0.0])
        item['label'][self.item_labs[idx]] = 1.0
        item['sub_label'] = torch.FloatTensor([0.0  for _ in range(6)])
        if int(self.item_sublabs[idx]) != 0:
            item['sub_label'][self.item_sublabs[idx]-1] = 1.0
        return item

    def __len__(self):
        return len(self.labs)

def _build_dataloader(texts, labs, sublabs, tokenizer, max_len, batch_size):
    encodings = tokenizer(texts, max_length=max_len, padding='max_length', truncation=True)
    dataset = BasicDataset(encodings, labs, sublabs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def build_dataloader(fp, batch_size=128, max_len=256):
    datas = pd.read_csv(fp)
    datas['lab'] = datas['sublab'].apply(lambda x: 0 if x==0 else 1)
    datas = datas[:1000]

    tokenizer = BertTokenizer.from_pretrained(
        '/Users/user/Desktop/git_projects/text-classification-nlp-pytorch/resources/chinese_bert')

    train_datas, valid_datas = train_test_split(datas, test_size=0.1, random_state=20)

    train_texts = train_datas['content_filter'].tolist()
    train_labs = train_datas['lab'].tolist()
    train_sub_labs = train_datas['sublab'].tolist()
    train_dataloader = _build_dataloader(train_texts, train_labs, train_sub_labs, tokenizer, max_len, batch_size)

    valid_texts = valid_datas['content_filter'].tolist()
    valid_labs = valid_datas['lab'].tolist()
    valid_sub_labs = valid_datas['sublab'].tolist()
    valid_dataloader = _build_dataloader(valid_texts, valid_labs, valid_sub_labs, tokenizer, max_len, batch_size)
    return train_dataloader, valid_dataloader

