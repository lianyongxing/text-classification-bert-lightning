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
from sklearn.model_selection import train_test_split


class BasicDataset(Dataset):

    def __init__(self, encodings, labs):
        self.encodings = encodings
        self.labs = labs
        self.final_labs = torch.LongTensor([float(i) for i in self.labs])

    def __getitem__(self, idx):
        item = {key: torch.LongTensor(val[idx]) for key, val in self.encodings.items()}
        item['label'] = self.final_labs[idx]
        return item

    def __len__(self):
        return len(self.labs)

    @classmethod
    def get_labels(cls, ):
        return [0, 1]

def _build_dataloader(sentences, labs, tokenizer, max_length, batch_size, shuffle=True):
    encodings = tokenizer(sentences, max_length=max_length, padding='max_length', truncation=True)
    dataset = BasicDataset(encodings, labs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def build_test_dataloader(fp, max_length=30, batch_size=128, bert_path=None):
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    test_datas = pd.read_csv(fp)
    test_texts = test_datas['content_filter'].tolist()
    test_dataloader = _build_dataloader(test_texts, [99]*len(test_texts), tokenizer, max_length, batch_size, shuffle=False)
    return test_dataloader


def build_dataloader(fp, max_length=30, batch_size=128, bert_path=None):
    datas = pd.read_csv(fp)[:1000]

    tokenizer = BertTokenizer.from_pretrained(bert_path)

    train_datas, valid_datas = train_test_split(datas, test_size=0.2, random_state=20)

    train_texts = train_datas['content_filter'].tolist()
    train_labs = train_datas['lab'].tolist()
    train_dataloader = _build_dataloader(train_texts, train_labs, tokenizer, max_length, batch_size)

    valid_texts = valid_datas['content_filter'].tolist()
    valid_labs = valid_datas['lab'].tolist()
    valid_dataloader = _build_dataloader(valid_texts, valid_labs, tokenizer, max_length, batch_size)

    return train_dataloader, valid_dataloader


if __name__ == "__main__":
    build_dataloader('/Users/user/Downloads/final_train_v1.csv')