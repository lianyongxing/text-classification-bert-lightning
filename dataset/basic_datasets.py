# -*- coding: utf-8 -*-
# @Time    : 11/16/22 3:36 PM
# @Author  : LIANYONGXING
# @FileName: basic_datasets.py
# @Software: PyCharm

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torch.utils.data.dataloader import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, SequentialSampler
import numpy as np
from tqdm import tqdm


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


def build_test_dataloader(fp, max_length, batch_size, bert_path):
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    test_datas = pd.read_csv(fp)

    input_ids, input_masks, input_types, = [], [], []
    for idx, line in tqdm(test_datas.iterrows()):
        sent = line['content_filter']
        encode_dict = tokenizer.encode_plus(text=sent, max_length=max_length, padding='max_length', truncation=True)
        input_ids.append(encode_dict['input_ids'])
        input_types.append(encode_dict['token_type_ids'])
        input_masks.append(encode_dict['attention_mask'])

    input_ids, input_types, input_masks = np.array(input_ids), np.array(input_types), np.array(input_masks)
    test_data = TensorDataset(torch.LongTensor(input_ids),
                              torch.LongTensor(input_masks),
                              torch.LongTensor(input_types)
                              )
    test_sampler = SequentialSampler(test_data)
    test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    return test_loader


def build_dataloader(fp, max_length, batch_size, bert_path, test_ratio=0.2):
    datas = pd.read_csv(fp)
    datas = datas[~datas['content_filter'].isna()]

    tokenizer = BertTokenizer.from_pretrained(bert_path, use_fast=True)

    train_datas, valid_datas = train_test_split(datas, test_size=test_ratio, random_state=20)
    valid_datas.to_csv('valid_datas.csv')

    train_texts = train_datas['content_filter'].tolist()
    train_labs = train_datas['lab'].tolist()
    train_dataloader = _build_dataloader(train_texts, train_labs, tokenizer, max_length, batch_size)

    valid_texts = valid_datas['content_filter'].tolist()
    valid_labs = valid_datas['lab'].tolist()
    valid_dataloader = _build_dataloader(valid_texts, valid_labs, tokenizer, max_length, batch_size)

    return train_dataloader, valid_dataloader


if __name__ == "__main__":
    pass