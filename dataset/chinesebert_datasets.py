# -*- coding: utf-8 -*-
# @Time    : 4/19/23 5:29 PM
# @Author  : LIANYONGXING
# @FileName: leadernote_datasets.py
from functools import partial
import torch
from torch.utils.data import DataLoader
from dataset.chinesebert_basic_datasets import ChineseBertDataset
from dataset.collate_functions import collate_to_max_length
import pandas as pd
from sklearn.model_selection import train_test_split


class BasicDataset(ChineseBertDataset):

    def __init__(self, datas, chinese_bert_path, max_length: int = 512):
        super().__init__(chinese_bert_path, max_length)
        self.lable_map = {0: 0, 1: 1}
        self.texts = datas['content_filter'].tolist()
        self.labs = datas['lab'].tolist()

    def __len__(self):
        return len(self.labs)

    def __getitem__(self, idx):
        line = self.texts[idx]
        sentence = line[:self.max_length - 2]
        # 将句子转为ids
        tokenizer_output = self.tokenizer.encode(sentence)
        bert_tokens = tokenizer_output.ids
        pinyin_tokens = self.convert_sentence_to_pinyin_ids(sentence, tokenizer_output)
        # 验证正确性，id个数应该相同
        assert len(bert_tokens) <= self.max_length
        assert len(bert_tokens) == len(pinyin_tokens)
        # 转化list为tensor
        input_ids = torch.LongTensor(bert_tokens)
        pinyin_ids = torch.LongTensor(pinyin_tokens).view(-1)
        label = torch.LongTensor([int(self.labs[idx])])
        return input_ids, pinyin_ids, label


def build_dataloader(fp, bert_path, max_len, batch_size):
    datas = pd.read_csv(fp)
    datas = datas[~datas['content_filter'].isna()]

    train_datas, valid_datas = train_test_split(datas, test_size=0.2, random_state=20)

    trainset = BasicDataset(train_datas, bert_path, max_len)
    validset = BasicDataset(valid_datas, bert_path, max_len)

    train_loader = DataLoader(
        dataset=trainset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
        collate_fn=partial(collate_to_max_length, fill_values=[0, 0, 0])
    )
    valid_loader = DataLoader(
        dataset=validset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        collate_fn=partial(collate_to_max_length, fill_values=[0, 0, 0])
    )

    return train_loader, valid_loader


def build_test_dataloader(fp, max_length, batch_size, bert_path):
    datas = pd.read_csv(fp)
    datas = datas[~datas['content_filter'].isna()]
    # mock label
    datas['lab'] = 1
    testset = BasicDataset(datas, bert_path, max_length)
    test_loader = DataLoader(
        dataset=testset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        collate_fn=partial(collate_to_max_length, fill_values=[0, 0, 0])
    )
    return test_loader


def unit_test():
    data_path = '/Users/user/Downloads/final_train_v1.csv'
    chinese_bert_path = "/Users/user/Desktop/git_projects/ChineseBERT-base"

    datas = pd.read_csv(data_path)[:1000]
    datas = datas[~datas['content_filter'].isna()]

    dataset = BasicDataset(datas, chinese_bert_path=chinese_bert_path)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=10,
        num_workers=0,
        shuffle=True,
        collate_fn=partial(collate_to_max_length, fill_values=[0, 0, 0])
    )
    for input_ids, pinyin_ids, label in dataloader:
        bs, length = input_ids.shape
        print(input_ids.shape)
        print(pinyin_ids.reshape(bs, length, -1).shape)
        print(label.view(-1).shape)
        print()


if __name__ == '__main__':
    unit_test()