# -*- coding: utf-8 -*-
# @Time    : 12/13/22 6:01 PM
# @Author  : LIANYONGXING
# @FileName: chinesebert_datasets.py
# @Software: PyCharm
# @Repo    : https://github.com/lianyongxing/
import json
import os
from typing import List
import pandas as pd
import tokenizers
from functools import partial
import torch
from pypinyin import pinyin, Style
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from datasets.collate_functions import collate_to_max_length

class ChineseBertDataset(Dataset):

    def __init__(self, encodings, labs):
        self.encodings = encodings
        self.labs = labs
        self.max_len = None

    def __getitem__(self, idx):
        item = {key: torch.LongTensor(val[idx]) for key, val in self.encodings.items()}
        item['label'] =  torch.LongTensor([int(self.labs[idx])])

        return item['ids'], item['pinyins'], item['label']

    def __len__(self):
        return len(self.labs)

class ChineseBertTokenEncoder(object):

    def __init__(self, bert_path, max_length: int = 512):
        super().__init__()
        vocab_file = os.path.join(bert_path, 'vocab.txt')
        config_path = os.path.join(bert_path, 'config')
        self.max_length = max_length
        self.tokenizer = BertWordPieceTokenizer(vocab_file)

        # load pinyin map dict
        with open(os.path.join(config_path, 'pinyin_map.json'), encoding='utf8') as fin:
            self.pinyin_dict = json.load(fin)
        # load char id map tensor
        with open(os.path.join(config_path, 'id2pinyin.json'), encoding='utf8') as fin:
            self.id2pinyin = json.load(fin)
        # load pinyin map tensor
        with open(os.path.join(config_path, 'pinyin2tensor.json'), encoding='utf8') as fin:
            self.pinyin2tensor = json.load(fin)

    def tokenize_sentence(self, sentence):
        # convert sentence to ids
        tokenizer_output = self.tokenizer.encode(sentence)
        bert_tokens = tokenizer_output.ids
        pinyin_tokens = self.convert_sentence_to_pinyin_ids(sentence, tokenizer_output)
        # assert???token nums should be same as pinyin token nums
        assert len(bert_tokens) <= self.max_length
        assert len(bert_tokens) == len(pinyin_tokens)
        # convert list to tensor
        input_ids = torch.LongTensor(bert_tokens)
        # input_ids = bert_tokens
        # pinyin_ids = sum(pinyin_tokens, [])
        pinyin_ids = torch.LongTensor(pinyin_tokens).view(-1)
        return input_ids, pinyin_ids

    def convert_sentence_to_pinyin_ids(self, sentence: str, tokenizer_output: tokenizers.Encoding) -> List[List[int]]:
        # get pinyin of a sentence
        pinyin_list = pinyin(sentence, style=Style.TONE3, heteronym=True, errors=lambda x: [['not chinese'] for _ in x])
        pinyin_locs = {}
        # get pinyin of each location
        for index, item in enumerate(pinyin_list):
            pinyin_string = item[0]
            # not a Chinese character, pass
            if pinyin_string == "not chinese":
                continue
            if pinyin_string in self.pinyin2tensor:
                pinyin_locs[index] = self.pinyin2tensor[pinyin_string]
            else:
                ids = [0] * 8
                for i, p in enumerate(pinyin_string):
                    if p not in self.pinyin_dict["char2idx"]:
                        ids = [0] * 8
                        break
                    ids[i] = self.pinyin_dict["char2idx"][p]
                pinyin_locs[index] = ids

        # find chinese character location, and generate pinyin ids
        pinyin_ids = []
        for idx, (token, offset) in enumerate(zip(tokenizer_output.tokens, tokenizer_output.offsets)):
            if offset[1] - offset[0] != 1:
                pinyin_ids.append([0] * 8)
                continue
            if offset[0] in pinyin_locs:
                pinyin_ids.append(pinyin_locs[offset[0]])
            else:
                pinyin_ids.append([0] * 8)

        return pinyin_ids


def build_dataloader(fp, batch_size=16):
    datas = pd.read_csv(fp)[:1000]
    # datas = datas[datas['content_filter'].str.len() <= 400]
    texts = datas['content_filter'].tolist()    # ?????????????????????512???
    texts = [i[: 510] for i in texts]
    labs = datas['lab'].tolist()

    train_encodings = {'ids': [], 'pinyins': []}
    tokenEncoder = ChineseBertTokenEncoder('/Users/user/Desktop/git_projects/ChineseBERT-base')

    for text in texts:
        input_id, pinyin_id = tokenEncoder.tokenize_sentence(text)
        train_encodings['ids'].append(input_id)
        train_encodings['pinyins'].append(pinyin_id)

    train_dataset = ChineseBertDataset(train_encodings, labs)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=partial(collate_to_max_length, fill_values=[0, 0, 0]))
    return train_dataloader


if __name__ == '__main__':
    data_path = '/Users/user/Downloads/final_train_v1.csv'
    train_loader = build_dataloader(data_path, 16)
    print(next(iter(train_loader)))
