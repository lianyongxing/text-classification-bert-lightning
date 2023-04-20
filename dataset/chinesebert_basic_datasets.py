# -*- coding: utf-8 -*-
# @Time    : 4/19/23 5:01 PM
# @Author  : LIANYONGXING
# @FileName: chinesebert_basic_datasets.py

import json
import os
from typing import List

import tokenizers
from pypinyin import pinyin, Style
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import Dataset
import torch


class ChineseBertDataset(Dataset):

    def __init__(self, chinese_bert_path, max_length: int = 512):
        """
        Dataset Base class
        Args:
            data_path: dataset file path
            chinese_bert_path: pretrain model path
            max_length: max sentence length
        """
        super().__init__()
        self.vocab_file = os.path.join(chinese_bert_path, 'vocab.txt')
        self.config_path = os.path.join(chinese_bert_path, 'config')
        self.max_length = max_length
        self.tokenizer = BertWordPieceTokenizer(self.vocab_file)
        # load pinyin map dict
        with open(os.path.join(self.config_path, 'pinyin_map.json'), encoding='utf8') as fin:
            self.pinyin_dict = json.load(fin)
        # load char id map tensor
        with open(os.path.join(self.config_path, 'id2pinyin.json'), encoding='utf8') as fin:
            self.id2pinyin = json.load(fin)
        # load pinyin map tensor
        with open(os.path.join(self.config_path, 'pinyin2tensor.json'), encoding='utf8') as fin:
            self.pinyin2tensor = json.load(fin)

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

    def encode_input_text(self, text):
        """encoding single sentence"""
        tokenizer_output = self.tokenizer.encode(text)
        bert_tokens = tokenizer_output.ids
        pinyin_tokens = self.convert_sentence_to_pinyin_ids(text, tokenizer_output)
        # 验证正确性，id个数应该相同
        assert len(bert_tokens) <= self.max_length
        assert len(bert_tokens) == len(pinyin_tokens)
        # 转化list为tensor
        input_ids = torch.LongTensor(bert_tokens)
        pinyin_ids = torch.LongTensor(pinyin_tokens).view(-1)
        attention_mask = (input_ids != 0).long()

        input_ids = input_ids.unsqueeze(0)
        pinyin_ids = pinyin_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
        return input_ids, pinyin_ids, attention_mask