# -*- coding: utf-8 -*-
# @Time    : 11/16/22 3:05 PM
# @Author  : LIANYONGXING
# @FileName: bert.py
# @Software: PyCharm
# @Repo    : https://github.com/lianyongxing/text-classification-bert-lightning

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, BertTokenizer
import warnings
import pytorch_lightning as pl


warnings.filterwarnings('ignore')

# bert_path = "/data/yxlian/tianran_data/chinese_bert"  # 该文件夹下存放三个文件（'vocab.txt', 'pytorch_model.bin', 'config.json'）

class Bert(pl.LightningModule):

    def __init__(self, bert_path, classes=2):
        super(Bert, self).__init__()
        self.config = BertConfig.from_pretrained(bert_path)  # 导入模型超参数
        self.bert = BertModel.from_pretrained(bert_path, output_attentions=True)  # 加载预训练模型权重
        self.fc = nn.Linear(self.config.hidden_size, classes)  # 直接分类
        self.tokenizer = BertTokenizer.from_pretrained(bert_path, tokenize_chinese_chars=True)
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_len = 128

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        out_pool = outputs[1]  # 池化后的输出 [bs, config.hidden_size]
        logit = self.fc(out_pool)  # [bs, classes]
        return logit