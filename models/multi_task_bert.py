# -*- coding: utf-8 -*-
# @Time    : 11/24/22 4:38 PM
# @Author  : LIANYONGXING
# @FileName: multi_task_bert.py
# @Software: PyCharm
# @Repo    : https://github.com/lianyongxing/

import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, BertTokenizer
import warnings

warnings.filterwarnings('ignore')

# 定义model
class MultiTaskBert(pl.LightningModule):

    def __init__(self, bert_path, classes=7):
        super(MultiTaskBert, self).__init__()

        self.max_len = 256
        self.classes_num = classes
        self.tokenizer = BertTokenizer.from_pretrained(bert_path, tokenize_chinese_chars=True)

        self.config = BertConfig.from_pretrained(bert_path)  # 导入模型超参数
        self.bert = BertModel.from_pretrained(bert_path)  # 加载预训练模型权重
        self.fc = nn.Linear(self.config.hidden_size, 2)  # 直接分类是否涉政
        self.heads = nn.ModuleList([nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
        ) for _ in range(self.classes_num - 1)])
        self.labs_cls = nn.ModuleList([nn.Sequential(
            nn.Linear(self.config.hidden_size, 1),
        ) for _ in range(self.classes_num - 1)])

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        out_pool = outputs[1]  # 池化后的输出 [bs, config.hidden_size]
        logit_mtask = self.fc(out_pool)  # [bs, classes] 是否涉政
        logit1 = [(out_pool + head(out_pool)) / 2 for head in self.heads]
        logit_stask = torch.stack([self.labs_cls[idx](lg) for idx, lg in enumerate(logit1)]).squeeze(2).t()

        return logit_mtask, logit_stask


if __name__ == '__main__':

    bert_path = "/data/yxlian/tianran_data/chinese_bert"
    tokenizer = BertTokenizer.from_pretrained(bert_path, tokenize_chinese_chars=True)  # 初始化分词器

    model = MultiTaskBert(bert_path)
    model.predict("今天天气很不错哦")
    print(model)