# -*- coding: utf-8 -*-
# @Time    : 12/13/22 5:58 PM
# @Author  : LIANYONGXING
# @FileName: chinesebert_test.py
# @Software: PyCharm
# @Repo    : https://github.com/lianyongxing/

import argparse

# from datasets.bert_dataset import BertDataset
from modeling_glycebert import GlyceBertModel

# def sentence_hidden():
#     # init args
#     parser = argparse.ArgumentParser(description="Chinese Bert Hidden")
#     parser.add_argument("--pretrain_path", required=True, type=str, help="pretrain model path")
#     parser.add_argument("--sentence", required=True, type=str, help="input sentence")
#     args = parser.parse_args()
#
#     # step 1: tokenizer
#     tokenizer = BertDataset(args.pretrain_path)
#
#     # step 2: load model
#     chinese_bert = GlyceBertModel.from_pretrained(args.pretrain_path)
#
#     # step 3: get hidden
#     input_ids, pinyin_ids = tokenizer.tokenize_sentence(args.sentence)
#     length = input_ids.shape[0]
#     input_ids = input_ids.view(1, length)
#     pinyin_ids = pinyin_ids.view(1, length, 8)
#     output_hidden = chinese_bert.forward(input_ids, pinyin_ids)[0]
#     print(output_hidden)
from transformers import BertConfig
from modeling_glycebert import GlyceBertForSequenceClassification


bert_dir = '/Users/user/Desktop/git_projects/ChineseBERT-base'
bert_config = BertConfig.from_pretrained(bert_dir, output_hidden_states=False, num_labels=2)
base_model = GlyceBertForSequenceClassification.from_pretrained(bert_dir, config=bert_config)
from datasets.chinesebert_datasets import ChineseBertTokenEncoder
from task.chinesebert_text_classification_train_task import ChineseBertTextClassificationTask
tokenizer = ChineseBertTokenEncoder(bert_dir)
sentence = "点开百度地图看看"

model = ChineseBertTextClassificationTask(base_model)

input_ids, pinyin_ids = tokenizer.tokenize_sentence(sentence)
# # ids = ids.reshape(1, -1)
# ids = ids.view(1, -1)
# pinyins = pinyins.reshape(1, -1)
#
# batch, length = ids.shape
# pinyin_ids = pinyins.view(batch, length, 8)

length = input_ids.shape[0]
# input_ids = input_ids.view(1, length)
# pinyin_ids = pinyin_ids.view(1, length, 8)
input_ids = input_ids.reshape(1, -1)

pinyin_ids = pinyin_ids.view(1, length, 8)

output_hidden = model.forward(input_ids, pinyin_ids)[0]
print(output_hidden)

# attention_mask = (ids != 0).long()

# res2 = base_model(ids, pinyins, attention_mask)
# print(res2)

# length = input_ids.shape[0]
# input_ids = input_ids.view(1, length)
# pinyin_ids = pinyin_ids.view(1, length, 8)
# output_hidden = chinese_bert.forward(input_ids, pinyin_ids)[0]
# print(output_hidden)