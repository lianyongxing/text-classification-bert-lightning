# -*- coding: utf-8 -*-
# @Time    : 3/31/23 7:59 PM
# @Author  : LIANYONGXING
# @FileName: single_case_test.py

import torch
from models.bert import Bert
import torch.nn.functional as F


def inference(text, model):

    input_ids, input_masks, input_types = [], [], []
    encode_dict = model.tokenizer.encode_plus(text, max_length = 256, padding = 'max_length', truncation = True)
    input_ids.append(encode_dict['input_ids'])
    input_types.append(encode_dict['token_type_ids'])
    input_masks.append(encode_dict['attention_mask'])
    logits = bert_model(torch.LongTensor(input_ids), torch.LongTensor(input_masks),
                        torch.LongTensor(input_types))
    y_pred_res = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()[0]
    y_pred_prob = F.softmax(logits, dim=1).detach().cpu().numpy()[0][1]
    return y_pred_res, y_pred_prob


if __name__ == '__main__':

    bert_model = Bert(bert_path='/Users/user/Downloads/chinese_bert')
    params = torch.load('new_version2/epoch=0-step=20.ckpt')['state_dict']

    new_params = {k.replace('model.', ''): v for k, v in params.items()}
    bert_model.load_state_dict(new_params)

    text = "你点开百度地图看看沈阳周围有多少蒙古自治县蒙古人来沈阳还有顾虑真是奇了怪了诉我直言我初中几何老师是蒙古人全校教几何最好也是全校脾气最火爆的老实"

    print(inference(text, bert_model))