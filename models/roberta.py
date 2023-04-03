# -*- coding: utf-8 -*-
# @Time    : 4/3/23 10:35 AM
# @Author  : LIANYONGXING
# @FileName: roberta.py

import torch.nn as nn
from transformers import BertModel, BertConfig, BertTokenizer
import pytorch_lightning as pl
import os
import torch


class Roberta(pl.LightningModule):
    def __init__(self, bert_path, classes=2):
        super(Roberta, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.config = BertConfig.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path, config=self.config)

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.config.hidden_size, classes)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, head_mask=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                                        head_mask=head_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def generate_jit_model():
    data_dir = ""
    raw_checkpoint_path = os.path.join(data_dir, "")
    jit_model_path = os.path.join(data_dir, "")

    model = Roberta.from_pretrained(raw_checkpoint_path, num_labels=2)


    input_ids = [  101,  1728,   711,  2571,  4633,  1825,   749,   138, 13030,  8154,
                    140,  1744,  2412,  5310,  3338,   749,  8024,   711,   784,   720,
                    392,  1920,  8172,  5520,  3250,  6963,  1359,  5273,   749,  8043,
                    102]

    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.unsqueeze(0) # Batch size 1, 2 choices
    print(input_ids)

    model = model.cuda()
    input_ids = input_ids.cuda()

    script_module = torch.jit.trace(model,input_ids,strict=False)
    torch.jit.save(script_module, jit_model_path)

if __name__ == '__main__':
    generate_jit_model()
