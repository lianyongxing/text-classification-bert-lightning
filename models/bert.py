# -*- coding: utf-8 -*-
# @Time    : 11/16/22 3:05 PM
# @Author  : LIANYONGXING
# @FileName: bert.py
# @Software: PyCharm
# @Repo    : https://github.com/lianyongxing/

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, BertTokenizer
import warnings

warnings.filterwarnings('ignore')

# bert_path = "/data/yxlian/tianran_data/chinese_bert"  # 该文件夹下存放三个文件（'vocab.txt', 'pytorch_model.bin', 'config.json'）

class Bert(nn.Module):

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

    # def predict(self, raw_text):
    #     text = self.text_filtering(raw_text)
    #     if text == "":
    #         return 0, 0
    #     input_ids, input_masks, input_types = [], [], []  # input char ids, segment type ids, attention mask  # 标签
    #     encode_dict = self.tokenizer.encode_plus(text, max_length=self.max_len, padding='max_length', truncation=True)
    #     input_ids.append(encode_dict['input_ids'])
    #     input_types.append(encode_dict['token_type_ids'])
    #     input_masks.append(encode_dict['attention_mask'])
    #     logits = self.forward(torch.LongTensor(input_ids).to(self.DEVICE),
    #                           torch.LongTensor(input_masks).to(self.DEVICE),
    #                           torch.LongTensor(input_types).to(self.DEVICE))
    #     y_pred_res = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()[0]
    #     y_pred_prob = F.softmax(logits, dim=1).detach().cpu().numpy()[0][1]
    #     return y_pred_res, y_pred_prob
    #
    # def predict_dataloader(self, data_loader):
    #     val_pred = []
    #     scores = []
    #     with torch.no_grad():
    #         for idx, (ids, att, tpe) in tqdm(enumerate(data_loader)):
    #             y_pred = self.forward(ids.to(self.DEVICE), att.to(self.DEVICE), tpe.to(self.DEVICE))
    #             y_pred_lab = torch.argmax(y_pred, dim=1).detach().cpu().numpy().tolist()
    #             y_pred_prob = F.softmax(y_pred, dim=1).detach().cpu().numpy()
    #             score = [y_pred_prob[idx, 1] for idx, i in enumerate(y_pred_lab)]
    #             #                 print(y_pred_prob)
    #             scores.extend(score)
    #             val_pred.extend(y_pred_lab)
    #     return val_pred, scores
    #
    # @staticmethod
    # def text_filtering(line):
    #     """
    #     filtering raw text
    #     Args:
    #         line:
    #             raw text
    #     Returns:
    #             filtered text
    #     """
    #     line = line.strip()
    #     line = re.sub(r'@[\w.?!,]+', '', line)
    #     for k in emojis.keys():
    #         line = line.replace(k, emojis[k])
    #     line = emojiswitch.demojize(line, delimiters=("", ""))
    #     line = re.sub(r'\[\w+\]', '', line)  # 去除表情，例如：[微笑R]或者[泪]
    #     line = line.replace("\xa0", '')  # 去除\xa0
    #     line = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fa5]', '', line).lower()  # 去除除了汉字大小写和数字
    #     line = zhconv.convert(line, 'zh-cn')  # 繁体转简体
    #     line = line[:(line + line).find(line, 1)]  # 去除循环字符串
    #     return line


# emojis = {"💩": "屎",
#           "🧠": "脑子",
#           '👴': "爷",
#           '㊗️': "祝",
#           '🐷': "猪",
#           '🍃': "叶",
#           '⬆️': "上",
#           '🧠': "脑子",
#           '🐸': '蛙',
#           "🐻": '熊',
#           "🈲️": "禁",
#           "⏩": "加速",
#           "🍯": "蜂蜜",
#           "1⃣️": "一",
#           "7⃣️": "七",
#           "🐴": "马",
#           "❤️": "心",
#           "👍": "赞",
#           "👮‍♀️": "警察",
#           "👮": "警察",
#           "🐭": "鼠",
#           "🐶": "狗",
#           "👀": "眼",
#           "🐢": "龟",
#           "🌷": "",
#           "💰": "钱",
#           "🙊": "",
#           "🍠": "小红书",
#           "😰": "",
#           "🙏🏻": "",
#           "💪": "",
#           "👎": "",
#           "🐑人": "阳性患者",
#           "🚔": "警",
#           "☂️": "伞",
#           "🀄️": "中",
#           "🌂": "伞",
#           "📖": "书",
#           "👊🏻": "拳打",
#           "🥺": ""
#           }