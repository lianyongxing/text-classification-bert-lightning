# -*- coding: utf-8 -*-
# @Time    : 11/24/22 4:39 PM
# @Author  : LIANYONGXING
# @FileName: utils.py
# @Software: PyCharm
import zhconv
import re

def text_filtering(line):
    """
    filtering raw text
    Args:
        line:
            raw text
    Returns:
            filtered text
    """
    line = str(line)
    line = line.strip()
    line = re.sub(r'@[\w.?!,]+', '', line)
    for k in emojis.keys():
        line = line.replace(k, emojis[k])
    line = re.sub(r'\[\w+\]', '', line)  # 去除表情，例如：[微笑R]或者[泪]
    line = line.replace("\xa0", '')  # 去除\xa0
    line = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fa5]', '', line).lower()  # 去除除了汉字大小写和数字
    line = zhconv.convert(line, 'zh-cn')  # 繁体转简体
    line = line[:(line+line).find(line, 1)] # 去除循环字符串
    return line


emojis = {"💩": "屎",
          "🧠": "脑子",
          '👴': "爷",
          '㊗️': "祝",
          '🐷': "猪",
          '🍃': "叶",
          '🐖': "猪",
          '⬆️': "上",
          '🧠': "脑子",
          '⚰️': '棺材',
          '🐸': '蛙',
          "🇨🇳": "中国",
          "🐻": '熊',
          "🈲️": "禁",
          "⏩": "加速",
          "🍯": "蜂蜜",
          "1⃣️": "一",
          "7⃣️": "七",
          "🐎": "马",
          "🐴": "马",
          "🐔": '鸡',
          "❤️": "心",
          "👍": "赞",
          "🚪": "门",
          "🪖": "军盔",
          "🇺🇸": "美国",
          "🇲🇾": "马来西亚",
          "🇹🇼": "中华民国",
          "🇬🇧": "英国",
          "🇩🇪": "德国",
          "🇯🇵": "日本",
          "🇦🇺": "澳大利亚",
          "🇰🇷": "韩国",
          "🇰🇵": "朝鲜",
          "🇫🇷": "法国",
          "🇺🇦": "乌克兰",
          "🇷🇺": "俄罗斯",
          "🇨🇦": "加拿大"
          }
