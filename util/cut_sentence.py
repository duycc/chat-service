import logging
import string
import jieba
import jieba.posseg as psg
import re

from util import stopwords
from config import keywords_path


jieba.setLogLevel(logging.INFO)
jieba.load_userdict(keywords_path)

letters = string.ascii_letters
filters = [",", "-", ".", " "]


def _cut_by_word(sentence):
    # 中文按字分词，英文按单词分词
    sentence = re.sub("\s+", " ", sentence).strip()
    result = []
    temp = ""
    for word in sentence:
        if word.lower() in letters:
            temp += word.lower()
        else:
            if temp != "":
                result.append(temp)
                temp = ""
            if word in filters:
                continue
            result.append(word)
    if temp != "":
        result.append(temp)
    return result


def _cut(sentence, use_stopwords, use_seg):
    if not use_seg:
        result = jieba.lcut(sentence)
    else:
        result = [(p.word, p.flag) for p in psg.cut(sentence)]
    if use_stopwords:
        if use_seg:
            result = [p for p in result if p[0] not in stopwords]
        else:
            result = [w for w in result if w not in stopwords]
    return result


def cut(sentence: str, by_word=False, use_stopwords=False, use_seg=False) -> list:
    if by_word:
        return _cut_by_word(sentence)
    else:
        return _cut(sentence, use_stopwords, use_seg)
