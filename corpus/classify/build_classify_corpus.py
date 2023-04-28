import json
import random
from tqdm.auto import tqdm
from util import cut

BASE_PATH = "corpus/classify/"


def save_line(line, label, file, file_by_word):
    l = " ".join(cut(line, by_word=False)) + "\t__label__" + label + "\n"
    file.write(l)

    l_by_word = " ".join(cut(line, by_word=True)) + "\t__label__" + label + "\n"
    file_by_word.write(l_by_word)


def handle_chat_corpus(file, file_by_word):
    xiaohuangji_chat_path = BASE_PATH + "小黄鸡未分词.conv"
    data = list()  # [[Q, A], [Q, A] ... [Q, A]]
    group = list()  # [Q, A]
    # 读取小黄鸡聊天数据
    with open(xiaohuangji_chat_path, "r") as chat_file:
        chat_lines = chat_file.readlines()
        for line in tqdm(chat_lines, desc="reading chat data"):
            if line.startswith("E"):
                if len(group) > 0:
                    data.append(group)
                    group = []
            elif line.startswith("M"):
                group.append(line[1:].strip())
        if len(group) > 0:
            data.append(group)

    # 保存小黄鸡聊天数据
    for group in tqdm(data, desc="saving chat data"):
        if len(group) == 2:
            question = group[0].strip().lower()
            # answer = group[1].strip().lower()
            if question != "":
                save_line(question, "chat", file, file_by_word)


def handle_qa_corpus(file, file_by_word):
    crawler_qa_path = BASE_PATH + "爬虫抓取的问题.csv"
    manual_qa_path = BASE_PATH + "手动构造的问题.json"
    qa_data = set()
    # 读取问答数据
    with open(crawler_qa_path, "r") as crawler_qa:
        for line in tqdm(crawler_qa.readlines(), desc="reading crawler QA data"):
            qa_data.add(line.strip().lower())
    for _, values in tqdm(json.load(open(manual_qa_path, "r")).items(), desc="reading manual QA data"):
        for v in values:
            for line in v:
                qa_data.add(line.strip().lower())
    # 保存问答数据
    for line in tqdm(qa_data, desc="saving QA data"):
        if len(line) > 0:
            save_line(line, "qa", file, file_by_word)


def build_classify_corpus():
    data_path = BASE_PATH + "data.txt"
    data_by_word_path = BASE_PATH + "data_by_word.txt"

    with open(data_path, "a") as file, open(data_by_word_path, "a") as file_by_word:
        # 1. 构造小黄鸡聊天数据集
        handle_chat_corpus(file, file_by_word)
        # 2. 构造问答数据集
        handle_qa_corpus(file, file_by_word)


def split_classify_corpus():
    with open(BASE_PATH + "data_train.txt", "a") as data_train, open(BASE_PATH + "data_test.txt", "a") as data_test:
        for line in tqdm(open(BASE_PATH + "data.txt", "r").readlines(), desc="splitting data set"):
            if random.random() < 0.8:
                data_train.write(line)
            else:
                data_test.write(line)

    with open(BASE_PATH + "data_by_word_train.txt", "a") as data_by_word_train, open(
        BASE_PATH + "data_by_word_test.txt", "a"
    ) as data_by_word_test:
        for line in tqdm(open(BASE_PATH + "data_by_word.txt", "r").readlines(), desc="splitting data_by_word set"):
            if random.random() < 0.8:
                data_by_word_train.write(line)
            else:
                data_by_word_test.write(line)
