from util import cut
from corpus.classify import build_classify_corpus, split_classify_corpus
from classify import train, evaluate
from classify import Classify


def util_test():
    sentence = "人工智能和python有什么关系?"
    print(cut(sentence, by_word=False))


def build_corpus_test():
    # build_classify_corpus()
    split_classify_corpus()


def classify_test():
    # train()
    # evaluate()
    classify_model = Classify()

    sentence = "银 行 风 险 经 理 面 试 经 常 遇 到 的 问 题"
    label, prob = classify_model.predict(sentence)
    print("label={} prob={}".format(label, prob))


if __name__ == "__main__":
    # util_test()
    # build_corpus_test()
    classify_test()
