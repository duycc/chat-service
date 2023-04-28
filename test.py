from util import cut
from corpus.classify import build_classify_corpus, split_classify_corpus
from classify import train, evaluate


def util_test():
    sentence = "人工智能和python有什么关系?"
    print(cut(sentence, by_word=False))


def build_corpus_test():
    # build_classify_corpus()
    split_classify_corpus()


def classify_test():
    train()
    evaluate()


if __name__ == "__main__":
    # util_test()
    # build_corpus_test()
    classify_test()
