from util import cut


def util_test():
    sentence = "人工智能和python有什么关系?"
    print(cut(sentence, by_word=False))


if __name__ == "__main__":
    util_test()
