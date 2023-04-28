import fasttext
import numpy as np

fasttext.FastText.eprint = lambda x: None


def get_dataset_path(by_word=True, train=True):
    if by_word:
        return "corpus/classify/data_by_word_train.txt" if train else "corpus/classify/data_by_word_test.txt"
    else:
        return "corpus/classify/data_train.txt" if train else "corpus/classify/data_test.txt"


def train():
    train_data_path = get_dataset_path(by_word=True, train=True)
    model = fasttext.train_supervised(train_data_path, dim=100, epoch=20, wordNgrams=2)
    # model.save_model("classify/models/classify_by_word_100_20_1.model")  # 0.994214320038443
    model.save_model("classify/models/classify_by_word_100_20_2.model")  # 0.9967996155694377
    # model.save_model("classify/models/classify_100_20_1.model")  # 0.9960127130887027
    # model.save_model("classify/models/classify_100_20_2.model")  # 0.9964846383511509


def evaluate():
    test_data_path = get_dataset_path(by_word=True, train=False)
    model = fasttext.load_model("classify/models/classify_by_word_100_20_2.model")

    sentences = []
    labels = []
    with open(test_data_path, "r") as test_data:
        for line in test_data.readlines():
            line = line.strip()
            tmp = line.split("\t")
            if len(tmp) == 2:
                sentences.append(tmp[0])
                labels.append(tmp[1])

    predict = model.predict(sentences)[0]
    predict = [p[0] for p in predict]
    assert len(labels) == len(predict)

    acc = np.mean([1 if labels[i] == predict[i] else 0 for i in range(len(labels))])
    print(acc)
