import fasttext

import config
from util import cut

fasttext.FastText.eprint = lambda x: None


class Classify:
    def __init__(self) -> None:
        self.model = fasttext.load_model(config.classify_model_path)

    def predict(self, sentence, by_word=True, use_stopwords=False) -> str:
        sentence = " ".join(cut(sentence, by_word, use_stopwords))
        ret = self.model.predict(sentence)
        predict_label = ret[0][0]
        predict_prob = ret[-1][0]
        return predict_label, predict_prob
