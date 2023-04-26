from config import stopwords_path


stopwords = list(set([w.strip() for w in open(stopwords_path, "r").readlines()]))
