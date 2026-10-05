import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_vectorizer(max_features=10000, ngram_range=(1, 2), min_df=3, max_df=0.85):
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
        stop_words="english"
    )

def fit_and_save_vectorizer(corpus, save_path="models/tfidf_vectorizer.joblib"):
    vec = get_tfidf_vectorizer()
    vec.fit(corpus)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, save_path)
    return vec

def load_vectorizer(load_path="models/tfidf_vectorizer.joblib"):
    return joblib.load(load_path)
