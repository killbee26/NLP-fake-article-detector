"""TF-IDF (1-2 grams) + Logistic Regression baseline, exported as plain JSON for Kotlin.

The tokenizer is deliberately simple (lowercase, [a-z0-9]+ tokens) so Kotlin can reproduce it exactly.
"""
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from common import ARTIFACTS, DATA, tfidf_tokenize

train = pd.read_parquet(DATA / "train.parquet")
test = pd.read_parquet(DATA / "test.parquet")

vec = TfidfVectorizer(tokenizer=tfidf_tokenize, lowercase=False, token_pattern=None,
                      ngram_range=(1, 2), max_features=30000, min_df=3, sublinear_tf=True)
Xtr = vec.fit_transform(train["content"])
clf = LogisticRegression(C=4.0, max_iter=2000)
clf.fit(Xtr, train["label"])

pred = clf.predict(vec.transform(test["content"]))
print(classification_report(test["label"], pred, target_names=["FAKE", "REAL"], digits=4))

vocab = {term: int(i) for term, i in vec.vocabulary_.items()}
export = {
    "ngram_max": 2,
    "sublinear_tf": True,
    "vocab": vocab,
    "idf": np.round(vec.idf_, 6).tolist(),
    "coef": np.round(clf.coef_[0], 6).tolist(),   # positive -> REAL (label 1)
    "intercept": float(clf.intercept_[0]),
}
(ARTIFACTS / "tfidf_lr.json").write_text(json.dumps(export))
with open(DATA / "tfidf_lr.pkl", "wb") as f:
    pickle.dump((vec, clf), f)
print("saved", ARTIFACTS / "tfidf_lr.json")
