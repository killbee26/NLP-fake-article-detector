"""Evaluate TF-IDF+LR, DistilBERT fp32 ONNX and int8 ONNX on the held-out test set (CPU only)."""
import json
import os
import pickle
import time

import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from transformers import AutoTokenizer

from common import ARTIFACTS, DATA, MAX_LEN, MODEL_DIR

test = pd.read_parquet(DATA / "test.parquet")
texts, y = test["content"].tolist(), test["label"].to_numpy()
tok = AutoTokenizer.from_pretrained(MODEL_DIR)


def summarize(name, preds, size_mb, latency_ms):
    p, r, f1, _ = precision_recall_fscore_support(y, preds, average=None, labels=[0, 1])
    _, _, macro_f1, _ = precision_recall_fscore_support(y, preds, average="macro")
    res = {
        "name": name,
        "accuracy": round(float(accuracy_score(y, preds)), 4),
        "precision_fake": round(float(p[0]), 4), "recall_fake": round(float(r[0]), 4),
        "precision_real": round(float(p[1]), 4), "recall_real": round(float(r[1]), 4),
        "f1_macro": round(float(macro_f1), 4),
        "confusion": confusion_matrix(y, preds, labels=[0, 1]).tolist(),  # rows=true [FAKE, REAL]
        "size_mb": round(size_mb, 1),
        "latency_ms": round(latency_ms, 1),
    }
    print(json.dumps({k: v for k, v in res.items() if k != "confusion"}), res["confusion"])
    return res


results = []

# --- TF-IDF + Logistic Regression
with open(DATA / "tfidf_lr.pkl", "rb") as f:
    vec, clf = pickle.load(f)
preds = clf.predict(vec.transform(texts))
t0 = time.perf_counter()
for t in texts[:200]:
    clf.predict(vec.transform([t]))
lat = (time.perf_counter() - t0) / 200 * 1000
results.append(summarize("TF-IDF + Logistic Regression", preds,
                         os.path.getsize(ARTIFACTS / "tfidf_lr.json") / 1e6, lat))


# --- DistilBERT ONNX (fp32 and int8), CPU provider only
def run_onnx(path, name):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    preds = []
    for i in range(0, len(texts), 32):
        enc = tok(texts[i:i + 32], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="np")
        logits = sess.run(["logits"], {"input_ids": enc["input_ids"].astype(np.int64),
                                       "attention_mask": enc["attention_mask"].astype(np.int64)})[0]
        preds.extend(logits.argmax(-1).tolist())
    t0 = time.perf_counter()
    for t in texts[:50]:
        enc = tok([t], truncation=True, max_length=MAX_LEN, return_tensors="np")
        sess.run(["logits"], {"input_ids": enc["input_ids"].astype(np.int64),
                              "attention_mask": enc["attention_mask"].astype(np.int64)})
    lat = (time.perf_counter() - t0) / 50 * 1000
    return summarize(name, np.array(preds), os.path.getsize(path) / 1e6, lat)


results.append(run_onnx(MODEL_DIR / "model_fp32.onnx", "DistilBERT (fp32 ONNX)"))
results.append(run_onnx(ARTIFACTS / "model_int8.onnx", "DistilBERT (int8 ONNX)"))

meta = {
    "dataset": "ISOT Fake and Real News (cleaned: Reuters datelines, URLs, handles, image credits removed; subject/date dropped)",
    "test_size": len(texts),
    "train_note": "DistilBERT fine-tuned 1 epoch on a 12k-article subset, max_len 128; TF-IDF on the full train split",
    "latency_note": "single article, CPU, 4 threads",
    "models": results,
}
(ARTIFACTS / "metrics.json").write_text(json.dumps(meta, indent=2))
print("saved", ARTIFACTS / "metrics.json")
