"""Reference outputs for the Kotlin parity tests, plus demo samples for the UI."""
import json
import pickle

import numpy as np
import onnxruntime as ort
import pandas as pd
from transformers import AutoTokenizer

from common import ARTIFACTS, DATA, MAX_LEN, MODEL_DIR, clean_text

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
sess = ort.InferenceSession(str(ARTIFACTS / "model_int8.onnx"), providers=["CPUExecutionProvider"])
sess_fp32 = ort.InferenceSession(str(MODEL_DIR / "model_fp32.onnx"), providers=["CPUExecutionProvider"])
with open(DATA / "tfidf_lr.pkl", "rb") as f:
    vec, clf = pickle.load(f)

test = pd.read_parquet(DATA / "test.parquet")
tricky = [
    "Café owners in São Paulo say the économie is booming!!! #news @someone https://t.co/xyz",
    "BREAKING: Obama's   secret plan—revealed (WATCH) ... you won't believe it?!",
    "WASHINGTON (Reuters) - The U.S. Senate on Tuesday passed a $1.5 trillion tax bill, 51-49.",
    "Supercalifragilisticexpialidocious antidisestablishmentarianism 12,345.67 e-mail co-operate",
    "\u65e5\u672c Tokyo \u6771\u4eac mixed scripts and emoji \U0001F600 test",
    "",
]
texts = tricky + test.sample(n=14, random_state=7)["content"].tolist()


def logits_of(text, session=sess):
    enc = tok([text], truncation=True, max_length=MAX_LEN, return_tensors="np")
    out = session.run(["logits"], {"input_ids": enc["input_ids"].astype(np.int64),
                                "attention_mask": enc["attention_mask"].astype(np.int64)})[0][0]
    return enc["input_ids"][0].tolist(), out.tolist()


fixtures = []
for raw in texts:
    cleaned = clean_text(raw)
    ids, logits = logits_of(cleaned)
    fixtures.append({
        "raw": raw,
        "cleaned": cleaned,
        "input_ids": ids,
        "logits": logits,
        "logits_fp32": logits_of(cleaned, sess_fp32)[1],
        "tfidf_prob_real": float(clf.predict_proba(vec.transform([cleaned]))[0, 1]),
    })
(ARTIFACTS / "fixtures.json").write_text(json.dumps(fixtures, ensure_ascii=False, indent=1), encoding="utf-8")

# Agreement set: the Kotlin app's ONNX Runtime (1.17) may round int8 kernels differently from Python's,
# so we check that it makes the same predictions on 300 held-out articles.
agree = [{"text": t, "label": int(l), "pred": int(np.argmax(logits_of(t)[1]))}
         for t, l in test.sample(n=300, random_state=11)[["content", "label"]].itertuples(index=False)]
(DATA / "agreement.json").write_text(json.dumps(agree, ensure_ascii=False), encoding="utf-8")

# Demo samples: short test articles both models classify correctly with high confidence.
samples = {"real": [], "fake": []}
for _, row in test.sample(frac=1, random_state=3).iterrows():
    words = row["content"].split()
    if not 60 <= len(words) <= 220:
        continue
    _, lg = logits_of(row["content"])
    p = np.exp(lg) / np.exp(lg).sum()
    tf = clf.predict_proba(vec.transform([row["content"]]))[0]
    key = "real" if row["label"] == 1 else "fake"
    if p[row["label"]] > 0.9 and tf[row["label"]] > 0.7 and len(samples[key]) < 5:
        samples[key].append(row["content"])
    if all(len(v) == 5 for v in samples.values()):
        break
(ARTIFACTS / "samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"saved {len(fixtures)} fixtures, samples real={len(samples['real'])} fake={len(samples['fake'])}")
