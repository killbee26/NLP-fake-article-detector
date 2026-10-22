"""Download ISOT (HF mirror), clean leakage, dedupe, split 80/10/10."""
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from common import DATA, article

ds = load_dataset("ErfanMoosaviMonazzah/fake-news-detection-dataset-English")
df = pd.concat([ds[s].to_pandas() for s in ds.keys()], ignore_index=True)
print("raw rows:", len(df))

df["content"] = [article(t, x) for t, x in zip(df["title"], df["text"])]
df = df[df["content"].str.split().str.len() >= 20]         
df = df.drop_duplicates(subset="content")
df = df[["content", "label"]].reset_index(drop=True)         
print("clean rows:", len(df), "label counts:", df["label"].value_counts().to_dict())

train, rest = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
val, test = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=42)
for name, part in [("train", train), ("val", val), ("test", test)]:
    part.to_parquet(DATA / f"{name}.parquet", index=False)
    print(name, len(part))
