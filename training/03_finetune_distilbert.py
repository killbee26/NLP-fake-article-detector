"""Fine-tune distilbert-base-uncased (transfer learning) on a 12k-article subset. ~4 min on an RTX 4060."""
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments)

from common import DATA, LABELS, MAX_LEN, MODEL_DIR

BASE = "distilbert-base-uncased"
TRAIN_SUBSET = 12000

print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
tok = AutoTokenizer.from_pretrained(BASE)

train = pd.read_parquet(DATA / "train.parquet").sample(n=TRAIN_SUBSET, random_state=42)
val = pd.read_parquet(DATA / "val.parquet").sample(n=2000, random_state=42)


def encode(df):
    ds = Dataset.from_pandas(df.rename(columns={"label": "labels"}), preserve_index=False)
    return ds.map(lambda b: tok(b["content"], truncation=True, max_length=MAX_LEN),
                  batched=True, remove_columns=["content"])


train_ds, val_ds = encode(train), encode(val)

model = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=2, id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)})


def metrics(p):
    preds = np.argmax(p.predictions, axis=-1)
    return {"accuracy": accuracy_score(p.label_ids, preds), "f1": f1_score(p.label_ids, preds)}


args = TrainingArguments(
    output_dir=str(MODEL_DIR / "checkpoints"),
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    warmup_steps=TRAIN_SUBSET // 32 // 10,  # ~10% of the steps

    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=50,
    report_to=[],
    dataloader_num_workers=0,
)
trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                  data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
trainer.train()
print(trainer.evaluate())
trainer.save_model(str(MODEL_DIR))
tok.save_pretrained(str(MODEL_DIR))
print("saved", MODEL_DIR)
