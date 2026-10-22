"""Export the fine-tuned model to ONNX (fp32) and dynamically quantize it to int8 for CPU inference."""
import shutil

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import ARTIFACTS, MAX_LEN, MODEL_DIR

model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, attn_implementation="eager").eval()
tok = AutoTokenizer.from_pretrained(MODEL_DIR)

dummy = tok(["a dummy sentence for export"], return_tensors="pt", padding="max_length",
            truncation=True, max_length=MAX_LEN)
fp32 = MODEL_DIR / "model_fp32.onnx"
with torch.no_grad():
    torch.onnx.export(
        model, (dummy["input_ids"], dummy["attention_mask"]), str(fp32),
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch"}},
        opset_version=17, do_constant_folding=True, dynamo=False,
    )

int8 = ARTIFACTS / "model_int8.onnx"
quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
if (MODEL_DIR / "vocab.txt").exists():
    shutil.copy(MODEL_DIR / "vocab.txt", ARTIFACTS / "vocab.txt")
else:  # newer transformers only save tokenizer.json; rebuild vocab.txt (one token per line, line number = id)
    vocab = tok.get_vocab()
    (ARTIFACTS / "vocab.txt").write_text("\n".join(t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])) + "\n",
                                         encoding="utf-8")
print(f"fp32 {fp32.stat().st_size / 1e6:.1f} MB -> int8 {int8.stat().st_size / 1e6:.1f} MB")
