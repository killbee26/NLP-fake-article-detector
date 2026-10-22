# Fake News Detector (DistilBERT + Kotlin Compose Desktop)

Train on a GPU in Python, then run fully offline on **any laptop's CPU** in a Kotlin / Compose Multiplatform desktop app.

```
training/ (Python, GPU)                        app/ (Kotlin, CPU only)
01 clean ISOT, split 80/10/10                  core/        TextCleaner, WordPieceTokenizer (pure Kotlin),
02 TF-IDF(1-2 gram) + LogReg  -> tfidf_lr.json              TfidfLrClassifier, DistilBertClassifier (ONNX Runtime)
03 fine-tune DistilBERT (transfer learning)    composeApp/  Analyze tab (verdict, model comparison, word influence,
04 export ONNX + int8 quantize -> model_int8.onnx           pipeline stepper) + Model report tab (metrics, confusion)
05 evaluate -> metrics.json
06 parity fixtures + demo samples
```

## Run the app (any machine with JDK 17+; no Python, no GPU)
```bash
cd app
gradlew.bat :composeApp:run
```
`app/gradle.properties` points `org.gradle.java.home` at Android Studio's bundled JDK 21. On another machine, change or remove that line.
Build an installer with a bundled JRE: `gradlew.bat :composeApp:packageDistributionForCurrentOS`.

## Re-train (needs Python 3.10+, ideally an NVIDIA GPU)
```bash
python -m venv .venv
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\pip install -r training\requirements.txt
cd training
..\.venv\Scripts\python 01_prepare_data.py   (then 02 ... 06 in order)
```

## Verify Kotlin == Python
`gradlew.bat :core:test` checks that the cleaner, the tokenizer (exact token ids), TF-IDF probabilities and DistilBERT logits match the Python outputs.

## Syllabus mapping
| Unit | Where |
|---|---|
| I: NLP pipeline, tokenization, preprocessing | `TextCleaner`, `WordPieceTokenizer`, pipeline stepper in the UI |
| II: BoW / TF-IDF / n-grams | TF-IDF (1-2 gram) + Logistic Regression baseline, word-influence view |
| IV: Transformers, transfer learning | Fine-tuned `distilbert-base-uncased` |
| V: Evaluation, deployment | Accuracy/P/R/F1/confusion matrix, int8 quantization, offline CPU desktop app |

## Dataset note (leakage)
ISOT real articles almost all start with `CITY (Reuters) -`. A model can score ~100% just by spotting "Reuters".
We strip datelines, the word "Reuters", URLs, @handles and image credits, and drop the `subject`/`date` columns, so the models must learn from content.

## Future work
Batch CSV scoring, sentence-level explanations for DistilBERT, cross-dataset evaluation (e.g. LIAR), Android target.
