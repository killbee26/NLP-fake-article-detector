"""Shared paths and text cleaning. The Kotlin TextCleaner mirrors clean_text() exactly."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
MODEL_DIR = ROOT / "models" / "distilbert-fakenews"
DATA.mkdir(exist_ok=True)
ARTIFACTS.mkdir(exist_ok=True)

MAX_LEN = 128
LABELS = ["FAKE", "REAL"]  # index 0 = fake, 1 = real (dataset convention)

# ISOT leakage: real articles start with "WASHINGTON (Reuters) - ", fake ones often carry
# image credits / twitter handles. Strip these so models learn content, not source artefacts.
_DATELINE = re.compile(r"^.{0,120}?\((reuters|ap)\)\s*-\s*", re.IGNORECASE)
_URL = re.compile(r"(https?://\S+|www\.\S+|pic\.twitter\.com/\S+)", re.IGNORECASE)
_HANDLE = re.compile(r"@\w+")
_CREDIT = re.compile(r"(featured image|photo by|image via|getty images|21st century wire)[^.]*\.?", re.IGNORECASE)
_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    t = text or ""
    t = _DATELINE.sub("", t)
    t = re.sub(r"\breuters\b", "", t, flags=re.IGNORECASE)
    t = _URL.sub(" ", t)
    t = _HANDLE.sub(" ", t)
    t = _CREDIT.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    return t


_TOKEN = re.compile(r"[a-z0-9]+")


def tfidf_tokenize(text: str):
    """TF-IDF baseline tokenizer: kept trivial so Kotlin's TfidfLrClassifier reproduces it exactly."""
    return _TOKEN.findall(text.lower())


def article(title: str, text: str) -> str:
    return clean_text(f"{title}. {text}")
