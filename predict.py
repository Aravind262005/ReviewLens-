"""
predict.py — Sentiment prediction using the trained model & vectorizer.

Supports both whole-text prediction (original) and sentence-level
analysis (added for better mixed/neutral review handling).
"""

import pickle
import os
from preprocess import preprocess
from nltk.tokenize import sent_tokenize

# Resolve paths relative to this file so Flask can find them regardless of cwd
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_BASE_DIR, "models", "model.pkl")
_VECTORIZER_PATH = os.path.join(_BASE_DIR, "models", "vectorizer.pkl")

# Load model + vectorizer once at import time
with open(_MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(_VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)


# ─── Original functions (KEPT UNCHANGED) ────────────────────────────────────

def predict_review(text: str) -> str:
    """Return 'Positive' or 'Negative' for the given review text."""
    clean = preprocess(text)
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]
    return "Positive" if pred == 1 else "Negative"


def get_confidence(text: str) -> float:
    """
    Return the model's confidence (probability) for its chosen class.
    Falls back to 0.75 if the model doesn't support predict_proba.
    """
    clean = preprocess(text)
    vec = vectorizer.transform([clean])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
        return float(max(proba))
    elif hasattr(model, "decision_function"):
        # For SVM and similar — map absolute decision value to [0.5, 1.0]
        decision = abs(float(model.decision_function(vec)[0]))
        import math
        return 1.0 / (1.0 + math.exp(-decision))  # sigmoid
    else:
        return 0.75  # safe default


# ─── NEW: Sentence-level analysis ───────────────────────────────────────────

def _predict_sentence(sentence: str) -> int:
    """Predict a single sentence. Returns 1 (Positive) or 0 (Negative)."""
    clean = preprocess(sentence)
    if not clean.strip():
        return 1  # skip empty sentences, treat as neutral-positive
    vec = vectorizer.transform([clean])
    return int(model.predict(vec)[0])


def analyze_review(text: str) -> dict:
    """
    Perform sentence-level sentiment analysis on the full review text.

    Steps:
      1. Split text into individual sentences using NLTK sent_tokenize()
      2. Predict sentiment for EACH sentence independently
      3. Calculate positive_ratio and negative_ratio
      4. Determine final sentiment from the ratio thresholds

    Returns a dict with:
      - final_sentiment: "Positive" / "Negative" / "Mixed"
      - positive_ratio : float (0.0 – 1.0)
      - negative_ratio : float (0.0 – 1.0)
      - confidence     : float (0.0 – 100.0), derived from the ratio
      - sentence_results: list of per-sentence predictions (for debugging)
    """
    # 1. Split into sentences
    sentences = sent_tokenize(text)

    # Guard: single-word or very short input — fall back to whole-text prediction
    if len(sentences) == 0:
        fallback = predict_review(text)
        ratio = 1.0 if fallback == "Positive" else 0.0
        return {
            "final_sentiment": fallback,
            "positive_ratio": ratio,
            "negative_ratio": 1.0 - ratio,
            "confidence": ratio * 100,
            "sentence_results": [],
        }

    # 2. Predict each sentence individually
    sentence_results = []
    for sent in sentences:
        pred = _predict_sentence(sent)
        sentence_results.append({
            "sentence": sent,
            "prediction": "Positive" if pred == 1 else "Negative",
        })

    # 3. Calculate ratios
    total = len(sentence_results)
    positive_count = sum(1 for r in sentence_results if r["prediction"] == "Positive")
    negative_count = total - positive_count

    positive_ratio = positive_count / total
    negative_ratio = negative_count / total

    # 4. Determine final sentiment from ratio thresholds
    if positive_ratio > 0.7:
        final_sentiment = "Positive"
    elif positive_ratio < 0.3:
        final_sentiment = "Negative"
    else:
        final_sentiment = "Mixed"

    # 5. Confidence = how strongly the review leans one way
    #    For Positive/Negative → the dominant ratio; for Mixed → capped lower
    confidence = positive_ratio * 100

    return {
        "final_sentiment": final_sentiment,
        "positive_ratio": round(positive_ratio, 3),
        "negative_ratio": round(negative_ratio, 3),
        "confidence": round(confidence, 1),
        "sentence_results": sentence_results,
    }