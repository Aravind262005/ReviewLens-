"""
ReviewLens: Product Insight Analyzer
Flask backend — handles routing, prediction, and result assembly.
"""

from flask import Flask, render_template, request, jsonify
from predict import predict_review, get_confidence, analyze_review
from extractor import extract_pros_cons

app = Flask(__name__)


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    """Serve the main landing page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Receive review text via AJAX, run sentiment prediction,
    extract pros/cons, compute score & verdict, and return JSON.

    Uses sentence-level analysis for accurate mixed/neutral handling.
    """
    data = request.get_json()
    review_text = data.get("review", "").strip()

    # Guard: empty input
    if not review_text:
        return jsonify({"error": "Please enter a review to analyze."}), 400

    # 1. Sentence-level sentiment analysis (NEW)
    analysis = analyze_review(review_text)
    sentiment = analysis["final_sentiment"]
    positive_pct = round(analysis["positive_ratio"] * 100, 1)
    negative_pct = round(analysis["negative_ratio"] * 100, 1)
    confidence = analysis["confidence"]

    # 2. Pros & Cons extraction (unchanged)
    pros, cons = extract_pros_cons(review_text)

    # 3. Score (1–5) derived from positive ratio
    score = _calculate_score(sentiment, analysis["positive_ratio"])

    # 4. Verdict
    verdict = _determine_verdict(score)

    return jsonify({
        "sentiment": sentiment,
        "positive_pct": positive_pct,
        "negative_pct": negative_pct,
        "confidence": confidence,
        "score": score,
        "verdict": verdict,
        "pros": pros,
        "cons": cons,
    })


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _calculate_score(sentiment: str, positive_ratio: float) -> int:
    """
    Map sentiment + positive ratio to a 1–5 star score.
    Uses sentence-level positive ratio for a more graded result.
    """
    if sentiment == "Positive":
        if positive_ratio >= 0.90:
            return 5
        elif positive_ratio >= 0.75:
            return 4
        else:
            return 3
    elif sentiment == "Mixed":
        # Mixed sentiment always maps to 3 (moderate)
        return 3
    else:  # Negative
        if positive_ratio <= 0.10:
            return 1
        elif positive_ratio <= 0.25:
            return 2
        else:
            return 3


def _determine_verdict(score: int) -> str:
    """Convert a 1–5 score into a human‑readable verdict."""
    if score >= 4:
        return "Recommended"
    elif score == 3:
        return "Moderate"
    else:
        return "Not Recommended"


# ─── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True)