"""
extractor.py — Extract Pros and Cons from review text using TF-IDF keyword analysis.

Positive-sentiment keywords → Pros
Negative-sentiment keywords → Cons
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import preprocess
import re

# ── Curated word banks for sentiment classification ──────────────────────────

_POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "love", "best", "perfect",
    "awesome", "fantastic", "wonderful", "nice", "happy", "recommend",
    "quality", "comfortable", "durable", "fast", "easy", "beautiful",
    "reliable", "solid", "smooth", "impressive", "useful", "brilliant",
    "superior", "outstanding", "superb", "pleased", "satisfied", "worth",
    "premium", "elegant", "lightweight", "sturdy", "crisp", "clear",
    "responsive", "efficient", "powerful", "value", "affordable",
}

_NEGATIVE_WORDS = {
    "bad", "worst", "terrible", "horrible", "poor", "hate", "slow",
    "broken", "cheap", "waste", "disappointed", "ugly", "useless",
    "heavy", "loud", "expensive", "defective", "fragile", "flimsy",
    "uncomfortable", "unreliable", "complicated", "annoying", "weak",
    "faulty", "awful", "overpriced", "mediocre", "boring", "difficult",
    "problem", "issue", "fail", "lacks", "missing", "damage", "crack",
    "leak", "noise", "delay", "refund", "return",
}


def extract_pros_cons(text: str, max_items: int = 5):
    """
    Analyse *text* and return (pros: list[str], cons: list[str]).

    Strategy:
      1. Split the review into sentences.
      2. Run TF-IDF over the sentences.
      3. For each top keyword, classify it as a pro or con using the
         curated word banks above.
      4. Build human-readable bullet points from the matching sentences.
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 3]

    if not sentences:
        return ["No specific pros found"], ["No specific cons found"]

    # Preprocess each sentence for TF-IDF
    cleaned_sentences = [preprocess(s) for s in sentences]
    cleaned_sentences = [s for s in cleaned_sentences if s.strip()]

    if not cleaned_sentences:
        return ["No specific pros found"], ["No specific cons found"]

    # Extract TF-IDF features from the sentences
    try:
        tfidf = TfidfVectorizer(max_features=50, stop_words="english")
        tfidf_matrix = tfidf.fit_transform(cleaned_sentences)
        feature_names = tfidf.get_feature_names_out()
    except ValueError:
        # Edge case: vocabulary is empty after stop-word removal
        return ["No specific pros found"], ["No specific cons found"]

    # Aggregate TF-IDF scores across all sentences
    scores = tfidf_matrix.sum(axis=0).A1  # dense 1-D array
    keyword_scores = sorted(
        zip(feature_names, scores), key=lambda x: x[1], reverse=True
    )

    pros: list[str] = []
    cons: list[str] = []
    seen: set[str] = set()  # prevent duplicate bullets

    for word, _ in keyword_scores:
        if len(pros) >= max_items and len(cons) >= max_items:
            break

        # Find the original sentence that best represents this keyword
        best_sentence = _find_best_sentence(word, sentences)
        bullet = _humanise(best_sentence or word)

        # Skip if we already used this exact bullet text
        if bullet in seen:
            continue

        if word in _POSITIVE_WORDS and len(pros) < max_items:
            pros.append(bullet)
            seen.add(bullet)
        elif word in _NEGATIVE_WORDS and len(cons) < max_items:
            cons.append(bullet)
            seen.add(bullet)

    # Fallback — if nothing was classified, provide generic feedback
    if not pros:
        pros.append("No specific pros found")
    if not cons:
        cons.append("No specific cons found")

    return pros, cons



# ─── Helpers ─────────────────────────────────────────────────────────────────

def _find_best_sentence(keyword: str, sentences: list[str]) -> str | None:
    """Return the shortest sentence that contains *keyword* (case-insensitive)."""
    matches = [s for s in sentences if keyword.lower() in s.lower()]
    return min(matches, key=len) if matches else None


def _humanise(sentence: str) -> str:
    """Clean up a sentence for display as a bullet point."""
    sentence = sentence.strip().rstrip(".")
    # Capitalise first letter
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]
    # Truncate overly long sentences
    if len(sentence) > 120:
        sentence = sentence[:117] + "…"
    return sentence
