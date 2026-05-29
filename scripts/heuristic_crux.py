"""Heuristic crux extraction without LLM calls."""

from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

CONTRAST_THRESHOLD = 0.04
TOP_TERMS = 5

EMPIRICAL_TERMS = frozenset(
    {"evidence", "experiment", "data", "measure", "study", "empirical", "observable", "test", "observation"}
)
PREDICTION_TERMS = frozenset(
    {"will", "forecast", "timeline", "expect", "likely", "probability", "predict", "future", "years"}
)
VALUES_TERMS = frozenset(
    {"should", "ought", "value", "moral", "prefer", "utility", "ethics", "normative", "desirable"}
)

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
MARKUP = re.compile(r"<[^>]+>")


def _strip_markup(text: str) -> str:
    text = MARKUP.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_usable_term(term: str) -> bool:
    if len(term) < 3 or "{" in term or "}" in term:
        return False
    return not term.startswith("mjx")


def _build_pair_tfidf(text_a: str, text_b: str) -> tuple[TfidfVectorizer, np.ndarray]:
    vectorizer = TfidfVectorizer(
        max_features=10_000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )
    matrix = vectorizer.fit_transform([text_a, text_b]).toarray()
    return vectorizer, matrix


def contrast_terms(
    text_a: str,
    text_b: str,
    *,
    top_n: int = TOP_TERMS,
) -> tuple[list[str], list[str], list[str], float]:
    """Return enriched terms for A/B, shared terms, and contrast score."""
    vectorizer, matrix = _build_pair_tfidf(text_a, text_b)
    features = vectorizer.get_feature_names_out()
    vec_a = matrix[0]
    vec_b = matrix[1]
    diff = vec_a - vec_b

    order_a = np.argsort(-diff)
    order_b = np.argsort(diff)

    terms_a: list[str] = []
    for idx in order_a:
        if diff[idx] <= 0:
            break
        term = features[idx]
        if _is_usable_term(term):
            terms_a.append(term)
        if len(terms_a) >= top_n:
            break

    terms_b: list[str] = []
    for idx in order_b:
        if diff[idx] >= 0:
            break
        term = features[idx]
        if _is_usable_term(term):
            terms_b.append(term)
        if len(terms_b) >= top_n:
            break

    shared_scores = np.minimum(vec_a, vec_b)
    shared_order = np.argsort(-shared_scores)
    shared: list[str] = []
    for idx in shared_order:
        if shared_scores[idx] <= 0:
            break
        term = features[idx]
        if _is_usable_term(term):
            shared.append(term)
        if len(shared) >= top_n:
            break

    top_diffs = sorted((float(diff[i]) for i in range(len(diff)) if diff[i] > 0), reverse=True)
    contrast_score = sum(top_diffs[:top_n])

    return terms_a, terms_b, shared, contrast_score


def classify_crux_type(terms: list[str], texts: tuple[str, ...]) -> str:
    """Classify crux type from contrast/shared terms and passage text."""
    haystack = " ".join(terms).lower() + " " + " ".join(texts).lower()
    words = set(re.findall(r"[a-z]+", haystack))

    scores = {
        "empirical": len(words & EMPIRICAL_TERMS),
        "prediction": len(words & PREDICTION_TERMS),
        "values": len(words & VALUES_TERMS),
    }
    best_type, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score == 0:
        return "empirical"
    return best_type


def format_crux_question(shared_topic: str, term_a: str, term_b: str) -> str:
    if shared_topic and shared_topic not in {term_a, term_b}:
        return (
            f"To what extent does {shared_topic} require {term_a} "
            f"versus {term_b}?"
        )
    return f"Does the central disagreement turn primarily on {term_a} or {term_b}?"


def _sentence_term_score(sentence: str, term: str) -> int:
    sentence_lower = sentence.lower()
    term_lower = term.lower()
    if term_lower in sentence_lower:
        return term_lower.count(" ") + 2
    parts = term_lower.split()
    return sum(1 for part in parts if part in sentence_lower)


def extract_evidence_quote(text: str, term: str, *, max_sentences: int = 2) -> str | None:
    """Pick sentences with the strongest overlap with the contrast term."""
    text = _strip_markup(text)
    sentences = [s.strip() for s in SENTENCE_SPLIT.split(text.strip()) if s.strip()]
    if not sentences:
        return None

    ranked = sorted(
        sentences,
        key=lambda sentence: (_sentence_term_score(sentence, term), len(sentence)),
        reverse=True,
    )
    if _sentence_term_score(ranked[0], term) == 0:
        return ranked[0][:280] if ranked else None

    selected: list[str] = []
    for sentence in ranked:
        if len(sentence) > 400:
            continue
        if _sentence_term_score(sentence, term) == 0 and selected:
            break
        selected.append(sentence)
        if len(selected) >= max_sentences:
            break
    return " ".join(selected)


def extract_heuristic_crux(text_a: str, text_b: str) -> dict:
    """Extract a crux between two passages using TF-IDF contrast heuristics."""
    terms_a, terms_b, shared, contrast_score = contrast_terms(text_a, text_b)

    if contrast_score < CONTRAST_THRESHOLD or not terms_a or not terms_b:
        return {
            "has_crux": False,
            "no_crux_reason": "insufficient term contrast between passages",
            "crux_question": None,
            "type": None,
            "evidence_a": None,
            "evidence_b": None,
        }

    term_a = terms_a[0]
    term_b = terms_b[0]
    shared_topic = shared[0] if shared else term_a
    question = format_crux_question(shared_topic, term_a, term_b)
    all_terms = terms_a + terms_b + shared
    crux_type = classify_crux_type(all_terms, (text_a, text_b))

    return {
        "has_crux": True,
        "no_crux_reason": None,
        "crux_question": question,
        "type": crux_type,
        "evidence_a": extract_evidence_quote(text_a, term_a),
        "evidence_b": extract_evidence_quote(text_b, term_b),
    }
