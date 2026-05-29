"""Heuristic per-post claim summarization (no LLM calls).

Given the text of a single post, pick the sentences that read most like the
core claim(s) the author is making. We score sentences by:

  1. Assertion cues  — modal/argumentative phrases ("we should", "I argue", ...)
  2. Topic salience  — overlap with the document's most frequent content words
  3. Position        — a small bonus for sentences near the top (theses lead)

Most posts yield a single dominant claim; a second is returned only when it is
nearly as strong, so callers naturally get "sometimes more than one".
"""

from __future__ import annotations

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

MARKUP = re.compile(r"<[^>]+>")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
WORD = re.compile(r"[a-zA-Z][a-zA-Z'-]+")

MATH_SYMBOLS = set("=∑∏∈∉⊂⊆⊃⊇⋅×÷→⇒⇔≈≤≥≠√∫∮∀∃∂∇±∓°∝≡⟨⟩∧∨¬")
ABBREV_TAIL = re.compile(r"\b(?:i\.e|e\.g|cf|vs|etc|fig|eq)\.?$", re.IGNORECASE)
MIN_CONTENT_WORDS = 5

MAX_CLAIM_CHARS = 240
MIN_SENTENCE_CHARS = 40
MAX_SENTENCE_CHARS = 320
# A second claim is only kept when it scores nearly as high as the first, so
# most posts surface a single dominant claim and only multi-thesis posts add one.
SECONDARY_CLAIM_RATIO = 0.85

ASSERTION_CUES = (
    "i argue",
    "i claim",
    "i think",
    "i believe",
    "i expect",
    "i suspect",
    "my claim",
    "my view",
    "we should",
    "we need",
    "we must",
    "should be",
    "must be",
    "the key",
    "the point",
    "the problem",
    "the core",
    "the main",
    "the central",
    "the real",
    "in fact",
    "claim that",
    "argue that",
    "believe that",
    "the reason",
    "this means",
    "which means",
    "implies that",
    "turns out",
    "the takeaway",
    "i conclude",
    "therefore",
    "fundamentally",
)


def _strip_markup(text: str) -> str:
    text = MARKUP.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _content_words(sentence: str) -> list[str]:
    return [
        word
        for word in (match.group(0).lower() for match in WORD.finditer(sentence))
        if len(word) >= 4 and word not in ENGLISH_STOP_WORDS
    ]


def _looks_like_sentence(sentence: str) -> bool:
    if not (MIN_SENTENCE_CHARS <= len(sentence) <= MAX_SENTENCE_CHARS):
        return False
    if "http" in sentence or "www." in sentence:
        return False
    # Drop equations / formula-laden lines, which read as noise out of context.
    if any(ch in MATH_SYMBOLS for ch in sentence):
        return False
    # Drop fragments truncated at an abbreviation ("... observable O, i.e.").
    if ABBREV_TAIL.search(sentence.rstrip(".!?").strip()) or ABBREV_TAIL.search(sentence.strip()):
        return False
    if len(_content_words(sentence)) < MIN_CONTENT_WORDS:
        return False
    letters = sum(ch.isalpha() for ch in sentence)
    return letters >= 0.6 * len(sentence)


def _truncate(text: str, limit: int = MAX_CLAIM_CHARS) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "\u2026"


def _cue_score(lower: str) -> int:
    return sum(1 for cue in ASSERTION_CUES if cue in lower)


def summarize_claims(text: str, *, max_claims: int = 2) -> list[str]:
    """Return up to ``max_claims`` short claim summaries for a post.

    Returns ``[]`` when no sentence reads clearly like a claim (e.g. very short
    or markup-only posts).
    """
    clean = _strip_markup(text)
    sentences = [s.strip() for s in SENTENCE_SPLIT.split(clean) if s.strip()]
    candidates = [s for s in sentences if _looks_like_sentence(s)]
    if not candidates:
        return []

    doc_freq: dict[str, int] = {}
    for sentence in candidates:
        for word in set(_content_words(sentence)):
            doc_freq[word] = doc_freq.get(word, 0) + 1

    total = len(candidates)
    scored: list[tuple[float, int, str]] = []
    for index, sentence in enumerate(candidates):
        words = _content_words(sentence)
        if not words:
            continue
        unique = set(words)
        # Average topic salience keeps the term comparable across posts of very
        # different length, so the assertion-cue bonus stays meaningful.
        salience = sum(doc_freq.get(word, 0) for word in unique) / len(unique)
        cue = _cue_score(sentence.lower())
        position = max(0.0, 1.0 - index / total)
        score = salience + 1.6 * cue + 0.8 * position
        scored.append((score, index, sentence))

    if not scored:
        return []

    scored.sort(key=lambda item: (-item[0], item[1]))
    top_score = scored[0][0]

    claims: list[str] = []
    seen_words: list[set[str]] = []
    for score, _, sentence in scored:
        if claims and score < SECONDARY_CLAIM_RATIO * top_score:
            break
        words = set(_content_words(sentence))
        if any(len(words & prev) >= 0.6 * len(words) for prev in seen_words if words):
            continue
        claims.append(_truncate(sentence))
        seen_words.append(words)
        if len(claims) >= max_claims:
            break

    return claims
