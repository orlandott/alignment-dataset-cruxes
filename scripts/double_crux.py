"""Per-post double crux: does the top comment push back on the post, and if so
what is the falsifiable question they actually disagree about?

This works on a (post, top-comment) pair rather than two different articles:

  1. Summarize the comment's central claim (reusing the post-claim summarizer).
  2. Decide whether the comment *disagrees* with the post (cue-based + a TF-IDF
     contrast fallback).
  3. If it disagrees, extract a double crux via the existing TF-IDF contrast
     heuristic, framed as "post position vs comment position".

No LLM calls; everything here is free and deterministic.
"""

from __future__ import annotations

import re

from scripts.heuristic_crux import (
    classify_crux_type,
    contrast_terms,
    extract_evidence_quote,
)
from scripts.post_claims import summarize_claims

MARKUP = re.compile(r"<[^>]+>")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Phrases that signal a commenter is pushing back rather than agreeing/expanding.
DISAGREEMENT_CUES = (
    "i disagree",
    "i don't think",
    "i do not think",
    "i don't buy",
    "i'm not convinced",
    "not convinced",
    "i'm skeptical",
    "skeptical",
    "i doubt",
    "this is wrong",
    "seems wrong",
    "is false",
    "isn't true",
    "not true",
    "incorrect",
    "mistaken",
    "the problem with",
    "the issue with",
    "but this",
    "however",
    "on the contrary",
    "i object",
    "i'd push back",
    "push back",
    "fails to",
    "doesn't follow",
    "does not follow",
    "i think this is",
    "disagree",
    "counterexample",
    "that's not",
    "i don't agree",
)

# Strong agreement phrasing that should veto a weak single-cue match.
AGREEMENT_CUES = (
    "i agree",
    "great post",
    "strongly agree",
    "this is great",
    "well said",
    "i think this is right",
    "exactly",
    "+1",
    "thanks for",
)


def _strip_markup(text: str) -> str:
    text = MARKUP.sub(" ", text or "")
    return re.sub(r"\s+", " ", text).strip()


def _first_sentence(text: str, limit: int = 240) -> str:
    text = _strip_markup(text)
    parts = [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]
    sentence = parts[0] if parts else text
    if len(sentence) > limit:
        sentence = sentence[: limit - 1].rstrip() + "\u2026"
    return sentence


def summarize_comment_claim(comment_text: str) -> str:
    """One short statement of what the top comment is claiming."""
    claims = summarize_claims(comment_text, max_claims=1)
    if claims:
        return claims[0]
    return _first_sentence(comment_text)


def format_comment_claim(claim: str, *, disagrees: bool) -> str:
    """Frame a comment's claim for display.

    Disagreeing comments are the "pushback", so we present them as a
    counterclaim; agreeing/expanding comments are left as a plain point.
    """
    claim = (claim or "").strip()
    if not claim:
        return claim
    prefix = "The counterclaim is: " if disagrees else "The comment's point: "
    # Avoid double-prefixing if a claim already starts with the framing.
    if claim.lower().startswith(("the counterclaim is", "the comment's point")):
        return claim
    return f"{prefix}{claim}"


def detect_disagreement(comment_text: str) -> bool:
    """Heuristically decide whether a comment pushes back on the post."""
    lower = _strip_markup(comment_text).lower()
    disagree_hits = sum(1 for cue in DISAGREEMENT_CUES if cue in lower)
    agree_hits = sum(1 for cue in AGREEMENT_CUES if cue in lower)
    if disagree_hits == 0:
        return False
    # A lone soft cue ("however") inside an otherwise agreeing comment is weak.
    if disagree_hits == 1 and agree_hits >= 1:
        return False
    return True


def extract_double_crux(post_text: str, comment_text: str) -> dict:
    """Return a double crux dict for a post and its (disagreeing) top comment.

    Shape mirrors the LLM contract used elsewhere but with post/comment-specific
    evidence keys.
    """
    terms_post, terms_comment, shared, contrast_score = contrast_terms(post_text, comment_text)

    if not terms_post or not terms_comment:
        return {
            "has_crux": False,
            "no_crux_reason": "not enough contrast between post and comment",
            "crux_question": None,
            "type": None,
            "evidence_post": None,
            "evidence_comment": None,
        }

    term_post = terms_post[0]
    term_comment = terms_comment[0]
    shared_topic = shared[0] if shared else term_post
    crux_type = classify_crux_type(terms_post + terms_comment + shared, (post_text, comment_text))

    if shared_topic and shared_topic not in {term_post, term_comment}:
        question = (
            f"On {shared_topic}, does the disagreement hinge on {term_post} "
            f"(post) versus {term_comment} (comment)?"
        )
    else:
        question = (
            f"Is the key disagreement really about {term_post} (post) "
            f"versus {term_comment} (comment)?"
        )

    return {
        "has_crux": True,
        "no_crux_reason": None,
        "crux_question": question,
        "type": crux_type,
        "evidence_post": extract_evidence_quote(post_text, term_post),
        "evidence_comment": extract_evidence_quote(comment_text, term_comment),
    }


def analyze_top_comment(post_text: str, comment_text: str) -> dict:
    """Bundle the comment claim, disagreement flag, and (optional) double crux."""
    disagrees = detect_disagreement(comment_text)
    crux = extract_double_crux(post_text, comment_text) if disagrees else None
    return {
        "claim": format_comment_claim(summarize_comment_claim(comment_text), disagrees=disagrees),
        "disagrees": disagrees,
        "crux": crux,
    }
