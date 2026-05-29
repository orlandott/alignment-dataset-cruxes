from scripts.heuristic_crux import (
    classify_crux_type,
    contrast_terms,
    extract_evidence_quote,
    extract_heuristic_crux,
    format_crux_question,
)


def test_contrast_terms_finds_divergent_vocabulary():
    text_a = (
        "We should prioritize corrigibility and value learning in aligned systems. "
        "Moral preferences must be respected when designing agents."
    )
    text_b = (
        "Scaling laws and compute forecasts predict AGI timelines within decades. "
        "Experimental evidence from benchmark data supports rapid capability growth."
    )
    terms_a, terms_b, shared, score = contrast_terms(text_a, text_b)
    assert terms_a
    assert terms_b
    assert score > 0


def test_contrast_terms_low_for_identical_passages():
    text = "Alignment research should focus on corrigibility and interpretability methods."
    _, _, _, score = contrast_terms(text, text)
    assert score == 0


def test_classify_crux_type_empirical():
    assert classify_crux_type(["evidence", "experiment"], ("study data", "")) == "empirical"


def test_classify_crux_type_prediction():
    assert classify_crux_type(["forecast", "timeline"], ("will likely happen", "")) == "prediction"


def test_classify_crux_type_values():
    assert classify_crux_type(["should", "moral"], ("we ought to prefer", "")) == "values"


def test_classify_crux_type_defaults_to_empirical():
    assert classify_crux_type(["alignment", "research"], ("", "")) == "empirical"


def test_format_crux_question_with_shared_topic():
    question = format_crux_question("ai safety", "corrigibility", "scaling laws")
    assert "ai safety" in question
    assert "corrigibility" in question
    assert "scaling laws" in question


def test_format_crux_question_without_shared_topic():
    question = format_crux_question("corrigibility", "corrigibility", "scaling laws")
    assert "corrigibility" in question
    assert "scaling laws" in question


def test_extract_evidence_quote_prefers_matching_sentence():
    text = (
        "General intro without keywords. "
        "Corrigibility is essential for safe AI systems. "
        "Another unrelated sentence."
    )
    quote = extract_evidence_quote(text, "corrigibility")
    assert quote is not None
    assert "Corrigibility" in quote


def test_extract_heuristic_crux_returns_edge_shape():
    text_a = (
        "We should prioritize corrigibility and value learning in aligned systems. "
        "Moral preferences must guide agent design."
    )
    text_b = (
        "Scaling laws and compute forecasts predict AGI timelines within decades. "
        "Experimental evidence from benchmark data supports rapid capability growth."
    )
    result = extract_heuristic_crux(text_a, text_b)
    assert result["has_crux"] is True
    assert result["crux_question"]
    assert result["type"] in {"empirical", "values", "prediction"}
    assert result["evidence_a"]
    assert result["evidence_b"]


def test_extract_heuristic_crux_rejects_similar_passages():
    text = "Alignment research should focus on corrigibility and interpretability."
    result = extract_heuristic_crux(text, text)
    assert result["has_crux"] is False
    assert result["no_crux_reason"]
