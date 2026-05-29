from scripts.double_crux import (
    analyze_top_comment,
    detect_disagreement,
    extract_double_crux,
    summarize_comment_claim,
)

POST = (
    "We should expect short AI timelines because compute scaling has reliably "
    "produced capability gains, and current trends will continue for years."
)
DISAGREEING_COMMENT = (
    "I disagree with this. I don't think compute scaling alone predicts capability. "
    "The problem with this argument is that it ignores data bottlenecks, so timelines "
    "are much more uncertain than the post claims."
)
AGREEING_COMMENT = (
    "I agree, this is a great post. Compute scaling really does seem to drive "
    "capability gains and I think this is right."
)


def test_detect_disagreement_true():
    assert detect_disagreement(DISAGREEING_COMMENT) is True


def test_detect_disagreement_false_for_agreement():
    assert detect_disagreement(AGREEING_COMMENT) is False


def test_detect_disagreement_false_for_neutral():
    assert detect_disagreement("Here is a related link and some context.") is False


def test_summarize_comment_claim_nonempty():
    claim = summarize_comment_claim(DISAGREEING_COMMENT)
    assert isinstance(claim, str) and claim


def test_extract_double_crux_has_question_when_contrast():
    crux = extract_double_crux(POST, DISAGREEING_COMMENT)
    assert crux["has_crux"] is True
    assert crux["crux_question"]
    assert crux["type"] in {"empirical", "values", "prediction"}
    assert "evidence_post" in crux and "evidence_comment" in crux


def test_analyze_top_comment_disagreeing_attaches_crux():
    result = analyze_top_comment(POST, DISAGREEING_COMMENT)
    assert result["disagrees"] is True
    assert result["crux"] is not None
    assert result["claim"]


def test_analyze_top_comment_agreeing_has_no_crux():
    result = analyze_top_comment(POST, AGREEING_COMMENT)
    assert result["disagrees"] is False
    assert result["crux"] is None
