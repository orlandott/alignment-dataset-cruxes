from scripts.post_claims import summarize_claims


def test_summarize_claims_returns_core_assertion():
    text = (
        "Some background on the history of the field. "
        "I argue that alignment fundamentally requires corrigibility rather than capability control. "
        "A few unrelated logistical notes follow."
    )
    claims = summarize_claims(text)
    assert claims
    assert any("corrigibility" in claim.lower() for claim in claims)


def test_summarize_claims_can_return_multiple():
    text = (
        "The key point is that scaling laws will continue to drive capability gains for years. "
        "We must therefore invest far more heavily in interpretability research right now. "
        "Here is a short anecdote with no real argument attached to it whatsoever today."
    )
    claims = summarize_claims(text, max_claims=2)
    assert 1 <= len(claims) <= 2


def test_summarize_claims_respects_max_claims():
    text = " ".join(
        f"We must therefore prioritize approach number {i} because it clearly matters most of all."
        for i in range(6)
    )
    claims = summarize_claims(text, max_claims=2)
    assert len(claims) <= 2


def test_summarize_claims_empty_for_markup_only():
    assert summarize_claims("<div></div>") == []


def test_summarize_claims_empty_for_short_text():
    assert summarize_claims("Too short.") == []


def test_summarize_claims_truncates_long_sentence():
    # ~290 chars: within the sentence-length window but above the claim cap.
    long_sentence = "We should " + "align " * 46 + "now."
    assert 240 < len(long_sentence) <= 320
    claims = summarize_claims(long_sentence)
    assert claims
    assert all(len(claim) <= 241 for claim in claims)
    assert claims[0].endswith("\u2026")
