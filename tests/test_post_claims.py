import json

from scripts.build_crux_map import AUTHORED_CLAIMS_PATH, get_authored_claims
from scripts.post_claims import looks_like_transcript, summarize_claims


def test_looks_like_transcript_explicit_title():
    assert looks_like_transcript("Some body text.", "Full transcript of my talk") is True


def test_looks_like_transcript_podcast_title_needs_speaker_turns():
    body = "\n".join(f"**Daniel Filan:** q{i}\nJan Leike: a{i}" for i in range(8))
    assert looks_like_transcript(body, "AXRP Episode 24 - Superalignment") is True


def test_looks_like_transcript_podcast_title_alone_is_not_enough():
    body = "Some announcements: there's a survey, the store is closing, and a Patreon."
    assert looks_like_transcript(body, "AXRP announcement: Survey, Store Closing") is False


def test_looks_like_transcript_by_many_speaker_turns():
    body = "\n".join(f"**Daniel Filan:** q{i}\nQuintin Pope: a{i}" for i in range(20))
    assert looks_like_transcript(body, "Shard Theory deep dive") is True


def test_looks_like_transcript_false_for_short_illustrative_dialogue():
    body = "\n".join(f"Alice: clue {i}\nBob: response {i}" for i in range(4))
    assert looks_like_transcript(body, "How do low level hypotheses constrain high level ones?") is False


def test_looks_like_transcript_false_for_normal_post():
    body = (
        "Recently I have been learning about industry norms and incentive structures. "
        "I wanted to share some findings because they may be important."
    )
    assert looks_like_transcript(body, "The 6D effect") is False


def test_authored_claims_file_is_valid_and_nonempty():
    assert AUTHORED_CLAIMS_PATH.exists()
    data = json.loads(AUTHORED_CLAIMS_PATH.read_text(encoding="utf-8"))
    claims = data["claims"]
    assert claims
    for post_id, post_claims in claims.items():
        assert isinstance(post_id, str) and post_id
        assert 1 <= len(post_claims) <= 2
        assert all(isinstance(c, str) and c.strip() for c in post_claims)


def test_get_authored_claims_is_cached():
    assert get_authored_claims() is get_authored_claims()


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
