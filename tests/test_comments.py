from scripts.comments import (
    TopComment,
    append_comment_cache,
    append_reference_cache,
    comment_to_dict,
    endpoint_for_source,
    extract_post_id,
    load_comment_cache,
    load_reference_cache,
    _pick_top_comment,
)


def test_extract_post_id():
    url = "https://www.lesswrong.com/posts/yxFkuyPANtL6GSwiC/the-majority-is-always-wrong"
    assert extract_post_id(url) == "yxFkuyPANtL6GSwiC"
    assert extract_post_id("https://example.com/no-post-here") is None
    assert extract_post_id("") is None


def test_endpoint_for_source_routes_through_lesswrong():
    # AF shares the LW backend and the AF endpoint is rate-limited, so every
    # source resolves to the LessWrong GraphQL endpoint.
    assert "lesswrong" in endpoint_for_source("alignmentforum")
    assert "lesswrong" in endpoint_for_source("lesswrong")
    assert "lesswrong" in endpoint_for_source("unknown-source")


def test_pick_top_comment_skips_deleted_and_replies():
    results = [
        {"_id": "a", "baseScore": 5, "deleted": True, "topLevelCommentId": "a",
         "user": {"displayName": "X"}, "contents": {"plaintextMainText": "deleted"}},
        {"_id": "b", "baseScore": 20, "deleted": False, "topLevelCommentId": "root",
         "user": {"displayName": "Reply"}, "contents": {"plaintextMainText": "a reply"}},
        {"_id": "c", "baseScore": 12, "deleted": False, "topLevelCommentId": "c",
         "user": {"displayName": "Top"}, "contents": {"plaintextMainText": "real top comment"}},
    ]
    top = _pick_top_comment(results)
    assert top is not None
    assert top.author == "Top"
    assert top.score == 12


def test_comment_cache_roundtrip(tmp_path):
    path = tmp_path / "comments.jsonl"
    append_comment_cache(path, "post1", {"author": "X", "score": 3, "text": "hi"})
    append_comment_cache(path, "post2", None)
    cache = load_comment_cache(path)
    assert cache["post1"]["author"] == "X"
    assert cache["post2"] is None


def test_comment_to_dict():
    c = TopComment(comment_id="x", author="A", score=7, text="t")
    assert comment_to_dict(c) == {"comment_id": "x", "author": "A", "score": 7, "text": "t"}


def test_reference_cache_roundtrip(tmp_path):
    path = tmp_path / "references.jsonl"
    append_reference_cache(path, "post1", 5)
    append_reference_cache(path, "post2", 0)
    append_reference_cache(path, "post3", None)
    cache = load_reference_cache(path)
    assert cache["post1"] == 5
    assert cache["post2"] == 0
    assert cache["post3"] is None
