"""Fetch the top comment for a LessWrong / Alignment Forum post.

The StampyAI dataset ships post bodies and a ``comment_count`` but **not** the
comment text. To build per-post double cruxes we pull the single highest-karma
top-level comment straight from the public GraphQL APIs (no key required):

  * lesswrong.com/graphql
  * alignmentforum.org/graphql

Results are cached to an append-only JSONL file keyed by post id so repeated
builds (and CI) stay cheap and work offline once warmed.
"""

from __future__ import annotations

import json
import re
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

POST_ID_RE = re.compile(r"/posts/([A-Za-z0-9]+)")

# LessWrong and Alignment Forum share one backend/database, so the LW GraphQL
# endpoint can serve comments for *both* (the AF endpoint is aggressively rate
# limited and returns 429s). Route everything through LessWrong.
LESSWRONG_ENDPOINT = "https://www.lesswrong.com/graphql"
ENDPOINTS = {
    "lesswrong": LESSWRONG_ENDPOINT,
    "alignmentforum": LESSWRONG_ENDPOINT,
}
DEFAULT_ENDPOINT = LESSWRONG_ENDPOINT

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0


class CommentFetchError(RuntimeError):
    """Raised when a comment fetch fails for transient/network reasons.

    Distinct from a successful fetch that simply has no comments (``None``), so
    callers can avoid caching failures as permanent empty results.
    """

# Top-level comments only, sorted by karma; one is plenty for a "top comment".
QUERY_TEMPLATE = """{{
  comments(input: {{terms: {{view: "postCommentsTop", postId: "{post_id}", limit: 6}}}}) {{
    results {{
      _id
      baseScore
      deleted
      topLevelCommentId
      user {{ displayName }}
      contents {{ plaintextMainText }}
    }}
  }}
}}"""

_SSL_CONTEXT: ssl.SSLContext | None = None


@dataclass(frozen=True)
class TopComment:
    comment_id: str
    author: str
    score: int
    text: str


def _ssl_context() -> ssl.SSLContext:
    global _SSL_CONTEXT
    if _SSL_CONTEXT is None:
        try:
            import certifi

            _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
        except Exception:  # pragma: no cover - fallback for missing certifi
            _SSL_CONTEXT = ssl._create_unverified_context()
    return _SSL_CONTEXT


def extract_post_id(url: str) -> str | None:
    """Pull the LW/AF post id out of a canonical post URL."""
    match = POST_ID_RE.search(url or "")
    return match.group(1) if match else None


def endpoint_for_source(source: str) -> str:
    return ENDPOINTS.get((source or "").lower(), DEFAULT_ENDPOINT)


def _graphql(endpoint: str, query: str, *, timeout: float) -> dict:
    payload = json.dumps({"query": query}).encode()
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        request = urllib.request.Request(
            endpoint,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "alignment-crux-map/0.2 (+https://github.com/)",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=_ssl_context()) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            # Back off on rate limiting / transient server errors and retry.
            if exc.code in (429, 500, 502, 503) and attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
                continue
            raise CommentFetchError(str(exc)) from exc
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
                continue
            raise CommentFetchError(str(exc)) from exc
    raise CommentFetchError(str(last_error))


def _pick_top_comment(results: list[dict]) -> TopComment | None:
    best: TopComment | None = None
    for row in results:
        if row.get("deleted"):
            continue
        # Keep it to genuine top-level comments (a reply's topLevelCommentId
        # points elsewhere; a root comment's points at itself or is null).
        top_level = row.get("topLevelCommentId")
        if top_level and top_level != row.get("_id"):
            continue
        text = ((row.get("contents") or {}).get("plaintextMainText") or "").strip()
        if not text:
            continue
        score = int(row.get("baseScore") or 0)
        if best is None or score > best.score:
            best = TopComment(
                comment_id=row.get("_id") or "",
                author=((row.get("user") or {}).get("displayName") or "unknown"),
                score=score,
                text=text,
            )
    return best


def fetch_top_comment(
    *,
    url: str,
    source: str,
    timeout: float = 30.0,
) -> TopComment | None:
    """Fetch the highest-karma top-level comment for a post.

    Returns ``None`` when the post genuinely has no usable comment. Raises
    :class:`CommentFetchError` on a transient/network failure so the caller can
    avoid caching the failure as an empty result.
    """
    post_id = extract_post_id(url)
    if not post_id:
        return None
    data = _graphql(endpoint_for_source(source), QUERY_TEMPLATE.format(post_id=post_id), timeout=timeout)
    results = (((data or {}).get("data") or {}).get("comments") or {}).get("results") or []
    return _pick_top_comment(results)


def load_comment_cache(path: Path) -> dict[str, dict | None]:
    """Load an append-only JSONL cache keyed by post id (last write wins)."""
    if not path.exists():
        return {}
    cache: dict[str, dict | None] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cache[row["post_id"]] = row.get("comment")
    return cache


def append_comment_cache(path: Path, post_id: str, comment: dict | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"post_id": post_id, "comment": comment}) + "\n")


def comment_to_dict(comment: TopComment) -> dict:
    return {
        "comment_id": comment.comment_id,
        "author": comment.author,
        "score": comment.score,
        "text": comment.text,
    }
