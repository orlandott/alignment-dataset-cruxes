#!/usr/bin/env python3
"""Build cruxes.json for the Crux Map visualization.

New approach (per-post, no cross-article edges):

  1. Load LessWrong + Alignment Forum posts from HuggingFace.
  2. Embed each post with dense, torch-free semantic vectors (model2vec) and
     project to 3D with PCA — falling back to TF-IDF + TruncatedSVD (LSA) when
     the embedding model is unavailable. (TF-IDF is always kept for labels.)
  3. Cluster posts with k-means on a higher-dimensional projection (up to
     CLUSTER_COMPONENTS), not just the 3 axes shown, so clusters capture
     structure the 3D view drops. k is auto-selected by silhouette score
     ("however many clusters make sense") unless --clusters is given.
  4. For each post: summarize its claim(s), fetch its top comment from the
     public LW/AF GraphQL API, and — when the comment disagrees — extract the
     double crux between the post and that comment.
  5. Write post nodes (with cluster + 3D PCA coords + summary + top comment +
     double crux) to cruxes.json. There are no edges.

Everything runs keyless by default (heuristic crux + free GraphQL comments,
both cached). --method anthropic upgrades summaries/cruxes when a key is set.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from huggingface_hub import hf_hub_download
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from scripts import comments as comments_mod
from scripts.double_crux import (
    analyze_top_comment,
    format_comment_claim,
    summarize_comment_claim,
)
from scripts.post_claims import looks_like_transcript, summarize_claims

AUTHORED_CLAIMS_PATH = ROOT / "data" / "authored_claims.json"
AUTHORED_CRUXES_PATH = ROOT / "data" / "authored_cruxes.json"
COMMENT_CACHE_PATH = ROOT / "data" / "processed" / "comments_cache.jsonl"
REFERENCE_CACHE_PATH = ROOT / "data" / "processed" / "references_cache.jsonl"

REPO_ID = "StampyAI/alignment-research-dataset"
SPLITS = ("lesswrong", "alignmentforum")
MODEL = "claude-sonnet-4-20250514"
# Torch-free static sentence embeddings (pure NumPy at inference). Dense
# semantic vectors pack far more variance into 3 components than sparse TF-IDF.
EMBED_MODEL = "minishlab/potion-base-8M"
MAX_PASSAGE_CHARS = 4_000
MAX_COMMENT_CHARS = 1_600

# Components used for clustering. The map only shows 3, but k-means runs on a
# higher-dimensional projection so clusters reflect structure the 3D view drops.
# ~20 is the sweet spot: it captures far more structure than 3D, while avoiding
# the distance concentration ("curse of dimensionality") that washes out
# cluster separation at 50D.
CLUSTER_COMPONENTS = 20

# Auto-k search range for silhouette selection.
MIN_K = 3
MAX_K = 10

_authored_claims_cache: dict[str, list[str]] | None = None
_authored_cruxes_cache: dict[str, dict] | None = None


@dataclass(frozen=True)
class Post:
    id: str
    title: str
    url: str
    source: str
    date_published: str
    authors: tuple[str, ...]
    text: str
    karma: int = 0
    comment_count: int = 0

    @property
    def primary_author(self) -> str:
        return self.authors[0] if self.authors else "unknown"


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def slugify(author: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", author.strip()).strip("_") or "unknown"


def truncate(text: str, limit: int = MAX_PASSAGE_CHARS) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


_MD_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_URL = re.compile(r"https?://\S+")
_CSS_DECL = re.compile(r"\{[^{}]*\}")
_CSS_SELECTOR = re.compile(r"[.#][A-Za-z][\w]*-[\w-]*")
_MJX_TOKEN = re.compile(r"\b(?:mjx|MJX)[\w-]*")


def clean_text(text: str) -> str:
    """Strip HTML, markdown links/images, bare URLs, and MathJax CSS noise.

    Post bodies from the dataset are markdown/HTML; without this, URL and
    markup tokens dominate TF-IDF and leak into cluster terms / crux questions.
    """
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = _MD_LINK.sub(r"\1", text)
    text = _URL.sub(" ", text)
    text = _CSS_DECL.sub(" ", text)
    text = _CSS_SELECTOR.sub(" ", text)
    text = _MJX_TOKEN.sub(" ", text)
    text = re.sub(r"@[A-Za-z-]+", " ", text)
    text = re.sub(r"[*_`>#]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def get_authored_claims() -> dict[str, list[str]]:
    """Load per-post authored claim summaries (cached), keyed by post id."""
    global _authored_claims_cache
    if _authored_claims_cache is None:
        if AUTHORED_CLAIMS_PATH.exists():
            data = json.loads(AUTHORED_CLAIMS_PATH.read_text(encoding="utf-8"))
            _authored_claims_cache = data.get("claims", {})
        else:
            _authored_claims_cache = {}
    return _authored_claims_cache


def get_authored_cruxes() -> dict[str, dict]:
    """Load hand-authored post↔top-comment double cruxes, keyed by post id.

    Each entry overrides the heuristic for that post. ``has_crux: true`` supplies
    a high-quality crux (question/type/evidence); ``has_crux: false`` records that
    the top comment is not really a disagreement, so no crux is shown.
    """
    global _authored_cruxes_cache
    if _authored_cruxes_cache is None:
        if AUTHORED_CRUXES_PATH.exists():
            data = json.loads(AUTHORED_CRUXES_PATH.read_text(encoding="utf-8"))
            _authored_cruxes_cache = data.get("cruxes", {})
        else:
            _authored_cruxes_cache = {}
    return _authored_cruxes_cache


def load_jsonl_split(split: str) -> list[dict]:
    path = hf_hub_download(REPO_ID, f"{split}.jsonl", repo_type="dataset")
    rows: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_posts(max_posts: int, top_authors: int) -> tuple[list[Post], list[str]]:
    """Return posts (most recent first, capped) and the active author list."""
    raw_rows: list[dict] = []
    for split in SPLITS:
        raw_rows.extend(load_jsonl_split(split))

    eligible: list[dict] = []
    for row in raw_rows:
        authors = [a.strip() for a in (row.get("authors") or []) if a and a.strip()]
        date = (row.get("date_published") or "").strip()
        text = (row.get("text") or "").strip()
        url = (row.get("url") or "").strip()
        # Need a URL with a post id so we can fetch its comments.
        if not authors or not date or not text or not comments_mod.extract_post_id(url):
            continue
        eligible.append({**row, "authors": authors})

    author_counts: Counter[str] = Counter()
    for row in eligible:
        for author in row["authors"]:
            author_counts[author] += 1

    selected_set = {author for author, _ in author_counts.most_common(top_authors)}
    filtered = [row for row in eligible if any(a in selected_set for a in row["authors"])]

    # Collapse LW/AF cross-posts (same title + author) to the highest-karma copy
    # so the map shows each post once.
    by_key: dict[tuple[str, str], dict] = {}
    for row in filtered:
        key = ((row.get("title") or "").strip().lower(), row["authors"][0])
        kept = by_key.get(key)
        if kept is None or int(row.get("karma") or 0) > int(kept.get("karma") or 0):
            by_key[key] = row
    filtered = list(by_key.values())

    filtered.sort(key=lambda row: row.get("date_published") or "", reverse=True)
    filtered = filtered[:max_posts]

    posts = [
        Post(
            id=row["id"],
            title=row.get("title") or "Untitled",
            url=row.get("url") or "",
            source=row.get("source") or "",
            date_published=row["date_published"],
            authors=tuple(row["authors"]),
            text=row["text"],
            karma=int(row.get("karma") or 0),
            comment_count=int(row.get("comment_count") or 0),
        )
        for row in filtered
    ]

    active_authors = sorted(
        {a for post in posts for a in post.authors if a in selected_set},
        key=lambda a: (-author_counts[a], a),
    )
    return posts, active_authors


def build_tfidf_matrix(texts: list[str]):
    min_df = 1 if len(texts) < 3 else 2
    vectorizer = TfidfVectorizer(
        max_features=20_000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=min_df,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def _pick_axis_terms(
    features: np.ndarray,
    order: np.ndarray,
    *,
    top_n: int,
) -> list[str]:
    picked: list[str] = []
    for idx in order:
        term = features[idx]
        if len(term) < 3 or "{" in term or term.startswith("mjx"):
            continue
        picked.append(term)
        if len(picked) >= top_n:
            break
    return picked


def describe_pca_axes(
    scores: np.ndarray,
    vectorizer: TfidfVectorizer,
    matrix,
    variance: list[float],
    *,
    top_n: int = 3,
) -> list[dict]:
    """Label each layout axis from distinctive TF-IDF terms at its poles.

    Works on the raw per-axis projection ``scores`` (n_posts × n_components),
    regardless of whether those came from dense embeddings or sparse TF-IDF,
    so the axes stay interpretable as words even when the geometry is semantic.
    Returns one entry per component, mapped to scene axes x=PC1, y=PC2, z=PC3.
    """
    axis_names = ("x", "y", "z")
    try:
        features = vectorizer.get_feature_names_out()
    except Exception:
        return []
    dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    labels: list[dict] = []
    n_axes = min(scores.shape[1], 3) if scores.ndim == 2 else 0
    for i in range(n_axes):
        col = scores[:, i]
        if np.allclose(col, 0):
            continue
        hi = col >= np.percentile(col, 75)
        lo = col <= np.percentile(col, 25)
        if not hi.any() or not lo.any():
            continue
        diff = dense[hi].mean(axis=0) - dense[lo].mean(axis=0)
        pos = _pick_axis_terms(features, np.argsort(-diff), top_n=top_n)
        neg = _pick_axis_terms(features, np.argsort(diff), top_n=top_n)
        labels.append(
            {
                "axis": axis_names[i],
                "component": i + 1,
                "positive": " · ".join(pos),
                "negative": " · ".join(neg),
                "variance_explained": round(float(variance[i]) if i < len(variance) else 0.0, 4),
            }
        )
    return labels


_embedder = None
_embedder_unavailable = False


def embed_texts(texts: list[str]) -> np.ndarray | None:
    """Dense semantic embeddings via model2vec (torch-free).

    Returns an (n, dim) array, or None when the library/model is unavailable
    (e.g. offline CI), in which case callers fall back to TF-IDF + SVD.
    """
    global _embedder, _embedder_unavailable
    if _embedder_unavailable:
        return None
    if _embedder is None:
        try:
            from model2vec import StaticModel

            _embedder = StaticModel.from_pretrained(EMBED_MODEL)
        except Exception as exc:  # missing dep or failed download
            log(f"Dense embeddings unavailable ({exc}); using TF-IDF instead.")
            _embedder_unavailable = True
            return None
    try:
        return np.asarray(_embedder.encode(list(texts)), dtype=float)
    except Exception as exc:
        log(f"Embedding failed ({exc}); using TF-IDF instead.")
        _embedder_unavailable = True
        return None


@dataclass
class PostGeometry:
    """3D layout plus the artifacts needed to cluster, and to label axes."""

    coords: np.ndarray  # (n, 3) L2-normalized, drives map positions only
    scores: np.ndarray  # (n, 3) pre-normalization projection, for axis labels
    cluster_features: np.ndarray  # (n, <=CLUSTER_COMPONENTS) L2-normalized, for k-means
    vectorizer: TfidfVectorizer  # always TF-IDF, used only for word labels
    matrix: object  # TF-IDF matrix, used only for word labels
    variance: list[float]  # explained variance ratio per component
    reduction: str  # "pca" (dense embeddings) or "truncated_svd" (TF-IDF)
    embedding_model: str | None  # model id when dense, else None


def _reduce(
    features,
    *,
    dense: bool,
    max_components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Reduce a feature matrix; return (coords3d, scores3d, cluster_features, variance).

    Dense embeddings use centered PCA; sparse TF-IDF uses TruncatedSVD (LSA).
    A single reducer is fit to up to ``max_components`` dimensions; the first 3
    (L2-normalized) drive the map layout, while the full L2-normalized
    projection is used for clustering — so clustering sees far more of the
    semantic structure than the 3 axes shown on screen.
    """
    n_samples, n_features = features.shape[0], features.shape[1]
    zeros3 = np.zeros((n_samples, 3))
    n_components = min(max_components, n_samples - 1, n_features - 1)
    if n_components < 1:
        return zeros3, zeros3, zeros3, []
    reducer = (
        PCA(n_components=n_components, random_state=42)
        if dense
        else TruncatedSVD(n_components=n_components, random_state=42)
    )
    projection = reducer.fit_transform(features)
    variance = [float(v) for v in reducer.explained_variance_ratio_]

    scores = projection[:, :3]
    coords = normalize(scores)
    if coords.shape[1] < 3:
        pad = np.zeros((coords.shape[0], 3 - coords.shape[1]))
        coords = np.hstack([coords, pad])
        scores = np.hstack([scores, np.zeros((scores.shape[0], 3 - scores.shape[1]))])

    cluster_features = normalize(projection)  # angular geometry in the full space
    return coords, scores, cluster_features, variance


def compute_post_coords(
    posts: list[Post],
    *,
    use_embeddings: bool = True,
) -> PostGeometry:
    """Project posts to a 3D layout, plus higher-dim features for clustering.

    Primary path: dense, torch-free semantic embeddings (model2vec) reduced
    with PCA — far more variance and cleaner topical geometry than sparse word
    counts. Fallback (no embedding model, or ``use_embeddings`` False): TF-IDF
    reduced with TruncatedSVD (LSA). TF-IDF is always computed regardless,
    because cluster and axis *labels* are word-based.

    The map shows the first 3 components, but clustering runs on up to
    ``CLUSTER_COMPONENTS`` components so it captures structure the 3D view
    cannot.
    """
    if not posts:
        return PostGeometry(
            np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 0)),
            TfidfVectorizer(), np.zeros((0, 0)), [], "truncated_svd", None,
        )

    texts = [truncate(clean_text(post.text), 4_000) for post in posts]
    vectorizer, matrix = build_tfidf_matrix(texts)

    embeddings = embed_texts(texts) if use_embeddings else None
    if embeddings is not None and embeddings.shape[0] == len(posts):
        coords, scores, cluster_features, variance = _reduce(
            embeddings, dense=True, max_components=CLUSTER_COMPONENTS
        )
        return PostGeometry(
            coords, scores, cluster_features, vectorizer, matrix, variance, "pca", EMBED_MODEL
        )

    coords, scores, cluster_features, variance = _reduce(
        matrix, dense=False, max_components=CLUSTER_COMPONENTS
    )
    return PostGeometry(
        coords, scores, cluster_features, vectorizer, matrix, variance, "truncated_svd", None
    )


def choose_k(coords: np.ndarray, *, min_k: int = MIN_K, max_k: int = MAX_K) -> int:
    """Pick the number of clusters with the best silhouette score."""
    n = coords.shape[0]
    if n < 4:
        return max(1, n)
    upper = min(max_k, n - 1)
    lower = min(min_k, upper)
    best_k, best_score = lower, -1.0
    for k in range(lower, upper + 1):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(coords)
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score(coords, labels))
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def compute_clusters(coords: np.ndarray, n_clusters: int) -> np.ndarray:
    """Return an (n,) array of k-means labels for the given coords."""
    n = coords.shape[0]
    k = min(n_clusters, n)
    if k < 2 or n == 0:
        return np.zeros(n, dtype=int)
    return KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(coords)


# Contraction fragments ("don't" → "don") and weak generic words that the
# tokenizer leaves behind; they look distinctive numerically but read as noise.
_LABEL_STOP = frozenset(
    {
        "don", "doesn", "didn", "isn", "wasn", "aren", "weren", "haven", "hasn",
        "hadn", "won", "wouldn", "couldn", "shouldn", "ain", "ve", "ll", "re",
        "wasn t", "doesn t", "don t", "didn t",
        "things", "thing", "know", "lot", "way", "ways", "actually", "really",
        "maybe", "doesn t", "kind", "sort", "stuff", "bit", "yeah", "okay",
    }
)


def _is_label_term(term: str) -> bool:
    if len(term) < 3 or "{" in term or term.startswith("mjx"):
        return False
    if term in _LABEL_STOP:
        return False
    # Reject pure-fragment bigrams where either half is a contraction fragment.
    parts = term.split()
    return not any(p in _LABEL_STOP for p in parts)


def cluster_top_terms(
    matrix,
    vectorizer: TfidfVectorizer,
    labels: np.ndarray,
    *,
    top_n: int = 4,
) -> dict[int, list[str]]:
    """Name each cluster by its most *distinctive* terms.

    Ranking by raw mean TF-IDF surfaces globally common words ("model", "things")
    that describe every cluster. Instead we score each term by how much more it
    appears in the cluster than in the rest of the corpus (mean inside minus mean
    outside), which yields crisp, differentiating labels. Contraction fragments
    and weak filler words are filtered out so labels read as real topics.
    """
    try:
        features = vectorizer.get_feature_names_out()
    except Exception:
        return {}
    dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    terms: dict[int, list[str]] = {}
    for cluster in sorted(set(int(c) for c in labels)):
        mask = labels == cluster
        if not mask.any():
            continue
        in_mean = dense[mask].mean(axis=0)
        out_mean = dense[~mask].mean(axis=0) if (~mask).any() else np.zeros_like(in_mean)
        score = in_mean - out_mean
        order = np.argsort(-score)
        picked: list[str] = []
        for idx in order:
            term = features[idx]
            if not _is_label_term(term):
                continue
            picked.append(term)
            if len(picked) >= top_n:
                break
        terms[cluster] = picked
    return terms


def cluster_exemplars(
    space: np.ndarray,
    labels: np.ndarray,
    posts: list[Post],
    *,
    top_n: int = 3,
) -> dict[int, list[str]]:
    """Return, per cluster, the titles of the posts closest to its centroid.

    Exemplar titles make a cluster's theme legible in a way bag-of-words terms
    often cannot (e.g. a list of concrete post titles vs. "agent · mind · tree").
    """
    exemplars: dict[int, list[str]] = {}
    for cluster in sorted(set(int(c) for c in labels)):
        idx = np.where(labels == cluster)[0]
        if idx.size == 0:
            continue
        centroid = space[idx].mean(axis=0)
        dist = np.linalg.norm(space[idx] - centroid, axis=1)
        nearest = idx[np.argsort(dist)[:top_n]]
        exemplars[cluster] = [posts[i].title for i in nearest]
    return exemplars


SYSTEM_PROMPT = (
    "You are a precise philosophical analyst. Given an alignment-research post and a "
    "high-karma comment that pushes back on it, identify the single most important double "
    "crux: the specific falsifiable question where the post author and commenter diverge. "
    "Respond only in valid JSON, no markdown."
)

USER_PROMPT_TEMPLATE = """Find the double crux between this post and its top comment.

POST — {author}, {source} ({date})
{post_text}

TOP COMMENT — {commenter} (karma {score})
{comment_text}

Return: {{ "has_crux": true/false, "no_crux_reason": "string or null", "crux_question": "one specific falsifiable question or null", "type": "empirical | values | prediction | null", "evidence_post": "1-2 sentence quote or null", "evidence_comment": "1-2 sentence quote or null" }}"""


def parse_model_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def anthropic_double_crux(post: Post, comment: comments_mod.TopComment) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for --method anthropic")

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        author=post.primary_author,
        source=post.source,
        date=post.date_published,
        post_text=truncate(clean_text(post.text)),
        commenter=comment.author,
        score=comment.score,
        comment_text=truncate(comment.text, MAX_COMMENT_CHARS),
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return parse_model_json(response.content[0].text)


def post_summary(post: Post) -> list[str]:
    authored = get_authored_claims().get(post.id)
    if not authored and looks_like_transcript(post.text, post.title):
        kind = (
            "podcast episode"
            if re.search(r"axrp|podcast", post.title or "", re.IGNORECASE)
            else "recorded discussion"
        )
        return [
            f"This post is a transcript of a {kind}, so it doesn't argue a single "
            "main claim — open the original to read the conversation."
        ]
    claims = authored or summarize_claims(clean_text(post.text))
    if not claims:
        return []
    first = claims[0]
    if not first.lower().startswith("the main claim is"):
        first = f"The main claim is: {first}"
    return [first, *claims[1:]]


def resolve_top_comment(
    post: Post,
    *,
    cache: dict[str, dict | None],
    cache_path: Path,
    offline: bool,
) -> comments_mod.TopComment | None:
    """Return the cached/fetched top comment for a post (or None)."""
    if post.id in cache:
        raw = cache[post.id]
        if not raw:
            return None
        return comments_mod.TopComment(
            comment_id=raw.get("comment_id", ""),
            author=raw.get("author", "unknown"),
            score=int(raw.get("score") or 0),
            text=raw.get("text", ""),
        )

    if offline:
        return None

    try:
        comment = comments_mod.fetch_top_comment(url=post.url, source=post.source)
    except comments_mod.CommentFetchError as exc:
        # Transient failure: skip caching so a later build retries this post.
        log(f"  comment fetch failed ({post.id}): {exc}")
        return None

    payload = comments_mod.comment_to_dict(comment) if comment else None
    comments_mod.append_comment_cache(cache_path, post.id, payload)
    cache[post.id] = payload
    return comment


def resolve_referenced_by(
    post: Post,
    *,
    cache: dict[str, int | None],
    cache_path: Path,
    offline: bool,
) -> int | None:
    """Return how many other posts reference this post (cached/fetched)."""
    if post.id in cache:
        return cache[post.id]
    if offline:
        return None
    try:
        count = comments_mod.fetch_referenced_by(url=post.url, source=post.source)
    except comments_mod.CommentFetchError as exc:
        log(f"  pingback fetch failed ({post.id}): {exc}")
        return None
    comments_mod.append_reference_cache(cache_path, post.id, count)
    cache[post.id] = count
    return count


def build_comment_block(
    post: Post,
    comment: comments_mod.TopComment | None,
    *,
    method: str,
) -> dict | None:
    if comment is None:
        return None

    authored = get_authored_cruxes().get(post.id) if method != "anthropic" else None
    if authored is not None:
        # Hand-authored override: trust it for both the disagreement decision
        # and the crux text (the heuristic produces template-y questions).
        has_crux = bool(authored.get("has_crux"))
        raw_claim = authored.get("comment_claim") or summarize_comment_claim(comment.text)
        analysis = {
            "claim": format_comment_claim(raw_claim, disagrees=has_crux),
            "disagrees": has_crux,
            "crux": {
                "has_crux": True,
                "crux_question": authored.get("crux_question"),
                "type": authored.get("type"),
                "evidence_post": authored.get("evidence_post"),
                "evidence_comment": authored.get("evidence_comment"),
            }
            if has_crux
            else None,
        }
    elif method == "anthropic":
        crux = anthropic_double_crux(post, comment)
        disagrees = bool(crux.get("has_crux"))
        analysis = {
            "claim": format_comment_claim(
                summarize_comment_claim(comment.text), disagrees=disagrees
            ),
            "disagrees": disagrees,
            "crux": crux if disagrees else None,
        }
    else:
        analysis = analyze_top_comment(clean_text(post.text), comment.text)

    return {
        "author": comment.author,
        "score": comment.score,
        "text": truncate(comment.text, MAX_COMMENT_CHARS),
        "claim": analysis["claim"],
        "disagrees": analysis["disagrees"],
        "crux": analysis["crux"],
    }


def build_graph(
    posts: list[Post],
    *,
    method: str,
    n_clusters: int,
    comment_cache_path: Path,
    reference_cache_path: Path,
    offline: bool,
    dry_run: bool,
) -> dict:
    geo = compute_post_coords(posts)
    coords = geo.coords
    cluster_space = geo.cluster_features
    axis_labels = describe_pca_axes(geo.scores, geo.vectorizer, geo.matrix, geo.variance)

    if n_clusters and n_clusters > 0:
        k = min(n_clusters, len(posts))
    else:
        k = choose_k(cluster_space)
    labels = compute_clusters(cluster_space, k)
    terms = cluster_top_terms(geo.matrix, geo.vectorizer, labels)
    exemplars = cluster_exemplars(cluster_space, labels, posts)

    comment_cache = comments_mod.load_comment_cache(comment_cache_path)
    reference_cache = comments_mod.load_reference_cache(reference_cache_path)

    nodes = []
    comment_total = 0
    crux_total = 0
    referenced_total = 0
    for index, post in enumerate(posts):
        row = coords[index] if index < coords.shape[0] else np.zeros(3)
        cluster = int(labels[index]) if index < len(labels) else 0

        comment_block: dict | None = None
        referenced_by: int | None = None
        if not dry_run:
            comment = resolve_top_comment(
                post,
                cache=comment_cache,
                cache_path=comment_cache_path,
                offline=offline,
            )
            comment_block = build_comment_block(post, comment, method=method)
            if comment_block:
                comment_total += 1
                crux = comment_block.get("crux")
                if crux and crux.get("has_crux"):
                    crux_total += 1

            referenced_by = resolve_referenced_by(
                post,
                cache=reference_cache,
                cache_path=reference_cache_path,
                offline=offline,
            )
            if referenced_by:
                referenced_total += 1

        nodes.append(
            {
                "id": post.id,
                "label": post.title,
                "author": post.primary_author,
                "authors": list(post.authors),
                "source": post.source,
                "url": post.url,
                "date": post.date_published,
                "karma": post.karma,
                "comment_count": post.comment_count,
                "referenced_by": referenced_by,
                "cluster": cluster,
                "x": float(row[0]),
                "y": float(row[1]),
                "z": float(row[2]),
                "summary": post_summary(post),
                "top_comment": comment_block,
            }
        )

    cluster_meta = [
        {
            "id": cluster,
            "terms": terms.get(cluster, []),
            "exemplars": exemplars.get(cluster, []),
            "size": int((labels == cluster).sum()),
        }
        for cluster in sorted(set(int(c) for c in labels))
    ]

    return {
        "meta": {
            "generated_at": datetime.now(UTC).isoformat(),
            "dataset": REPO_ID,
            "splits": list(SPLITS),
            "method": None if dry_run else method,
            "model": None if dry_run or method != "anthropic" else MODEL,
            "post_count": len(posts),
            "cluster_count": len(cluster_meta),
            "k_selected": int(k),
            "auto_k": not (n_clusters and n_clusters > 0),
            "comment_count": comment_total,
            "double_crux_count": crux_total,
            "referenced_post_count": referenced_total,
            "components": 3,
            "cluster_components": int(cluster_space.shape[1]) if cluster_space.ndim == 2 else 0,
            "reduction": geo.reduction,
            "embedding_model": geo.embedding_model,
            "variance_explained": round(float(sum(geo.variance[:3])), 4),
            "axis_labels": axis_labels,
        },
        "clusters": cluster_meta,
        "nodes": nodes,
        "edges": [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cruxes.json for Crux Map")
    parser.add_argument("--max-posts", type=int, default=400)
    parser.add_argument("--top-authors", type=int, default=40)
    parser.add_argument(
        "--clusters",
        type=int,
        default=0,
        help="k-means clusters (0 = auto-select by silhouette; default 0)",
    )
    parser.add_argument(
        "--method",
        choices=("heuristic", "anthropic"),
        default="heuristic",
        help="Summary/double-crux source (default: heuristic, keyless)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Never hit the network for comments; use only the cache",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build nodes + PCA + clusters only (no comments/cruxes)",
    )
    parser.add_argument(
        "--comment-cache",
        type=Path,
        default=COMMENT_CACHE_PATH,
        help="Append-only cache of fetched top comments",
    )
    parser.add_argument(
        "--reference-cache",
        type=Path,
        default=REFERENCE_CACHE_PATH,
        help="Append-only cache of fetched pingback (referenced-by) counts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "cruxes.json",
        help="Output graph JSON (default: ./cruxes.json)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log("Loading posts from HuggingFace...")
    posts, authors = parse_posts(args.max_posts, args.top_authors)
    log(f"Loaded {len(posts)} posts across {len(authors)} authors")

    log("Building post map (PCA + clusters + comments + double cruxes)...")
    graph = build_graph(
        posts,
        method=args.method,
        n_clusters=args.clusters,
        comment_cache_path=args.comment_cache,
        reference_cache_path=args.reference_cache,
        offline=args.offline,
        dry_run=args.dry_run,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(graph, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    meta = graph["meta"]
    log(
        f"Wrote {args.output} ({meta['post_count']} posts, "
        f"{meta['cluster_count']} clusters (k={meta['k_selected']}), "
        f"{meta['comment_count']} top comments, "
        f"{meta['double_crux_count']} double cruxes)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
