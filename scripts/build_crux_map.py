#!/usr/bin/env python3
"""Build cruxes.json for the Crux Map visualization.

Pipeline:
  1. Load LessWrong + Alignment Forum posts from HuggingFace
  2. Filter to posts with author + date, cap at ~500 from top authors
  3. Find author pairs with overlapping topics (TF-IDF cosine)
  4. For each pair, pick the most semantically similar post pair
  5. Extract cruxes via heuristic TF-IDF contrast (default) or Anthropic API
  6. Position authors in 3D with PCA (3 components) on TF-IDF profiles
  7. Write nodes + edges to cruxes.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from huggingface_hub import hf_hub_download
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from scripts.heuristic_crux import extract_heuristic_crux
from scripts.post_claims import summarize_claims

AUTHORED_CRUXES_PATH = ROOT / "data" / "authored_cruxes.json"
DEFAULT_CLUSTERS = 5

REPO_ID = "StampyAI/alignment-research-dataset"
SPLITS = ("lesswrong", "alignmentforum")
MODEL = "claude-sonnet-4-20250514"
MAX_PASSAGE_CHARS = 4_000

SYSTEM_PROMPT = (
    "You are a precise philosophical analyst. Your job is to identify the single most "
    "important crux between two authors — the specific falsifiable question where they "
    "diverge such that resolving it would substantially change at least one person's "
    "position. If no genuine crux exists, say so. Respond only in valid JSON, no markdown."
)

USER_PROMPT_TEMPLATE = """Find the crux between these two passages.

PASSAGE A — {author_a}, {source_a} ({date_a})
{text_a}

PASSAGE B — {author_b}, {source_b} ({date_b})
{text_b}

Return: {{ "has_crux": true/false, "no_crux_reason": "string or null", "crux_question": "one specific falsifiable question or null", "type": "empirical | values | prediction | null", "evidence_a": "1-2 sentence quote or null", "evidence_b": "1-2 sentence quote or null" }}"""


@dataclass(frozen=True)
class Post:
    id: str
    title: str
    url: str
    source: str
    date_published: str
    authors: tuple[str, ...]
    text: str

    @property
    def primary_author(self) -> str:
        return self.authors[0]

    def as_edge_post(self, author: str) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "date": self.date_published,
            "author": author,
            "claims": summarize_claims(self.text),
        }


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def slugify(author: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", author.strip()).strip("_") or "unknown"


def truncate(text: str, limit: int = MAX_PASSAGE_CHARS) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


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
    raw_rows: list[dict] = []
    for split in SPLITS:
        raw_rows.extend(load_jsonl_split(split))

    eligible: list[dict] = []
    for row in raw_rows:
        authors = [a.strip() for a in (row.get("authors") or []) if a and a.strip()]
        date = (row.get("date_published") or "").strip()
        text = (row.get("text") or "").strip()
        if not authors or not date or not text:
            continue
        eligible.append({**row, "authors": authors})

    author_counts = Counter()
    for row in eligible:
        for author in row["authors"]:
            author_counts[author] += 1

    selected_authors = [author for author, _ in author_counts.most_common(top_authors)]
    selected_set = set(selected_authors)

    filtered: list[dict] = []
    for row in eligible:
        if any(author in selected_set for author in row["authors"]):
            filtered.append(row)

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
        )
        for row in filtered
    ]

    active_authors = sorted(
        {author for post in posts for author in post.authors if author in selected_set},
        key=lambda author: (-author_counts[author], author),
    )
    return posts, active_authors


def posts_by_author(posts: list[Post], authors: list[str]) -> dict[str, list[Post]]:
    author_set = set(authors)
    grouped: dict[str, list[Post]] = defaultdict(list)
    for post in posts:
        for author in post.authors:
            if author in author_set:
                grouped[author].append(post)
    return dict(grouped)


def build_tfidf_matrix(texts: list[str]) -> tuple[TfidfVectorizer, np.ndarray]:
    min_df = 1 if len(texts) < 3 else 2
    vectorizer = TfidfVectorizer(
        max_features=20_000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=min_df,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def author_corpus(posts: list[Post]) -> str:
    return "\n\n".join(truncate(post.text, 2_000) for post in posts)


def find_candidate_pairs(
    authors: list[str],
    grouped: dict[str, list[Post]],
    min_similarity: float,
    max_pairs: int,
) -> list[tuple[str, str, float]]:
    corpora = [author_corpus(grouped[author]) for author in authors]
    _, matrix = build_tfidf_matrix(corpora)
    sim = cosine_similarity(matrix)

    candidates: list[tuple[str, str, float]] = []
    for i, j in combinations(range(len(authors)), 2):
        score = float(sim[i, j])
        if score >= min_similarity:
            candidates.append((authors[i], authors[j], score))

    candidates.sort(key=lambda item: item[2], reverse=True)
    return candidates[:max_pairs]


def best_post_pair(
    author_a: str,
    author_b: str,
    grouped: dict[str, list[Post]],
) -> tuple[Post, Post, float]:
    posts_a = grouped[author_a]
    posts_b = grouped[author_b]
    texts = [truncate(post.text, 2_000) for post in posts_a + posts_b]
    _, matrix = build_tfidf_matrix(texts)
    matrix_a = matrix[: len(posts_a)]
    matrix_b = matrix[len(posts_a) :]
    sim = cosine_similarity(matrix_a, matrix_b)

    flat_index = int(np.argmax(sim))
    idx_a, idx_b = divmod(flat_index, sim.shape[1])
    return posts_a[idx_a], posts_b[idx_b], float(sim[idx_a, idx_b])


def compute_pca_coords(
    authors: list[str],
    grouped: dict[str, list[Post]],
) -> np.ndarray:
    """Return an (n_authors, 3) array of normalized PCA coordinates."""
    if not authors:
        return np.zeros((0, 3))

    corpora = [author_corpus(grouped[author]) for author in authors]
    _, matrix = build_tfidf_matrix(corpora)
    dense = matrix.toarray()

    n_components = min(3, len(authors), dense.shape[1])
    if n_components == 0:
        return np.zeros((len(authors), 3))

    scaled = StandardScaler(with_std=True).fit_transform(dense)
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(scaled)

    if coords.shape[1] < 3:
        pad = np.zeros((coords.shape[0], 3 - coords.shape[1]))
        coords = np.hstack([coords, pad])

    # Scale into a comfortable range for 3D rendering.
    max_abs = np.max(np.abs(coords)) or 1.0
    return coords / max_abs


def compute_pca_positions(
    authors: list[str],
    grouped: dict[str, list[Post]],
) -> dict[str, dict[str, float]]:
    coords = compute_pca_coords(authors, grouped)
    return {
        author: {"x": float(row[0]), "y": float(row[1]), "z": float(row[2])}
        for author, row in zip(authors, coords, strict=True)
    }


def compute_clusters(
    coords: np.ndarray,
    authors: list[str],
    n_clusters: int,
) -> dict[str, int]:
    """Group authors into clusters via k-means on their PCA coordinates."""
    k = min(n_clusters, len(authors))
    if k < 2:
        return {author: 0 for author in authors}

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords)
    return {author: int(label) for author, label in zip(authors, labels, strict=True)}


def pair_cache_key(
    author_a: str,
    author_b: str,
    post_a_id: str,
    post_b_id: str,
    *,
    method: str = "heuristic",
) -> str:
    parts = sorted([author_a, author_b]) + sorted([post_a_id, post_b_id]) + [method]
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
    return f"{parts[0]}__{parts[1]}__{digest}"


def load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    cache: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cache[row["key"]] = row["result"]
    return cache


def append_cache(path: Path, key: str, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"key": key, "result": result}) + "\n")


def parse_model_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def extract_crux_anthropic(
    post_a: Post,
    post_b: Post,
    author_a: str,
    author_b: str,
) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for --method anthropic")

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        author_a=author_a,
        source_a=post_a.source,
        date_a=post_a.date_published,
        text_a=truncate(post_a.text),
        author_b=author_b,
        source_b=post_b.source,
        date_b=post_b.date_published,
        text_b=truncate(post_b.text),
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    raw = response.content[0].text
    return parse_model_json(raw)


def authored_pair_key(author_a: str, author_b: str) -> str:
    """Stable key for an author pair, independent of order."""
    return "__".join(sorted([slugify(author_a), slugify(author_b)]))


def load_authored_cruxes(path: Path) -> dict[str, dict]:
    """Load human/LLM-authored cruxes baked into the repo (no API at runtime)."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("pairs", {})


def extract_crux(
    post_a: Post,
    post_b: Post,
    author_a: str,
    author_b: str,
    *,
    method: str,
    dry_run: bool,
    authored: dict[str, dict] | None = None,
) -> dict:
    if dry_run:
        return {
            "has_crux": False,
            "no_crux_reason": "dry-run",
            "crux_question": None,
            "type": None,
            "evidence_a": None,
            "evidence_b": None,
        }

    if method == "authored":
        entry = (authored or {}).get(authored_pair_key(author_a, author_b))
        if entry:
            return entry
        # No authored crux for this pair: drop the edge rather than emit a
        # low-quality heuristic guess.
        return {
            "has_crux": False,
            "no_crux_reason": "no authored crux for this pair",
            "crux_question": None,
            "type": None,
            "evidence_a": None,
            "evidence_b": None,
        }

    if method == "heuristic":
        return extract_heuristic_crux(
            truncate(post_a.text),
            truncate(post_b.text),
        )

    if method == "anthropic":
        return extract_crux_anthropic(post_a, post_b, author_a, author_b)

    raise ValueError(f"Unknown extraction method: {method}")


def build_graph(
    posts: list[Post],
    authors: list[str],
    grouped: dict[str, list[Post]],
    *,
    min_similarity: float,
    max_pairs: int,
    method: str,
    dry_run: bool,
    cache_path: Path,
    n_clusters: int = DEFAULT_CLUSTERS,
    authored_path: Path = AUTHORED_CRUXES_PATH,
) -> dict:
    coords = compute_pca_coords(authors, grouped)
    positions = {
        author: {"x": float(row[0]), "y": float(row[1]), "z": float(row[2])}
        for author, row in zip(authors, coords, strict=True)
    }
    clusters = compute_clusters(coords, authors, n_clusters)
    cache = load_cache(cache_path)
    authored = load_authored_cruxes(authored_path) if method == "authored" else {}

    nodes = []
    for author in authors:
        author_posts = grouped.get(author, [])
        sources = sorted({post.source for post in author_posts})
        pos = positions[author]
        nodes.append(
            {
                "id": slugify(author),
                "label": author,
                "post_count": len(author_posts),
                "sources": sources,
                "cluster": clusters[author],
                **pos,
            }
        )

    id_map = {author: slugify(author) for author in authors}
    candidate_pairs = find_candidate_pairs(authors, grouped, min_similarity, max_pairs)

    edges = []
    authored_used = 0
    for index, (author_a, author_b, topic_similarity) in enumerate(candidate_pairs):
        post_a, post_b, post_similarity = best_post_pair(author_a, author_b, grouped)
        is_authored = method == "authored" and authored_pair_key(author_a, author_b) in authored
        cache_key = pair_cache_key(author_a, author_b, post_a.id, post_b.id, method=method)

        if not is_authored and cache_key in cache:
            result = cache[cache_key]
            log(f"  cache hit: {author_a} vs {author_b}")
        else:
            log(f"  extracting crux ({'authored' if is_authored else method}): {author_a} vs {author_b}")
            result = extract_crux(
                post_a,
                post_b,
                author_a,
                author_b,
                method=method,
                dry_run=dry_run,
                authored=authored,
            )
            if not dry_run and not is_authored:
                append_cache(cache_path, cache_key, result)

        if not result.get("has_crux"):
            continue

        crux_type = result.get("type")
        if crux_type not in {"empirical", "values", "prediction"}:
            crux_type = "empirical"

        if is_authored:
            authored_used += 1

        edges.append(
            {
                "id": f"edge_{index}",
                "source": id_map[author_a],
                "target": id_map[author_b],
                "type": crux_type,
                "origin": "authored" if is_authored else method,
                "topic_similarity": round(topic_similarity, 4),
                "post_similarity": round(post_similarity, 4),
                "crux_question": result.get("crux_question"),
                "evidence_a": result.get("evidence_a"),
                "evidence_b": result.get("evidence_b"),
                "post_a": post_a.as_edge_post(author_a),
                "post_b": post_b.as_edge_post(author_b),
            }
        )

    return {
        "meta": {
            "generated_at": datetime.now(UTC).isoformat(),
            "dataset": REPO_ID,
            "splits": list(SPLITS),
            "method": None if dry_run else method,
            "model": None if dry_run or method != "anthropic" else MODEL,
            "post_count": len(posts),
            "author_count": len(authors),
            "candidate_pair_count": len(candidate_pairs),
            "edge_count": len(edges),
            "authored_edge_count": authored_used,
            "cluster_count": len(set(clusters.values())),
            "pca_components": 3,
        },
        "nodes": nodes,
        "edges": edges,
    }


MD_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
CSS_DECL = re.compile(r"\{[^{}]*\}")
CSS_SELECTOR = re.compile(r"[.#][A-Za-z][\w]*-[\w-]*")
MJX_TOKEN = re.compile(r"\b(?:mjx|MJX)[\w-]*")


def clean_passage(text: str) -> str:
    """Strip HTML tags, inline MathJax CSS, and markdown link syntax."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = CSS_DECL.sub(" ", text)
    text = CSS_SELECTOR.sub(" ", text)
    text = MJX_TOKEN.sub(" ", text)
    text = re.sub(r"@[A-Za-z-]+", " ", text)
    text = MD_LINK.sub(r"\1", text)
    text = re.sub(r"[*_`>#]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def export_pairs(
    authors: list[str],
    grouped: dict[str, list[Post]],
    *,
    min_similarity: float,
    max_pairs: int,
    passage_chars: int,
    output: Path,
) -> int:
    """Dump candidate author pairs + their best post pair for offline authoring."""
    candidate_pairs = find_candidate_pairs(authors, grouped, min_similarity, max_pairs)
    rows = []
    for author_a, author_b, topic_similarity in candidate_pairs:
        post_a, post_b, post_similarity = best_post_pair(author_a, author_b, grouped)
        rows.append(
            {
                "key": authored_pair_key(author_a, author_b),
                "author_a": author_a,
                "author_b": author_b,
                "topic_similarity": round(topic_similarity, 4),
                "post_similarity": round(post_similarity, 4),
                "post_a": {
                    "title": post_a.title,
                    "url": post_a.url,
                    "source": post_a.source,
                    "date": post_a.date_published,
                    "text": truncate(clean_passage(post_a.text), passage_chars),
                },
                "post_b": {
                    "title": post_b.title,
                    "url": post_b.url,
                    "source": post_b.source,
                    "date": post_b.date_published,
                    "text": truncate(clean_passage(post_b.text), passage_chars),
                },
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"Wrote {output} ({len(rows)} candidate pairs)")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cruxes.json for Crux Map")
    parser.add_argument("--max-posts", type=int, default=500)
    parser.add_argument("--top-authors", type=int, default=25)
    parser.add_argument("--min-similarity", type=float, default=0.08)
    parser.add_argument("--max-pairs", type=int, default=40)
    parser.add_argument("--clusters", type=int, default=DEFAULT_CLUSTERS)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "cruxes.json",
        help="Output graph JSON (default: ./cruxes.json)",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "data" / "processed" / "crux_cache.jsonl",
        help="Append-only cache of crux extraction results",
    )
    parser.add_argument(
        "--method",
        choices=("authored", "heuristic", "anthropic"),
        default="authored",
        help="Crux source (default: authored — baked-in, no API key needed)",
    )
    parser.add_argument(
        "--export-pairs",
        type=Path,
        default=None,
        help="Dump candidate post pairs to this JSON path and exit (for offline authoring)",
    )
    parser.add_argument(
        "--passage-chars",
        type=int,
        default=2_000,
        help="Max chars per passage when exporting pairs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build nodes + PCA positions without extracting cruxes",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log("Loading posts from HuggingFace...")
    posts, authors = parse_posts(args.max_posts, args.top_authors)
    grouped = posts_by_author(posts, authors)
    log(f"Loaded {len(posts)} posts across {len(authors)} authors")

    if args.export_pairs is not None:
        return export_pairs(
            authors,
            grouped,
            min_similarity=args.min_similarity,
            max_pairs=args.max_pairs,
            passage_chars=args.passage_chars,
            output=args.export_pairs,
        )

    log("Building graph...")
    graph = build_graph(
        posts,
        authors,
        grouped,
        min_similarity=args.min_similarity,
        max_pairs=args.max_pairs,
        method=args.method,
        dry_run=args.dry_run,
        cache_path=args.cache,
        n_clusters=args.clusters,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(graph, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(
        f"Wrote {args.output} ({graph['meta']['edge_count']} edges, "
        f"{graph['meta']['authored_edge_count']} authored, "
        f"{graph['meta']['author_count']} nodes, "
        f"{graph['meta']['cluster_count']} clusters)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
