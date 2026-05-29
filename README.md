# Crux Map

Visualize AI alignment research as a **3D map of posts**, clustered by topic, where each post carries a summary, its top comment, and — when that comment pushes back — the **double crux** between post and comment.

Data comes from the [StampyAI alignment research dataset](https://huggingface.co/datasets/StampyAI/alignment-research-dataset) (`lesswrong` + `alignmentforum` splits). A Python pipeline embeds each post (TF-IDF), reduces it to **3 principal components** (TruncatedSVD / LSA), and clusters the posts with **k-means**, where _k_ is auto-selected by silhouette score ("however many clusters make sense"). The dataset only ships a `comment_count`, not comment text, so the **top comment** for each post is fetched from the public **LessWrong GraphQL API** (free, keyless — it serves Alignment Forum posts too).

In the 3D map:

- **Each point is a post**, positioned by its 3 principal components.
- **Color = topic cluster** (k-means). Point size scales with comment count.
- There are **no edges** — posts are not linked to each other. The relationships shown are _within_ each post: post ↔ its top comment.

Click a post to see, in the side panel: what the post says, its top comment + that comment's claim, whether the comment disagrees, and the double crux (a falsifiable question, typed empirical / values / prediction, with a quote from each side).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No API key is required.

## Pipeline

Default run (free — heuristic summaries/cruxes + free GraphQL comments, both cached):

```bash
python scripts/build_crux_map.py
```

Use only the cached comments (no network), or skip comments entirely:

```bash
python scripts/build_crux_map.py --offline
python scripts/build_crux_map.py --dry-run   # posts + PCA + clusters only
```

Optional **Anthropic** upgrade for summaries/double cruxes (needs `ANTHROPIC_API_KEY`):

```bash
cp .env.example .env && python scripts/build_crux_map.py --method anthropic
```

### How it works

1. Load LW + AF posts, keep those with an author, date, and a post URL (needed to fetch comments). Cross-posts (same post on both forums) are collapsed to one.
2. TF-IDF → **TruncatedSVD to 3 components** → L2-normalize. Working on the angular (cosine) geometry of TF-IDF — rather than StandardScaler'd PCA, which just isolates rare-term outliers — is what yields balanced, topically meaningful clusters.
3. **k-means** with _k_ chosen by the best silhouette score over a small range (override with `--clusters N`). Each cluster is named by its top TF-IDF terms.
4. For each post: summarize its claim(s); fetch the highest-karma top-level comment from the LW GraphQL API (cached in `data/processed/comments_cache.jsonl`); detect whether the comment disagrees; and if so extract the **double crux** between the post and the comment.

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--max-posts` | 150 | Cap on posts (nodes) loaded |
| `--top-authors` | 40 | Restrict to posts by the top-N most prolific authors |
| `--clusters` | 0 | k-means clusters (`0` = auto-select by silhouette) |
| `--method` | `heuristic` | `heuristic` (keyless) or `anthropic` (needs API key) |
| `--offline` | off | Never hit the network for comments; use only the cache |
| `--dry-run` | off | Posts + PCA + clusters only (no comments/cruxes) |
| `--comment-cache` | `data/processed/comments_cache.jsonl` | Append-only cache of fetched top comments |
| `--output` | `./cruxes.json` | Graph output |

Output `cruxes.json` schema:

- **meta** — `post_count`, `cluster_count`, `k_selected`, `components` (3), `comment_count`, `double_crux_count`, …
- **clusters** — `{ id, terms, size }` per cluster (terms name the topic).
- **nodes** — one per post: `title`, `author`, `source`, `url`, `date`, `karma`, `comment_count`, `cluster`, `x`/`y`/`z` (3-component coords), `summary` (claim list), and `top_comment`.
  - **top_comment** — `{ author, score, text, claim, disagrees, crux }`. `crux` (present only when `disagrees`) is `{ has_crux, crux_question, type, evidence_post, evidence_comment }`.
- **edges** — always `[]` (posts are not linked).

### Post summaries, comment claims & double cruxes

Post summaries use authored claims in `data/authored_claims.json` (keyed by post id) when present, otherwise the heuristic summarizer (`scripts/post_claims.py`). Comment claims and disagreement detection live in `scripts/double_crux.py`.

Double cruxes resolve per post id in two tiers:

1. **Authored cruxes** in `data/authored_cruxes.json` (written by reading each post and its top comment) override the heuristic for that post. An entry with `has_crux: true` supplies a specific, falsifiable question + type + each side's position; `has_crux: false` records that the top comment isn't really a disagreement (a cue-heuristic false positive), so no crux is shown.
2. Otherwise the pipeline falls back to the TF-IDF contrast heuristic in `scripts/heuristic_crux.py`, so newly-surfaced posts still get a (rougher) crux with no API calls.

The authored entries cover the posts currently surfaced in `cruxes.json`, so every displayed double crux is hand-written.

## Deploy (GitHub Pages)

The site deploys automatically on every push to `main`.

1. Create a GitHub repo and push this project.
2. In the repo: **Settings → Pages → Build and deployment → Source: GitHub Actions**.
3. Push to `main` — the **Deploy Crux Map to GitHub Pages** workflow publishes `index.html` + `cruxes.json`.

**Live site:** https://orlandott.github.io/alignment-dataset-cruxes/

The **Rebuild cruxes.json** workflow runs weekly (and on demand), keyless: it recomputes PCA positions + clusters, refreshes top comments via the GraphQL API, and commits the updated `cruxes.json` and comment cache.

## Frontend (local preview)

```bash
python -m http.server 8080
# open http://localhost:8080
```

## Tests

```bash
pytest
```
