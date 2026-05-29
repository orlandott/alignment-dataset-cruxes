# Crux Map

Visualize AI alignment research as a **3D map of posts**, clustered by topic, where each post carries a summary, its top comment, and — when that comment pushes back — the **double crux** between post and comment.

Data comes from the [StampyAI alignment research dataset](https://huggingface.co/datasets/StampyAI/alignment-research-dataset) (`lesswrong` + `alignmentforum` splits). A Python pipeline embeds each post with **dense, torch-free semantic vectors** ([model2vec](https://github.com/MinishLab/model2vec)) and reduces them with PCA: the **first 3 components** position posts on the map, while **k-means clusters on ~20 components** (so clusters reflect structure the 3D view drops). By default _k_ is **auto-selected by silhouette** (`--clusters N` to force a count). Dense embeddings pack ~5× more variance into 3 components than sparse TF-IDF (≈24% vs ≈5%) and group synonyms together; when the embedding model is unavailable (e.g. offline CI) the pipeline falls back to **TF-IDF → TruncatedSVD (LSA)**. Either way a TF-IDF view is kept purely for *labels*: each cluster is named by its most *distinctive* terms **plus the titles of its most representative posts** (the posts nearest the cluster centroid), and each axis is named by its distinctive term poles. The dataset only ships a `comment_count`, not comment text, so the **top comment** for each post is fetched from the public **LessWrong GraphQL API** (free, keyless — it serves Alignment Forum posts too).

In the 3D map:

- **Each point is a post**, positioned by its 3 principal components.
- **Color = topic cluster** (k-means).
- **Point size = how often the post is referenced by other posts** (LessWrong "pingbacks" — the count of other posts that link to it).
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
2. **Dense semantic embedding (model2vec) → PCA → L2-normalize.** Dense vectors put semantically similar posts near each other, so 3 components capture ~24% of variance (vs ~5% for TF-IDF). The map shows the first 3 components, but **clustering runs on ~20 components** (`CLUSTER_COMPONENTS`) — enough to capture structure the 3D view drops, while avoiding the distance concentration that washes out separation at 50D. Falls back to **TF-IDF → TruncatedSVD (LSA)** when the embedding model can't load. Axis labels are still derived from TF-IDF term poles, so each axis reads as words (e.g. _risks · safety ↔ mind · chatgpt_).
3. **k-means** into `--clusters` groups (default `0` = auto-select _k_ by the best silhouette score over a small range). Each cluster is then **matched to a human-readable theme** (e.g. _Mechanistic Interpretability_, _AI Risk & Policy_, _Corrigibility & Alignment Theory_, _Current AI Discourse_) by scoring its distinctive terms + exemplar titles against a small theme library; matching (rather than hardcoding cluster ids) means labels survive re-clustering, and each theme is used at most once. Domain-wide words (`ai`, `gpt`, `llm`, `model`) are treated as synonyms, not themes. Clusters also carry their distinctive `terms` and the titles of the posts nearest the centroid (`exemplars`) as supporting detail; a cluster that matches no theme falls back to its distinctive terms.
4. For each post: summarize its claim(s); fetch the highest-karma top-level comment from the LW GraphQL API (cached in `data/processed/comments_cache.jsonl`); detect whether the comment disagrees; and if so extract the **double crux** between the post and the comment.

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--max-posts` | 400 | Cap on posts (nodes) loaded |
| `--top-authors` | 40 | Restrict to posts by the top-N most prolific authors |
| `--clusters` | 0 | k-means clusters (`0` = auto-select _k_ by silhouette) |
| `--method` | `heuristic` | `heuristic` (keyless) or `anthropic` (needs API key) |
| `--offline` | off | Never hit the network for comments; use only the cache |
| `--dry-run` | off | Posts + PCA + clusters only (no comments/cruxes) |
| `--comment-cache` | `data/processed/comments_cache.jsonl` | Append-only cache of fetched top comments |
| `--output` | `./cruxes.json` | Graph output |

Output `cruxes.json` schema:

- **meta** — `post_count`, `cluster_count`, `k_selected`, `components` (3), `cluster_components` (dims used for k-means, ~20), `reduction` (`pca` for dense embeddings, else `truncated_svd`), `embedding_model`, `variance_explained` (fraction captured by the 3 components), `axis_labels` (per-axis `positive`/`negative` term poles + `variance_explained`), `comment_count`, `double_crux_count`, …
- **clusters** — `{ id, label, terms, exemplars, size }` per cluster (`label` = matched human theme, `terms` = distinctive words, `exemplars` = titles of the posts nearest the centroid).
- **nodes** — one per post: `title`, `author`, `source`, `url`, `date`, `karma`, `comment_count`, `referenced_by` (pingback count — how many other posts link to it), `cluster`, `x`/`y`/`z` (3-component coords), `summary` (claim list), and `top_comment`.
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
