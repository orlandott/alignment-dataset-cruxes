# Crux Map

Visualize disagreements in AI alignment research as a **3D graph** of authors and philosophical cruxes.

Data comes from the [StampyAI alignment research dataset](https://huggingface.co/datasets/StampyAI/alignment-research-dataset) (`lesswrong` + `alignmentforum` splits). A Python pipeline finds author pairs with overlapping topics, extracts cruxes between their most similar posts, and positions authors in 3D via **PCA (3 components)** on TF-IDF profiles.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No API key is required for the default workflow.

## Pipeline

Default run (free — heuristic TF-IDF crux extraction, no LLM costs):

```bash
python scripts/build_crux_map.py
# equivalent to:
python scripts/build_crux_map.py --method heuristic
```

Dry run (nodes + 3D PCA positions only, no edges):

```bash
python scripts/build_crux_map.py --dry-run
```

Optional Anthropic upgrade (requires `ANTHROPIC_API_KEY`):

```bash
cp .env.example .env   # add ANTHROPIC_API_KEY
python scripts/build_crux_map.py --method anthropic
```

### Heuristic method (default)

For each candidate author pair, the pipeline:

1. Picks the most semantically similar post pair (TF-IDF cosine)
2. Computes term contrast — top TF-IDF terms enriched in each passage vs the other
3. Finds shared topic terms (TF-IDF overlap)
4. Builds a crux question from those terms (e.g. *"To what extent does {topic} require {term_a} versus {term_b}?"*)
5. Classifies type (`empirical`, `prediction`, `values`) from keyword rules
6. Extracts 1–2 evidence sentences per post containing the top contrast term

Edges are included only when term contrast exceeds a threshold (posts are meaningfully divergent).

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--method` | `heuristic` | `heuristic` (free) or `anthropic` (Claude, needs API key) |
| `--max-posts` | 500 | Cap on posts loaded |
| `--top-authors` | 25 | Top authors by post count |
| `--min-similarity` | 0.08 | Min TF-IDF cosine for candidate pairs |
| `--max-pairs` | 40 | Max author pairs to analyze |
| `--output` | `./cruxes.json` | Graph output |
| `--cache` | `data/processed/crux_cache.jsonl` | Cached extraction results |

Output `cruxes.json` schema:

- **nodes** — authors with `post_count`, `x`/`y`/`z` (PCA), sized for the 3D graph
- **edges** — cruxes with `type` (`empirical` | `values` | `prediction`), question, evidence quotes, and links to source posts

## Deploy (GitHub Pages)

The site deploys automatically on every push to `main`.

1. Create a GitHub repo (e.g. `alignment-dataset-cruxes`) and push this project.
2. In the repo: **Settings → Pages → Build and deployment → Source: GitHub Actions**.
3. Push to `main` — the **Deploy Crux Map to GitHub Pages** workflow publishes `index.html` + `cruxes.json`.

**Live site:** https://orlandott.github.io/alignment-dataset-cruxes/

The **Rebuild cruxes.json** workflow runs weekly (and on demand) using the free heuristic method — no API secret required. Run **Actions → Rebuild cruxes.json → Run workflow** to refresh data manually.

## Frontend (local preview)

```bash
python -m http.server 8080
# open http://localhost:8080
```

## Tests

```bash
pytest
```
