# Crux Map

Visualize disagreements in AI alignment research as a **3D graph** of authors and philosophical cruxes.

Data comes from the [StampyAI alignment research dataset](https://huggingface.co/datasets/StampyAI/alignment-research-dataset) (`lesswrong` + `alignmentforum` splits). A one-time Python pipeline extracts author pairs, calls Claude to identify cruxes, and positions authors in 3D via **PCA (3 components)** on TF-IDF profiles.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add ANTHROPIC_API_KEY
```

## Pipeline

Dry run (no API — builds nodes + 3D PCA positions only):

```bash
python scripts/build_crux_map.py --dry-run
```

Full run:

```bash
python scripts/build_crux_map.py
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--max-posts` | 500 | Cap on posts loaded |
| `--top-authors` | 25 | Top authors by post count |
| `--min-similarity` | 0.08 | Min TF-IDF cosine for candidate pairs |
| `--max-pairs` | 40 | Max author pairs sent to Claude |
| `--output` | `./cruxes.json` | Graph output |
| `--cache` | `data/processed/crux_cache.jsonl` | Cached API responses |

Output `cruxes.json` schema:

- **nodes** — authors with `post_count`, `x`/`y`/`z` (PCA), sized for the 3D graph
- **edges** — cruxes with `type` (`empirical` | `values` | `prediction`), question, evidence quotes, and links to source posts

## Deploy (GitHub Pages)

The site deploys automatically on every push to `main`.

1. Create a GitHub repo (e.g. `alignment-dataset-cruxes`) and push this project.
2. In the repo: **Settings → Pages → Build and deployment → Source: GitHub Actions**.
3. Push to `main` — the **Deploy Crux Map to GitHub Pages** workflow publishes `index.html` + `cruxes.json`.

**Live site:** https://orlandott.github.io/alignment-dataset-cruxes/

To refresh crux data in CI, add an `ANTHROPIC_API_KEY` repo secret, then run **Actions → Rebuild cruxes.json → Run workflow**. That commits an updated `cruxes.json` and triggers a redeploy.

## Frontend (local preview)

```bash
python -m http.server 8080
# open http://localhost:8080
```
