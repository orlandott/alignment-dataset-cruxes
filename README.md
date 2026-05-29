# Crux Map

Visualize disagreements in AI alignment research as a **3D graph** of authors and philosophical cruxes.

Data comes from the [StampyAI alignment research dataset](https://huggingface.co/datasets/StampyAI/alignment-research-dataset) (`lesswrong` + `alignmentforum` splits). A Python pipeline finds author pairs with overlapping topics, attaches a crux between their most similar posts, positions authors in 3D via **PCA (3 components)** on TF-IDF profiles, and groups them into topic clusters via **k-means**.

In the 3D graph, **node color = topic cluster (k-means)** and **edge color = crux type** (empirical / values / prediction). Node size scales with post count.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No API key is required for the default workflow.

## Pipeline

Default run (free — uses the baked-in, hand-authored cruxes, no LLM costs):

```bash
python scripts/build_crux_map.py
# equivalent to:
python scripts/build_crux_map.py --method authored
```

Dry run (nodes + 3D PCA positions only, no edges):

```bash
python scripts/build_crux_map.py --dry-run
```

The pipeline also supports a free **heuristic** TF-IDF method and an optional **Anthropic** upgrade:

```bash
python scripts/build_crux_map.py --method heuristic
cp .env.example .env && python scripts/build_crux_map.py --method anthropic  # needs ANTHROPIC_API_KEY
```

### Authored cruxes (default)

The TF-IDF pair matcher reliably finds *topically* similar posts, but those aren't always real disagreements (and templated heuristic questions read as nonsense). So cruxes are authored once and committed to `data/authored_cruxes.json`, keyed by author pair. At build time the pipeline:

1. Recomputes candidate author pairs, 3D PCA positions, and k-means clusters from the live dataset.
2. For each pair, looks up an authored crux by author key. If one exists it is used; otherwise the edge is **dropped** (rather than emitting a low-quality guess).

This means the deployed/CI build never calls a paid API, yet the displayed cruxes are high quality.

To regenerate the candidate pairs for authoring (writes cleaned passages you can read and write cruxes against):

```bash
python scripts/build_crux_map.py --export-pairs data/processed/pairs.json
```

Then add/edit entries in `data/authored_cruxes.json` (each is `{ has_crux, crux_question, type, evidence_a, evidence_b }`) and rebuild.

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--method` | `authored` | `authored` (baked-in), `heuristic` (free TF-IDF), or `anthropic` (Claude, needs API key) |
| `--clusters` | 5 | k-means clusters for node coloring |
| `--max-posts` | 500 | Cap on posts loaded |
| `--top-authors` | 25 | Top authors by post count |
| `--min-similarity` | 0.08 | Min TF-IDF cosine for candidate pairs |
| `--max-pairs` | 40 | Max author pairs to analyze |
| `--export-pairs` | — | Dump candidate pairs + cleaned passages to a JSON path and exit |
| `--output` | `./cruxes.json` | Graph output |

Output `cruxes.json` schema:

- **nodes** — authors with `post_count`, `cluster` (k-means id), `x`/`y`/`z` (PCA), sized for the 3D graph
- **edges** — cruxes with `type` (`empirical` | `values` | `prediction`), `origin`, question, evidence quotes, and links to the two source posts

## Deploy (GitHub Pages)

The site deploys automatically on every push to `main`.

1. Create a GitHub repo (e.g. `alignment-dataset-cruxes`) and push this project.
2. In the repo: **Settings → Pages → Build and deployment → Source: GitHub Actions**.
3. Push to `main` — the **Deploy Crux Map to GitHub Pages** workflow publishes `index.html` + `cruxes.json`.

**Live site:** https://orlandott.github.io/alignment-dataset-cruxes/

The **Rebuild cruxes.json** workflow runs weekly (and on demand) using the free authored method — no API secret required. Run **Actions → Rebuild cruxes.json → Run workflow** to refresh node positions/clusters manually.

## Frontend (local preview)

```bash
python -m http.server 8080
# open http://localhost:8080
```

## Tests

```bash
pytest
```
