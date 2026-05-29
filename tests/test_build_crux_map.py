import numpy as np

from scripts.build_crux_map import (
    authored_pair_key,
    clean_passage,
    compute_clusters,
    compute_pca_coords,
    compute_pca_positions,
    parse_model_json,
    pair_cache_key,
    posts_by_author,
    slugify,
    truncate,
)
from scripts.build_crux_map import Post


def test_slugify():
    assert slugify("Eliezer Yudkowsky") == "Eliezer_Yudkowsky"


def test_truncate():
    assert truncate("hello", 10) == "hello"
    assert truncate("abcdefghij", 8) == "abcde..."


def test_parse_model_json_strips_fences():
    raw = '```json\n{"has_crux": true}\n```'
    assert parse_model_json(raw) == {"has_crux": True}


def test_pair_cache_key_is_stable():
    key_a = pair_cache_key("Alice", "Bob", "p1", "p2")
    key_b = pair_cache_key("Bob", "Alice", "p2", "p1")
    assert key_a == key_b


def test_compute_pca_positions_returns_xyz():
    posts = [
        Post("1", "t", "", "lesswrong", "2020", ("Alice",), "AI alignment risk debate"),
        Post("2", "t", "", "lesswrong", "2021", ("Alice",), "corrigibility and value learning"),
        Post("3", "t", "", "alignmentforum", "2020", ("Bob",), "forecasting timelines and compute"),
        Post("4", "t", "", "alignmentforum", "2021", ("Bob",), "interpretability and scaling laws"),
    ]
    grouped = posts_by_author(posts, ["Alice", "Bob"])
    positions = compute_pca_positions(["Alice", "Bob"], grouped)
    assert set(positions["Alice"]) == {"x", "y", "z"}
    assert all(isinstance(positions["Alice"][axis], float) for axis in "xyz")


def test_authored_pair_key_is_order_independent():
    assert authored_pair_key("Scott Garrabrant", "Diffractor") == authored_pair_key(
        "Diffractor", "Scott Garrabrant"
    )
    assert authored_pair_key("A B", "C") == "A_B__C"


def test_clean_passage_strips_markup_and_css():
    raw = "Hello <b>world</b> .mjx-chtml {display: inline-block; padding: 1px 0} [link](http://x) end"
    cleaned = clean_passage(raw)
    assert "mjx" not in cleaned
    assert "{" not in cleaned
    assert "http" not in cleaned
    assert "Hello" in cleaned and "world" in cleaned and "link" in cleaned


def test_compute_clusters_assigns_labels():
    authors = ["A", "B", "C", "D"]
    coords = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [5.0, 5.0, 5.0], [5.1, 5.0, 5.0]])
    clusters = compute_clusters(coords, authors, n_clusters=2)
    assert set(clusters) == set(authors)
    assert clusters["A"] == clusters["B"]
    assert clusters["C"] == clusters["D"]
    assert clusters["A"] != clusters["C"]


def test_compute_clusters_handles_single_author():
    coords = compute_pca_coords(["solo"], {"solo": []}) if False else np.zeros((1, 3))
    clusters = compute_clusters(coords, ["solo"], n_clusters=5)
    assert clusters == {"solo": 0}
