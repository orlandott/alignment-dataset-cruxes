from scripts.build_crux_map import (
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
