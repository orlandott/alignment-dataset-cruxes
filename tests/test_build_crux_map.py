import numpy as np

from scripts.build_crux_map import (
    Post,
    choose_k,
    cluster_top_terms,
    compute_clusters,
    compute_post_coords,
    parse_model_json,
    post_summary,
    slugify,
    truncate,
)


def _posts():
    return [
        Post("1", "Risk", "https://x/posts/a1/risk", "lesswrong", "2020", ("Alice",),
             "AI alignment risk debate and existential safety concerns about agents"),
        Post("2", "Values", "https://x/posts/a2/values", "lesswrong", "2021", ("Alice",),
             "corrigibility and value learning for aligned agents and safety"),
        Post("3", "Timelines", "https://x/posts/b1/time", "alignmentforum", "2020", ("Bob",),
             "forecasting timelines and compute scaling predictions for the future"),
        Post("4", "Scaling", "https://x/posts/b2/scale", "alignmentforum", "2021", ("Bob",),
             "interpretability and scaling laws and compute forecasts over years"),
    ]


def test_slugify():
    assert slugify("Eliezer Yudkowsky") == "Eliezer_Yudkowsky"


def test_truncate():
    assert truncate("hello", 10) == "hello"
    assert truncate("abcdefghij", 8) == "abcde..."


def test_parse_model_json_strips_fences():
    raw = '```json\n{"has_crux": true}\n```'
    assert parse_model_json(raw) == {"has_crux": True}


def test_compute_post_coords_returns_xyz_per_post():
    coords, vectorizer, matrix = compute_post_coords(_posts())
    assert coords.shape == (4, 3)
    assert np.all(np.abs(coords) <= 1.0 + 1e-9)


def test_choose_k_within_bounds():
    coords = np.array(
        [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [5.0, 5.0, 5.0], [5.05, 5.0, 5.0]]
    )
    k = choose_k(coords, min_k=2, max_k=3)
    assert 2 <= k <= 3


def test_compute_clusters_groups_neighbors():
    coords = np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [5.0, 5.0, 5.0], [5.1, 5.0, 5.0]]
    )
    labels = compute_clusters(coords, 2)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_compute_clusters_handles_single_post():
    labels = compute_clusters(np.zeros((1, 3)), 5)
    assert list(labels) == [0]


def test_cluster_top_terms_labels_clusters():
    posts = _posts()
    _, vectorizer, matrix = compute_post_coords(posts)
    labels = np.array([0, 0, 1, 1])
    terms = cluster_top_terms(matrix, vectorizer, labels, top_n=3)
    assert set(terms) == {0, 1}
    assert all(isinstance(t, list) and t for t in terms.values())


def test_post_summary_returns_list():
    post = _posts()[0]
    summary = post_summary(post)
    assert isinstance(summary, list)
