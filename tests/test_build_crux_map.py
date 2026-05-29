import numpy as np

import scripts.build_crux_map as bcm
from scripts.build_crux_map import (
    Post,
    build_comment_block,
    choose_k,
    cluster_top_terms,
    compute_clusters,
    compute_post_coords,
    describe_pca_axes,
    parse_model_json,
    post_summary,
    slugify,
    truncate,
)
from scripts.comments import TopComment


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
    geo = compute_post_coords(_posts(), use_embeddings=False)
    assert geo.coords.shape == (4, 3)
    assert np.all(np.abs(geo.coords) <= 1.0 + 1e-9)
    assert geo.reduction == "truncated_svd"
    assert geo.embedding_model is None
    # Clustering features are separate from the 3D display coords (here the
    # tiny corpus only supports 3 components, but they are distinct arrays).
    assert geo.cluster_features.shape[0] == 4
    assert geo.cluster_features.shape[1] >= 1


def test_describe_pca_axes_labels_components():
    geo = compute_post_coords(_posts(), use_embeddings=False)
    axes = describe_pca_axes(geo.scores, geo.vectorizer, geo.matrix, geo.variance)
    assert 1 <= len(axes) <= len(geo.variance)
    for entry in axes:
        assert entry["axis"] in ("x", "y", "z")
        assert entry["positive"]
        assert entry["negative"]
        assert 0 <= entry["variance_explained"] < 1


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
    geo = compute_post_coords(_posts(), use_embeddings=False)
    labels = np.array([0, 0, 1, 1])
    terms = cluster_top_terms(geo.matrix, geo.vectorizer, labels, top_n=3)
    assert set(terms) == {0, 1}
    assert all(isinstance(t, list) and t for t in terms.values())


def test_post_summary_returns_list():
    post = _posts()[0]
    summary = post_summary(post)
    assert isinstance(summary, list)


def test_authored_crux_overrides_heuristic(monkeypatch):
    post = _posts()[0]
    comment = TopComment("c1", "Critic", 12, "I disagree, the problem with this is X.")
    monkeypatch.setattr(
        bcm,
        "_authored_cruxes_cache",
        {
            post.id: {
                "has_crux": True,
                "crux_question": "Authored specific question?",
                "type": "values",
                "evidence_post": "post side",
                "evidence_comment": "comment side",
            }
        },
    )
    block = build_comment_block(post, comment, method="heuristic")
    assert block["disagrees"] is True
    assert block["crux"]["crux_question"] == "Authored specific question?"
    assert block["crux"]["type"] == "values"


def test_authored_crux_can_suppress_false_positive(monkeypatch):
    post = _posts()[0]
    comment = TopComment("c1", "Fan", 5, "I disagree but actually this is a great post, however.")
    monkeypatch.setattr(
        bcm,
        "_authored_cruxes_cache",
        {post.id: {"has_crux": False, "no_crux_reason": "actually agreement"}},
    )
    block = build_comment_block(post, comment, method="heuristic")
    assert block["disagrees"] is False
    assert block["crux"] is None
