import numpy as np

import scripts.build_crux_map as bcm
from scripts.build_crux_map import (
    Post,
    assign_cluster_themes,
    build_comment_block,
    choose_k,
    cluster_exemplars,
    cluster_label,
    cluster_top_terms,
    compute_clusters,
    compute_post_coords,
    compute_subclusters,
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


def test_cluster_top_terms_filters_filler_words():
    posts = [
        Post("1", "A", "u1", "lesswrong", "2020", ("X",),
             "we don t really know things actually maybe interpretability circuits"),
        Post("2", "B", "u2", "lesswrong", "2020", ("X",),
             "we don t really know things actually maybe governance policy regulation"),
    ]
    geo = compute_post_coords(posts, use_embeddings=False)
    terms = cluster_top_terms(geo.matrix, geo.vectorizer, np.array([0, 1]), top_n=4)
    for picked in terms.values():
        assert "don" not in picked
        assert "things" not in picked
        assert "know" not in picked


def test_assign_cluster_themes_matches_and_is_unique():
    match_terms = {
        0: ["interpretability", "activation", "circuit", "neuron"],
        1: ["risks", "catastrophic", "governance", "policy"],
        2: ["utility", "optimisation", "coherence", "consequentialism"],
    }
    exemplars = {
        0: ["Mechanistic circuits in a transformer"],
        1: ["An overview of catastrophic AI risks"],
        2: ["Strong coherence and expected utility"],
    }
    themes = assign_cluster_themes(match_terms, exemplars)
    assert themes[0] == "Mechanistic Interpretability"
    assert themes[1] == "AI Risk & Policy"
    assert themes[2] == "Agent Foundations"
    # Each theme is used at most once.
    assert len(set(themes.values())) == len(themes)


def test_assign_cluster_themes_leaves_unmatched_unassigned():
    themes = assign_cluster_themes(
        {0: ["banana", "umbrella", "weather"]}, {0: ["A post about nothing relevant"]}
    )
    assert 0 not in themes


def test_cluster_label_falls_back_to_terms():
    assert cluster_label("AI Risk & Policy", ["risks", "safety"]) == "AI Risk & Policy"
    assert cluster_label(None, ["risks", "safety", "ais", "control"]) == "risks · safety · ais"
    assert cluster_label(None, []) == "Misc"


def test_compute_subclusters_within_parents():
    posts = _posts() * 8  # 32 posts, duplicated text blocks still split in subclusters
    geo = compute_post_coords(posts, use_embeddings=False)
    labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    sub_labels, meta = compute_subclusters(
        geo.cluster_features, labels, geo.matrix, geo.vectorizer, posts
    )
    assert sub_labels.shape == (32,)
    assert 0 in meta and 1 in meta
    assert len(meta[0]) >= 1
    assert sum(s["size"] for s in meta[0]) == 16


def test_cluster_exemplars_returns_titles_nearest_centroid():
    posts = _posts()
    geo = compute_post_coords(posts, use_embeddings=False)
    labels = np.array([0, 0, 1, 1])
    exemplars = cluster_exemplars(geo.cluster_features, labels, posts, top_n=2)
    assert set(exemplars) == {0, 1}
    assert all(len(v) <= 2 and all(isinstance(t, str) for t in v) for v in exemplars.values())
    # Exemplars must be real post titles.
    titles = {p.title for p in posts}
    assert all(t in titles for v in exemplars.values() for t in v)


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
