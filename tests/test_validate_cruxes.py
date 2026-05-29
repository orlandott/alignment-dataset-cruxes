from scripts.validate_cruxes import validate_crux


def test_validate_crux_accepts_minimal_valid(tmp_path):
    path = tmp_path / "ok.json"
    path.write_text(
        '{"id":"x","prompt":"p","tags":[],"expected_behavior":"e","notes":"n"}',
        encoding="utf-8",
    )
    assert validate_crux(path) == []


def test_validate_crux_reports_missing_fields(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"id":"x"}', encoding="utf-8")
    errors = validate_crux(path)
    assert any("missing required field" in e for e in errors)
