#!/usr/bin/env python3
"""Validate crux JSON files under cruxes/."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REQUIRED_FIELDS = ("id", "prompt", "tags", "expected_behavior", "notes")


def validate_crux(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{path.name}: invalid JSON ({exc})"]

    if not isinstance(data, dict):
        return [f"{path.name}: root must be an object"]

    for field in REQUIRED_FIELDS:
        if field not in data:
            errors.append(f"{path.name}: missing required field '{field}'")

    if "tags" in data and not isinstance(data["tags"], list):
        errors.append(f"{path.name}: 'tags' must be a list")

    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[1] / "cruxes"
    files = sorted(p for p in root.glob("*.json") if p.name != "example.json")
    if not files:
        print("No crux files to validate (excluding example.json).")
        return 0

    all_errors: list[str] = []
    for path in files:
        all_errors.extend(validate_crux(path))

    if all_errors:
        print("Validation failed:")
        for err in all_errors:
            print(f"  - {err}")
        return 1

    print(f"Validated {len(files)} crux file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
