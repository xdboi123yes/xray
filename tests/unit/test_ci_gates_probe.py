"""Unit tests acting as automated probes for CI quality gates.

Verifies that comment language checks correctly detect and flag violations when
non-English text is introduced in source comments or docstrings.
"""

from __future__ import annotations

from pathlib import Path

from scripts.check_comment_language import check_file


def test_comment_language_checker_probe(tmp_path: Path) -> None:
    """Verify that the comment language checker flags Turkish characters in comments."""
    clean_py = tmp_path / "clean.py"
    clean_py.write_text("# This is a clean comment in English.", encoding="utf-8")

    dirty_py = tmp_path / "dirty.py"
    dirty_py.write_text("# Bu yorumda Türkçe karakterler var: ş, ç, ı.", encoding="utf-8")

    violations_clean = check_file(clean_py)
    violations_dirty = check_file(dirty_py)

    assert len(violations_clean) == 0, f"Expected 0 violations, got {violations_clean}"
    assert len(violations_dirty) > 0, "Expected violations for Turkish comments, but got none"
    assert "Turkish character found" in violations_dirty[0]
