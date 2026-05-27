"""Unit tests acting as automated probes for CI quality gates.

Verifies that comment language and translation key parity checks correctly detect and
flag violations when non-English text or key disparities are introduced.
"""

from __future__ import annotations

from pathlib import Path

from scripts.check_comment_language import check_file
from scripts.check_i18n_parity import flatten


def test_comment_language_checker_probe(tmp_path: Path) -> None:
    """Verify that the comment language checker flags Turkish characters in comments."""
    # Create a mock compliant Python file
    clean_py = tmp_path / "clean.py"
    clean_py.write_text("# This is a clean comment in English.", encoding="utf-8")
    
    # Create a mock non-compliant Python file with Turkish characters in a comment
    dirty_py = tmp_path / "dirty.py"
    dirty_py.write_text("# Bu yorumda Türkçe karakterler var: ş, ç, ı.", encoding="utf-8")
    
    # Run check_file
    violations_clean = check_file(clean_py)
    violations_dirty = check_file(dirty_py)
    
    assert len(violations_clean) == 0, f"Expected 0 violations, got {violations_clean}"
    assert len(violations_dirty) > 0, "Expected violations for Turkish comments, but got none"
    assert "Turkish character found" in violations_dirty[0]


def test_comment_language_checker_ts_probe(tmp_path: Path) -> None:
    """Verify that comment language checker flags Turkish characters in TypeScript files."""
    clean_ts = tmp_path / "clean.ts"
    clean_ts.write_text("// Decoupled presentation sub-components\nconst x = 5;", encoding="utf-8")
    
    dirty_ts = tmp_path / "dirty.ts"
    dirty_ts.write_text("/* Bu dosya Türkçe açıklama içeriyor */\nconst y = 10;", encoding="utf-8")
    
    violations_clean = check_file(clean_ts)
    violations_dirty = check_file(dirty_ts)
    
    assert len(violations_clean) == 0
    assert len(violations_dirty) > 0
    assert "Turkish character found" in violations_dirty[0]


def test_i18n_parity_flatten_and_diff() -> None:
    """Verify that the dotted-path flattening and key parity comparison logic works."""
    en_data = {
        "title": "Welcome",
        "nested": {
            "key1": "Value 1",
            "key2": "Value 2"
        }
    }
    
    tr_data_perfect = {
        "title": "Hoşgeldiniz",
        "nested": {
            "key1": "Değer 1",
            "key2": "Değer 2"
        }
    }
    
    tr_data_divergent = {
        "title": "Hoşgeldiniz",
        "nested": {
            "key1": "Değer 1"
            # Missing nested.key2
        },
        "extra_key": "Extra"
    }
    
    en_keys = flatten(en_data)
    tr_keys_perfect = flatten(tr_data_perfect)
    tr_keys_divergent = flatten(tr_data_divergent)
    
    assert en_keys == {"title", "nested.key1", "nested.key2"}
    assert tr_keys_perfect == {"title", "nested.key1", "nested.key2"}
    assert tr_keys_divergent == {"title", "nested.key1", "extra_key"}
    
    # Calculate differences exactly like check_i18n_parity.py does
    missing_in_tr = en_keys - tr_keys_divergent
    extra_in_tr = tr_keys_divergent - en_keys
    
    assert missing_in_tr == {"nested.key2"}
    assert extra_in_tr == {"extra_key"}
