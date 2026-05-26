#!/usr/bin/env python3
"""Fail if the EN and TR locale JSON files diverge in key coverage.

EN is the source of truth (default locale). Every key present in en.json
must also be present in tr.json — even if the value is a placeholder.
The check runs in CI to catch translation gaps before they reach users.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EN_PATH = Path("web/frontend/src/i18n/en.json")
TR_PATH = Path("web/frontend/src/i18n/tr.json")


def flatten(d: dict, prefix: str = "") -> set[str]:
    """Return the set of dotted-path keys for nested dict structures."""
    keys: set[str] = set()
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            keys |= flatten(v, path)
        else:
            keys.add(path)
    return keys


def main() -> int:
    if not EN_PATH.exists() or not TR_PATH.exists():
        print(f"ERROR: missing locale file ({EN_PATH} / {TR_PATH})", file=sys.stderr)
        return 2

    en = json.loads(EN_PATH.read_text(encoding="utf-8"))
    tr = json.loads(TR_PATH.read_text(encoding="utf-8"))

    en_keys = flatten(en)
    tr_keys = flatten(tr)

    missing_in_tr = sorted(en_keys - tr_keys)
    extra_in_tr = sorted(tr_keys - en_keys)

    ok = True
    if missing_in_tr:
        ok = False
        print("[i18n] Missing in tr.json:", file=sys.stderr)
        for k in missing_in_tr:
            print(f"  - {k}", file=sys.stderr)
    if extra_in_tr:
        ok = False
        print("[i18n] Extra in tr.json (not present in en.json):", file=sys.stderr)
        for k in extra_in_tr:
            print(f"  - {k}", file=sys.stderr)

    if ok:
        print(f"[i18n] OK — {len(en_keys)} keys in parity")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
