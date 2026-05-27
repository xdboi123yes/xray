"""Check that code comments, docstrings, and UI string literals are English-only.

This script greps Python, HTML, and TypeScript source files for non-ASCII
letters (specifically Turkish-only characters) appearing in comment,
docstring, or string literal positions. Fails if any are found.

Allowed: variable names, identifiers.
Forbidden: comments, docstrings, log messages, UI string literals (outside tr.json).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

TURKISH_CHARS = set("çğıöşüÇĞİÖŞÜ")

# Match comment lines, docstrings, string literals, and HTML text nodes
COMMENT_PATTERNS = {
    ".py": [
        re.compile(r"#.*"),
        re.compile(r'""".*?"""', re.DOTALL),
        re.compile(r"'''.*?'''", re.DOTALL),
    ],
    ".ts": [
        re.compile(r"//.*"),
        re.compile(r"/\*.*?\*/", re.DOTALL),
        re.compile(r"'(.*?)'"),
        re.compile(r'"(.*?)"'),
        re.compile(r"`(.*?)`", re.DOTALL),
    ],
    ".tsx": [
        re.compile(r"//.*"),
        re.compile(r"/\*.*?\*/", re.DOTALL),
        re.compile(r"'(.*?)'"),
        re.compile(r'"(.*?)"'),
        re.compile(r"`(.*?)`", re.DOTALL),
    ],
    ".html": [
        re.compile(r"<!--.*?-->", re.DOTALL),
        re.compile(r">([^<]+)<"),
    ]
}

EXCLUDED_PATHS = [
    "src/",
    "notebooks/",
    "thesis/",
    "web/frontend/src/i18n/tr.json",
    "web/frontend/src/locale/",
    "web/frontend/dist/",
    "web/frontend/node_modules/",
]


def check_file(path: Path) -> list[str]:
    """Return list of violation messages for the given file."""
    # Check if the path starts with any of the excluded path prefixes
    normalized_path = str(path).replace("\\", "/")
    if any(ex in normalized_path for ex in EXCLUDED_PATHS):
        return []

    suffix = path.suffix
    if suffix not in COMMENT_PATTERNS:
        return []

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []

    violations = []
    for pattern in COMMENT_PATTERNS[suffix]:
        for match in pattern.finditer(content):
            text = match.group(0)
            if any(c in TURKISH_CHARS for c in text):
                line_no = content[: match.start()].count("\n") + 1
                snippet = text.replace("\n", " ").strip()
                violations.append(
                    f"{path}:{line_no}: Turkish character found: {snippet[:60]!r}"
                )
    return violations


def main(argv: list[str]) -> int:
    files: list[Path] = []
    if len(argv) > 1:
        # Files passed as arguments (e.g. from pre-commit)
        files = [Path(p) for p in argv[1:]]
    else:
        # Fallback recursive scan for manual check-lang execution
        target_dirs = ["core", "application", "infrastructure", "web"]
        for target in target_dirs:
            p = Path(target)
            if p.exists():
                if p.is_file():
                    files.append(p)
                else:
                    files.extend(p.glob("**/*.py"))
                    files.extend(p.glob("**/*.ts"))
                    files.extend(p.glob("**/*.tsx"))
                    files.extend(p.glob("**/*.html"))

    all_violations: list[str] = []
    for f in files:
        all_violations.extend(check_file(f))

    if all_violations:
        print("Language policy violations (Madde 1):", file=sys.stderr)
        for v in all_violations:
            print(f"  {v}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
