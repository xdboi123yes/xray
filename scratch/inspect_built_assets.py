import re
from pathlib import Path

TURKISH_CHARS = re.compile("[çğıöşüÇĞİÖŞÜ]")
dist_dir = Path("/Users/alperen/Desktop/xray/web/frontend/dist")

for file_path in dist_dir.glob("**/*"):
    if file_path.is_file() and not file_path.name.startswith("tr"):
        try:
            content = file_path.read_text(encoding="utf-8")
            matches = list(TURKISH_CHARS.finditer(content))
            if matches:
                print(f"File: {file_path.relative_to(dist_dir)}")
                print(f"Total Matches: {len(matches)}")
                for m in matches[:10]:
                    start = max(0, m.start() - 30)
                    end = min(len(content), m.end() + 30)
                    snippet = content[start:end].replace('\n', ' ')
                    print(f"  Match: {m.group(0)!r} at position {m.start()} in context: {snippet!r}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
