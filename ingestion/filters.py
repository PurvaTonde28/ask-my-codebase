"""Shared filters used when deciding which walked files to ingest."""

from pathlib import Path


def is_non_english_doc(path: Path) -> bool:
    """Skip translated docs (docs/<lang>/docs/...), keep docs/en/... and
    anything not under docs/ at all (like the root README)."""
    parts = path.parts
    if "docs" in parts:
        idx = parts.index("docs")
        if idx + 1 < len(parts):
            candidate = parts[idx + 1]
            if candidate != "en":
                return True
    return False