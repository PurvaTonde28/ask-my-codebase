from dataclasses import dataclass
from pathlib import Path
import os
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
ALLOWED_FILENAMES = {"README", "LICENSE", "Dockerfile", "Makefile"}


@dataclass
class FileRecord:
    path: Path
    extension: str
    content: str
    size: int

def walk_repository(
        repo_path: Path,
        extensions: set[str] | None = None
) -> list[FileRecord]:
    """
    Walk a repository recursively and return readable text files.
    Args:
        repo_path: Path to the cloned repository.
        extensions: Allowed file extensions.
    Returns:
        A list of FileRecord objects.
    """
    if extensions is None:
        extensions = {".py",
                      ".md",
                      ".txt",
                      ".rst",
                      ".json",
                      ".yaml",
                      ".yml",
                      ".toml",}

    ignored_dirs = {
        ".git",
        "__pycache__",
        "venv",
        ".venv",
        "node_modules",
        ".mypy_cache",
        "build",
        "dist",
        ".idea",
        ".vscode",
    }
    # empty list to store every file
    records: list[FileRecord] = []
    for root, dirs, files in os.walk(repo_path):
        # Don't even enter ignored directories
        dirs[:] = [d for d in dirs if d not in ignored_dirs]

        for file in files:
            file_path = Path(root) / file

            # Skip unsupported file types (but keep known extensionless files)
            if file_path.suffix.lower() not in extensions and file_path.name not in ALLOWED_FILENAMES:
                continue

            # Skip very large files (> 5 MB)
            file_size = file_path.stat().st_size
            if file_size > MAX_FILE_SIZE:
                continue

            try:
                content = file_path.read_text(
                    encoding="utf-8",
                    errors="replace"
                )
            except (OSError, UnicodeError):
                # Skip unreadable files
                continue

            records.append(
                FileRecord(
                    path=file_path.relative_to(repo_path),
                    extension=file_path.suffix.lower(),
                    content=content,
                    size=file_path.stat().st_size,
                )
            )
    return records