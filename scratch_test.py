from pathlib import Path
import time

from ingestion.clone_repo import clone_repository
from ingestion.read_files import walk_repository
from splitters.python_code_splitter import split_python_file
from splitters.text_structure_splitter import split_markdown_file
from vectorstore.build_index import build_index, search


def is_non_english_doc(path: Path) -> bool:
    """fastapi's docs are structured as docs/<lang>/docs/... -- skip
    translations, keep docs/en/... and anything not under docs/ at all
    (like the root README)."""
    parts = path.parts
    if "docs" in parts:
        idx = parts.index("docs")
        if idx + 1 < len(parts):
            candidate = parts[idx + 1]
            if candidate != "en":
                return True
    return False


repo_path = clone_repository("https://github.com/tiangolo/fastapi")
print("Cloned to:", repo_path, flush=True)

records = walk_repository(repo_path)
py_records = [r for r in records if r.extension == ".py"]
md_records = [r for r in records if r.extension in (".md", ".rst", ".txt")]

md_records = [r for r in md_records if not is_non_english_doc(r.path)]

code_chunks = []
for r in py_records:
    code_chunks.extend(split_python_file(r))

text_chunks = []
for r in md_records:
    text_chunks.extend(split_markdown_file(r))

print(f"{len(code_chunks)} code chunks, {len(text_chunks)} text chunks (English docs only)", flush=True)

start = time.time()
build_index(
    code_chunks,
    text_chunks,
    output_dir="vectorstore/data",
    cooldown_every=1000,
    cooldown_seconds=5.0,
)
elapsed = time.time() - start
print(f"\nTotal build_index time: {elapsed / 60:.1f} minutes", flush=True)

print("\n--- Search test ---", flush=True)
results = search("how does dependency injection work", k=5)
for r in results:
    label = r.get("name") or r.get("header_path")
    print(f"[{r['type']}] {label} | {r['file_path']}")