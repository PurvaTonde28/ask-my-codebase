
from pathlib import Path
from ingestion.parse_git_log import build_commit_database

build_commit_database(Path("data/fastapi"), "data/commits.db")
print("Database built.")