from pathlib import Path
import sqlite3
import subprocess

FIELD_SEPARATOR = "\x1f"
COMMIT_SEPARATOR = "\x1e"


def build_commit_database(
    repo_path: Path,
    db_path: str,
) -> None:

    git_command = [
        "git",
        "log",
        "--pretty=format:%x1e%H%x1f%an%x1f%ae%x1f%aI%x1f%s",
        "--numstat",
        # NOTE: no -m flag, so merge commits intentionally show zero file
        # changes here (git's default for merges, since a merge has two
        # parents and its "diff" is ambiguous). This is a deliberate scope
        # decision for v1 -- merge commits still appear in the `commits`
        # table with full metadata, just no associated file_changes rows.
    ]

    result = subprocess.run(
        git_command,
        cwd=repo_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )

    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    cursor.execute("DROP TABLE IF EXISTS file_changes")
    cursor.execute("DROP TABLE IF EXISTS commits")

    cursor.execute("""
    CREATE TABLE commits (
        hash TEXT PRIMARY KEY,
        author TEXT,
        email TEXT,
        date TEXT,
        message TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE file_changes (
        hash TEXT,
        file_path TEXT,
        insertions INTEGER,
        deletions INTEGER,
        FOREIGN KEY (hash) REFERENCES commits(hash)
    )
    """)

    cursor.execute("""
    CREATE INDEX idx_file_path
    ON file_changes(file_path)
    """)

    cursor.execute("""
    CREATE INDEX idx_commit_date
    ON commits(date)
    """)

    commit_blocks = result.stdout.split(COMMIT_SEPARATOR)

    for block in commit_blocks:

        block = block.strip()

        if not block:
            continue

        # Split header from file changes
        lines = block.splitlines()

        header = lines[0]

        numstat_lines = lines[1:]

        # Parse commit fields
        commit_hash, author, email, date, message = (
            header.split(FIELD_SEPARATOR)
        )

        # Insert commit
        cursor.execute(
            """
            INSERT INTO commits
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                commit_hash,
                author,
                email,
                date,
                message,
            ),
        )

        # Parse every changed file
        for line in numstat_lines:

            parts = line.split("\t")

            if len(parts) != 3:
                continue

            # Handle binary files
            insertions, deletions, file_path = parts

            insertions = (
                None
                if insertions == "-"
                else int(insertions)
            )

            deletions = (
                None
                if deletions == "-"
                else int(deletions)
            )

            # Insert file change
            cursor.execute(
                """
                INSERT INTO file_changes
                VALUES (?, ?, ?, ?)
                """,
                (
                    commit_hash,
                    file_path,
                    insertions,
                    deletions,
                ),
            )

    connection.commit()
    connection.close()