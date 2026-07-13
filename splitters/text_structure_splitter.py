# splitters/text_structure_splitter.py

import re
from dataclasses import dataclass
from pathlib import Path

from ingestion.read_files import FileRecord

HEADER_REGEX = re.compile(r"^(#{1,6})\s+(.+)$")


@dataclass
class TextChunk:
    file_path: Path
    header_path: str
    start_line: int
    end_line: int
    content: str


def flush_chunk(
    chunks: list[TextChunk],
    record: FileRecord,
    header_stack: list[tuple[int, str]],
    chunk_lines: list[str],
    start_line: int,
    end_line: int,
) -> None:
    """
    Create a TextChunk from the buffered lines and append it to chunks.
    """

    content = "\n".join(chunk_lines).strip()

    if not content:
        return

    header_path = " > ".join(title for _, title in header_stack)

    chunks.append(
        TextChunk(
            file_path=record.path,
            header_path=header_path,
            start_line=start_line,
            end_line=end_line,
            content=content,
        )
    )


def split_markdown_file(record: FileRecord) -> list[TextChunk]:
    """
    Split a Markdown/text file into structural chunks.

    - Respects Markdown headers
    - Preserves fenced code blocks
    - Tracks nested header hierarchy
    """

    chunks: list[TextChunk] = []

    # Simple fallback for .txt and .rst
    if record.extension in {".txt", ".rst"}:
        content = record.content.strip()

        if content:
            chunks.append(
                TextChunk(
                    file_path=record.path,
                    header_path="",
                    start_line=1,
                    end_line=len(record.content.splitlines()),
                    content=content,
                )
            )

        return chunks

    # 1. Read file line by line
    lines = record.content.splitlines()

    header_stack: list[tuple[int, str]] = []

    current_chunk_lines: list[str] = []
    current_start_line = 1

    inside_code_block = False

    # Read file line by line
    for line_number, line in enumerate(lines, start=1):

        # Toggle fenced code block
        if line.strip().startswith("```"):
            inside_code_block = not inside_code_block
            current_chunk_lines.append(line)
            continue

        # Everything inside a code block belongs to the current chunk
        if inside_code_block:
            current_chunk_lines.append(line)
            continue

        header_match = HEADER_REGEX.match(line)

        if header_match:

            # Save previous section
            flush_chunk(
                chunks,
                record,
                header_stack,
                current_chunk_lines,
                current_start_line,
                line_number - 1,
            )

            current_chunk_lines = []

            level = len(header_match.group(1))
            title = header_match.group(2).strip()

            while header_stack and header_stack[-1][0] >= level:
                header_stack.pop()

            header_stack.append((level, title))

            current_start_line = line_number
            current_chunk_lines.append(line)

        else:
            current_chunk_lines.append(line)

    # Flush last section
    flush_chunk(
        chunks,
        record,
        header_stack,
        current_chunk_lines,
        current_start_line,
        len(lines),
    )

    return chunks