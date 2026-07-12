# splitters/python_code_splitter.py

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ingestion.read_files import FileRecord

@dataclass
class CodeChunk:
    file_path: Path
    node_type: str
    name: str
    start_line: int
    end_line: int
    content: str

def get_node_source(source: str, node: Any) -> tuple[str, int]:
    """
    Returns:
        content: Source code including decorators (if any)
        start_line: Actual starting line in the file
    """
    source_lines = source.splitlines()

    if getattr(node, "decorator_list", None):
        start_line = node.decorator_list[0].lineno
    else:
        start_line = node.lineno

    end_line = node.end_lineno

    content = "\n".join(source_lines[start_line - 1 : end_line])
    return content, start_line

def split_python_file(record: FileRecord) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []

    try:
        tree = ast.parse(record.content)
    except SyntaxError as e:
        print(f"[Python Splitter] Skipping {record.path}: {e}")
        return []

    source = record.content

    # step1: Module docstring
    module_docstring = ast.get_docstring(tree)

    if module_docstring:
        first_node = tree.body[0]
        chunks.append(
            CodeChunk(
                file_path=record.path,
                node_type="module_docstring",
                name="<module>",
                start_line=first_node.lineno,
                end_line=first_node.end_lineno,
                content=module_docstring,
            )
        )

    imports = []
    import_start_line = None
    import_end_line = None

    other_top_level = []
    other_start_line = None
    other_end_line = None

    for node in tree.body:

        # step2: Collect Imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if import_start_line is None:
                import_start_line = node.lineno
            import_end_line = node.end_lineno
            source_segment = ast.get_source_segment(source, node)

            if source_segment:
                imports.append(source_segment)

            continue

        # step 3: functions (normal + async)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            source_segment, start_line = get_node_source(source, node)

            node_type = (
                "async_function"
                if isinstance(node, ast.AsyncFunctionDef)
                else "function"
            )

            chunks.append(
                CodeChunk(
                    file_path=record.path,
                    node_type=node_type,
                    name=node.name,
                    start_line=start_line,
                    end_line=node.end_lineno,
                    content=source_segment,
                )
            )
            continue

        # step 4: Classes + Methods
        if isinstance(node, ast.ClassDef):
            class_source, start_line = get_node_source(source, node)

            if class_source:
                chunks.append(
                    CodeChunk(
                        file_path=record.path,
                        node_type="class",
                        name=node.name,
                        start_line=start_line,
                        end_line=node.end_lineno,
                        content=class_source,
                    )
                )

            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_source, start_line = get_node_source(source, item)

                    method_type = (
                        "async_method"
                        if isinstance(item, ast.AsyncFunctionDef)
                        else "method"
                    )

                    chunks.append(
                        CodeChunk(
                            file_path=record.path,
                            node_type=method_type,
                            name=f"{node.name}.{item.name}",
                            start_line=start_line,
                            end_line=item.end_lineno,
                            content=method_source,
                        )
                    )
            continue

        # step 5: anything else at module level (constants, bare code, etc.)
        # skip the docstring node itself, it's already handled in step 1
        if module_docstring and node is tree.body[0]:
            continue

        source_segment = ast.get_source_segment(source, node)
        if source_segment:
            if other_start_line is None:
                other_start_line = node.lineno
            other_end_line = node.end_lineno
            other_top_level.append(source_segment)

    if imports:
        chunks.append(
            CodeChunk(
                file_path=record.path,
                node_type="imports",
                name="<imports>",
                start_line=import_start_line,
                end_line=import_end_line,
                content="\n".join(imports),
            )
        )

    if other_top_level:
        chunks.append(
            CodeChunk(
                file_path=record.path,
                node_type="module_level_code",
                name="<module_level_code>",
                start_line=other_start_line,
                end_line=other_end_line,
                content="\n".join(other_top_level),
            )
        )

    return chunks