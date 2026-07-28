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

def get_class_header(source: str, node: ast.ClassDef) -> tuple[str, int]:
    """
    Returns just the class declaration -- decorators through the closing
    ':' -- WITHOUT the method/attribute bodies.

    Deliberately separate from get_node_source(): that function grabs the
    node's full line range, which for a ClassDef means every method body
    too. Using it for classes made every method's source text appear twice
    in the index (once inside the class chunk, once as its own method
    chunk) -- doubling embedding cost and giving the retriever near-
    duplicate results for the same query.
    """
    source_lines = source.splitlines()

    if node.decorator_list:
        start_line = node.decorator_list[0].lineno
    else:
        start_line = node.lineno

    if node.body:
        # max(..., node.lineno) matters for `class Foo: pass` -- when the
        # body starts on the SAME physical line as the class keyword,
        # body[0].lineno - 1 would land one line too early and drop the
        # "class Foo:" line itself from the header.
        header_end_line = max(node.body[0].lineno - 1, node.lineno)
    else:
        header_end_line = node.lineno

    header = "\n".join(source_lines[start_line - 1 : header_end_line]).rstrip()
    return header, start_line

def _format_params(args: ast.arguments) -> str:
    """
    Best-effort parameter list for a method summary line -- names only, no
    defaults/annotations. This is intentionally lossy: the exact signature
    already lives in that method's own chunk, so the summary just needs to
    be enough for a reader to recognize which method is which.
    """
    parts = [a.arg for a in getattr(args, "posonlyargs", [])]
    parts += [a.arg for a in args.args]
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    parts += [a.arg for a in args.kwonlyargs]
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")
    return ", ".join(parts)

def build_class_summary(
    header: str,
    docstring: str | None,
    method_signatures: list[str],
) -> str:
    """
    Summary-only content for a 'class' chunk: signature + docstring +
    method signatures (name/params only, no bodies). Replaces embedding the
    full class body, since every method body is already indexed separately.
    """
    parts = [header]

    if docstring:
        parts.append(f'    """{docstring}"""')

    if method_signatures:
        parts.append("")
        parts.append("    Methods:")
        parts.extend(f"      {sig}" for sig in method_signatures)

    return "\n".join(parts)

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
            class_header, class_start_line = get_class_header(source, node)
            class_docstring = ast.get_docstring(node)
            method_signatures: list[str] = []

            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_source, start_line = get_node_source(source, item)

                    is_async = isinstance(item, ast.AsyncFunctionDef)
                    method_type = "async_method" if is_async else "method"

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

                    prefix = "async def" if is_async else "def"
                    method_signatures.append(
                        f"{prefix} {item.name}({_format_params(item.args)})"
                    )

            # Summary chunk only: signature + docstring + method list, NOT
            # the full body. Every method's body already has its own chunk
            # above -- embedding the full class here as well used to
            # duplicate that same text in the index a second time.
            if class_header:
                class_content = build_class_summary(
                    class_header,
                    class_docstring,
                    method_signatures,
                )
                chunks.append(
                    CodeChunk(
                        file_path=record.path,
                        node_type="class",
                        name=node.name,
                        start_line=class_start_line,
                        end_line=node.end_lineno,
                        content=class_content,
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