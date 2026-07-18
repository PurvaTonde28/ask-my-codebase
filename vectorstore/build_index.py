from dataclasses import dataclass
from pathlib import Path
import hashlib
import pickle
import time

import faiss
import numpy as np
from fastembed import TextEmbedding

from splitters.python_code_splitter import CodeChunk
from splitters.text_structure_splitter import TextChunk


@dataclass
class EmbeddableChunk:
    text: str
    metadata: dict


def code_chunk_to_embeddable(chunk: CodeChunk) -> EmbeddableChunk:
    return EmbeddableChunk(
        text=f"{chunk.name}\n{chunk.content}",
        metadata={
            "type": "code",
            "file_path": str(chunk.file_path),
            "node_type": chunk.node_type,
            "name": chunk.name,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "content": chunk.content,
        },
    )


def text_chunk_to_embeddable(chunk: TextChunk) -> EmbeddableChunk:
    return EmbeddableChunk(
        text=f"{chunk.header_path}\n{chunk.content}",
        metadata={
            "type": "text",
            "file_path": str(chunk.file_path),
            "header_path": chunk.header_path,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "content": chunk.content,
        },
    )


MAX_EMBED_CHARS = 3000


def _cap_text(text: str, max_chars: int = MAX_EMBED_CHARS) -> str:
    return text[:max_chars]


def _chunk_id(metadata: dict) -> str:
    """
    Stable content-based ID for a chunk, used as the checkpoint key.
    Built from file path + line range + type, NOT just the text -- two
    different files can have identical short content (e.g. both just
    "import os"), which would wrongly collide if we hashed text alone.
    """
    kind = metadata.get("node_type") or metadata.get("header_path") or ""
    key = f"{metadata['file_path']}|{kind}|{metadata.get('start_line')}|{metadata.get('end_line')}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _checkpoint_path(output_path: Path) -> Path:
    return output_path / "checkpoint.pkl"


def _load_checkpoint(output_path: Path) -> dict:
    """Returns {chunk_id: (vector, metadata)} from a previous partial run, or {} if none exists."""
    ckpt_path = _checkpoint_path(output_path)
    if ckpt_path.exists():
        with open(ckpt_path, "rb") as f:
            return pickle.load(f)
    return {}


def _save_checkpoint(output_path: Path, checkpoint: dict) -> None:
    """
    Atomic save: write to a temp file, then rename over the real checkpoint.
    If the process dies mid-write, the rename never happens and the old
    checkpoint stays intact -- you never end up with a half-written,
    corrupted checkpoint file.
    """
    ckpt_path = _checkpoint_path(output_path)
    tmp_path = ckpt_path.with_suffix(".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(checkpoint, f)
    tmp_path.replace(ckpt_path)


_embedding_model: TextEmbedding | None = None
_index: faiss.Index | None = None
_metadata: list[dict] | None = None


def _get_embedding_model() -> TextEmbedding:
    global _embedding_model
    if _embedding_model is None:
        print("Loading embedding model (first time only)...", flush=True)
        _embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", threads=4)
        print("Model loaded.", flush=True)
    return _embedding_model


def build_index(
    code_chunks: list[CodeChunk],
    text_chunks: list[TextChunk],
    output_dir: str | Path = "vectorstore",
    cooldown_every: int = 1000,
    cooldown_seconds: float = 5.0,
    checkpoint_every: int = 500,
    resume: bool = True,
) -> None:
    """
    Embed all code + text chunks, build a FAISS index, persist index + metadata.

    Resumable: progress is checkpointed to output_dir/checkpoint.pkl every
    `checkpoint_every` chunks. If a previous run was interrupted, calling
    this again with the same chunks skips everything already embedded and
    only processes what's missing. Set resume=False to force a full re-embed.
    """
    global _index, _metadata

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("build_index() started", flush=True)

    embeddable_chunks: list[EmbeddableChunk] = []
    for chunk in code_chunks:
        embeddable_chunks.append(code_chunk_to_embeddable(chunk))
    for chunk in text_chunks:
        embeddable_chunks.append(text_chunk_to_embeddable(chunk))

    if not embeddable_chunks:
        raise ValueError("No chunks given to build_index() — nothing to embed.")

    print(f"Prepared {len(embeddable_chunks)} embeddable chunks", flush=True)

    checkpoint = _load_checkpoint(output_path) if resume else {}
    if checkpoint:
        print(f"Resuming: {len(checkpoint)} chunks already embedded from a previous run", flush=True)

    to_embed_chunks = []
    to_embed_ids = []
    for c in embeddable_chunks:
        cid = _chunk_id(c.metadata)
        if cid not in checkpoint:
            to_embed_chunks.append(c)
            to_embed_ids.append(cid)

    print(f"{len(to_embed_chunks)} chunks remaining to embed (of {len(embeddable_chunks)} total)", flush=True)

    if to_embed_chunks:
        model = _get_embedding_model()
        texts = [_cap_text(c.text) for c in to_embed_chunks]

        since_last_checkpoint = 0
        for i, vec in enumerate(model.embed(texts, batch_size=16)):
            cid = to_embed_ids[i]
            checkpoint[cid] = (vec, to_embed_chunks[i].metadata)
            since_last_checkpoint += 1

            if (i + 1) % 500 == 0:
                print(f"  embedded {i + 1}/{len(texts)}", flush=True)

            if since_last_checkpoint >= checkpoint_every:
                _save_checkpoint(output_path, checkpoint)
                since_last_checkpoint = 0
                print(f"  checkpoint saved ({len(checkpoint)} total chunks embedded so far)", flush=True)

            if cooldown_seconds > 0 and (i + 1) % cooldown_every == 0:
                _save_checkpoint(output_path, checkpoint)
                since_last_checkpoint = 0
                print(f"  cooling down {cooldown_seconds}s...", flush=True)
                time.sleep(cooldown_seconds)

        _save_checkpoint(output_path, checkpoint)
        print("Embedding loop finished.", flush=True)
    else:
        print("Nothing left to embed, all chunks already in checkpoint.", flush=True)

    print("Building FAISS index from checkpoint...", flush=True)
    vectors = []
    metadata = []
    for c in embeddable_chunks:
        cid = _chunk_id(c.metadata)
        vec, meta = checkpoint[cid]
        vectors.append(vec)
        metadata.append(meta)

    vectors = np.array(vectors, dtype=np.float32)
    faiss.normalize_L2(vectors)
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    index_path = output_path / "index.faiss"
    metadata_path = output_path / "metadata.pkl"

    faiss.write_index(index, str(index_path))
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    _index = index
    _metadata = metadata

    print(f"DONE. Indexed {len(embeddable_chunks)} chunks -> {index_path}, {metadata_path}", flush=True)


def load_index(output_dir: str | Path = "vectorstore") -> None:
    global _index, _metadata
    output_path = Path(output_dir)
    index_path = output_path / "index.faiss"
    metadata_path = output_path / "metadata.pkl"

    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"No index found at {output_path}. Run build_index() first.")

    _index = faiss.read_index(str(index_path))
    with open(metadata_path, "rb") as f:
        _metadata = pickle.load(f)


def search(query: str, k: int = 5) -> list[dict]:
    if _index is None or _metadata is None:
        raise RuntimeError("No index loaded. Call build_index() or load_index() first.")

    model = _get_embedding_model()
    query_vector = np.array(list(model.embed([query])), dtype=np.float32)
    faiss.normalize_L2(query_vector)

    distances, indices = _index.search(query_vector, k)

    results = []
    for idx in indices[0]:
        if idx != -1:
            results.append(_metadata[idx])
    return results