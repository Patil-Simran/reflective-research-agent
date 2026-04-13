"""Chroma persistence + ingestion."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma  # type: ignore[no-redef,assignment]

from reflective_research.config.settings import Settings

log = logging.getLogger(__name__)

_TEXT_SUFFIXES = {".txt", ".md", ".mdx"}


def get_vectorstore(settings: Settings, embeddings: Embeddings) -> Chroma:
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_persist_dir),
    )


def _load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix in _TEXT_SUFFIXES:
        text = path.read_text(encoding="utf-8", errors="replace")
        return [Document(page_content=text, metadata={"source": str(path.resolve())})]
    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
        return loader.load()
    log.debug("Skipping unsupported file type: %s", path)
    return []


def ingest_paths(
    settings: Settings,
    embeddings: Embeddings,
    paths: list[Path],
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> int:
    """Load files, split, upsert into Chroma. Returns number of chunks added."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    all_docs: list[Document] = []
    for raw in paths:
        p = raw.expanduser().resolve()
        if p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file():
                    all_docs.extend(_load_file(child))
        elif p.is_file():
            all_docs.extend(_load_file(p))

    if not all_docs:
        log.warning("No documents loaded from paths: %s", paths)
        return 0

    splits = splitter.split_documents(all_docs)
    vs = get_vectorstore(settings, embeddings)
    # Stable ids from content hash + source + start index
    ids: list[str] = []
    for doc in splits:
        src = doc.metadata.get("source", "")
        start = doc.metadata.get("start_index", 0)
        key = f"{src}|{start}|{doc.page_content[:80]}"
        ids.append(hashlib.sha256(key.encode()).hexdigest()[:32])
    vs.add_documents(splits, ids=ids)
    try:
        vs.persist()
    except Exception:
        # Newer Chroma may auto-persist
        pass
    log.info("Ingested %s chunks into collection %s", len(splits), settings.chroma_collection)
    return len(splits)


def similarity_search(
    settings: Settings,
    embeddings: Embeddings,
    query: str,
    k: int = 4,
) -> list[Document]:
    vs = get_vectorstore(settings, embeddings)
    try:
        return vs.similarity_search(query, k=k)
    except Exception as e:
        log.exception("RAG search failed: %s", e)
        return []
