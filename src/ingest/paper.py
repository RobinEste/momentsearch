"""Per-paper ingest pipeline — the document twin of pipeline.py.

pending -> fetching -> parsing -> embedding -> indexed | skipped | failed

Stages, mirroring the video flow rather than inventing a second shape:
  1. fetch   acquire the PDF into worker scratch (http download or bucket),
             hash the bytes, skip duplicates
  2. parse   pages -> cleaned text -> page-scoped chunks, token-limit verified
  3. embed   bge-embed the chunks -> idempotent Qdrant upsert into the SAME
             text collection the transcript branch uses

Three deliberate differences from the video flow, each with a reason:

  * There is no `chunking` status. Parsing and chunking together take
    milliseconds; a status that exists for 5ms is noise, not observability.
    `parsing` covers both, and `sampling` would have been a lie.
  * Parsing failure is fatal here. t_transcript in the video flow swallows every
    exception because a video without captions is still useful visually; a paper
    that yields no text is simply not ingested, and pretending otherwise would
    leave a source that claims to be indexed with nothing behind it.
  * Chunks carry a `page` in their payload and no timestamp. That is the whole
    point: the locator has to fit the source, and a paper has no 00:00.
"""
from __future__ import annotations

from pathlib import Path

from prefect import flow, task

from .. import config, db
from ..rag import vector_store
from ..rag.embeddings import count_tokens, embed_docs
from . import fetch as fetch_mod
from .chunking import chunk_text, context_line, verify_token_limit
from .pdf import parse_pdf

_EMBED_BATCH = 64


@task(name="fetch-document", retries=2, retry_delay_seconds=[30, 120])
def t_fetch(doc_id: str, user_id: str) -> tuple[str, str | None]:
    """Source PDF -> worker scratch file; duplicate check on the CONTENT hash.

    Returns (path, title). The title rides along because this task already has
    the manifest row, and a second read of it in the next task would cost
    another round trip to a database that is not on this machine.

    A path of "" means this is a duplicate of an already-indexed document for
    this user (row marked 'skipped' — a plain outcome, not a retryable error).

    Registration could only hash the URI, because POST /admin/documents must not
    touch the source. Here the bytes exist, so the row's source_hash is upgraded
    to the content hash: two different URLs serving the same paper now collapse
    into one indexed copy.
    """
    db.set_status(doc_id, "fetching")
    row = db.get_video(doc_id)
    if row is None:
        raise ValueError(f"no manifest row for {doc_id}")

    if row["storage_key"]:
        path = fetch_mod.fetch_upload(row["storage_key"], doc_id)
    elif row["url"]:
        path = fetch_mod.fetch_http(row["url"], doc_id,
                                    max_mb=config.MAX_DOCUMENT_MB)
    else:
        raise ValueError(f"{doc_id} has neither a url nor a storage key")

    source_hash = fetch_mod.sha256_file(path)
    db.set_status(doc_id, "fetching", source_hash=source_hash)
    duplicate = db.find_duplicate(user_id, source_hash, exclude_id=doc_id)
    if duplicate:
        path.unlink(missing_ok=True)
        db.set_status(doc_id, "skipped", error=f"duplicate of {duplicate['id']}")
        return "", None
    return str(path), row.get("title")


@task(name="parse-document")
def t_parse(doc_id: str, path: str, title: str | None) -> list[dict]:
    """PDF -> page-scoped chunks with their locator and heading path.

    A chunk never spans two pages: `page` is what a citation points at, and a
    chunk built across a page break makes that number false.
    """
    db.set_status(doc_id, "parsing", progress=0.0)
    pages = parse_pdf(Path(path).read_bytes())
    if not pages:
        raise RuntimeError("PDF has no pages.")

    # Chunk every page first, then verify the whole document in one go. Per page
    # this would be one HTTP round trip to the embedding service per page, and
    # that service also answers search queries — avoidable traffic there is
    # contention on exactly the path that has to stay decoupled from ingest.
    candidates = [(page, text) for page in pages for text in chunk_text(page.text)]
    verified = verify_token_limit([text for _, text in candidates], count_tokens)

    chunks: list[dict] = []
    for source, text in verified:
        page = candidates[source][0]
        chunks.append({"page": page.number, "section": page.section, "text": text,
                       # Contextual retrieval: the chunk alone often says "it
                       # improves recall by 12%" without saying what "it" is.
                       # Built from a template, never an LLM call — that would
                       # put the model in the ingest path.
                       "embed_text": (
                           f"{context_line(title, f'page {page.number}', page.section)}\n{text}"
                       ).strip()})

    empty = sum(1 for page in pages if not page.text.strip())
    print(f"[parse] {doc_id}: {len(pages)} pages -> {len(chunks)} chunks "
          f"({empty} pages without a text layer)")
    if not chunks:
        # Every page empty means a scanned PDF. Failing loudly beats an
        # 'indexed' row that answers nothing.
        raise RuntimeError("No text could be extracted — is this a scanned PDF?")
    return chunks


@task(name="embed-index-document", retries=2, retry_delay_seconds=60)
def t_embed_index(doc_id: str, user_id: str, chunks: list[dict]) -> int:
    """Batched bge embeddings -> idempotent upsert into the shared text
    collection. Deleting first drops points a previous, longer run left behind:
    deterministic ids overwrite, but they do not clean up orphans."""
    db.set_status(doc_id, "embedding", progress=0.0)
    vector_store.ensure_text_collection()
    vector_store.delete_video(user_id, doc_id)

    total = 0
    for start in range(0, len(chunks), _EMBED_BATCH):
        batch = chunks[start:start + _EMBED_BATCH]
        vectors = embed_docs([c["embed_text"] for c in batch])
        vector_store.upsert_chunks(
            user_id, doc_id, vectors,
            payloads=[{"user_id": user_id, "video_id": doc_id,
                       "modality": "text", "kind": "paper",
                       "page": c["page"], "section": c["section"],
                       "text": c["text"],
                       "embed_version": config.TEXT_EMBED_VERSION}
                      for c in batch],
            ns="page", start=start)
        total += len(batch)
        db.set_progress(doc_id, total / len(chunks))

    db.set_status(doc_id, "indexed", frame_count=total,
                  embed_version=config.TEXT_EMBED_VERSION, progress=1.0)
    return total


@flow(name="ms-ingest-paper", log_prints=True, timeout_seconds=3600)
def ingest_paper(video_id: str, user_id: str) -> dict:
    """Parameter is named video_id because the deployment contract is shared
    with the video flow (src/jobs.py schedules both the same way) and the
    manifest still calls its primary key that. Renaming both is queued as
    cleanup — see db.py's module docstring."""
    attempt = db.bump_attempts(video_id)
    path: str | None = None
    try:
        path, title = t_fetch(video_id, user_id)
        if not path:  # duplicate — already marked 'skipped' by t_fetch
            print(f"[ingest] {video_id} skipped (duplicate content)")
            return {"video_id": video_id, "skipped": True}
        chunks = t_parse(video_id, path, title)
        n = t_embed_index(video_id, user_id, chunks)
        pages = len({c["page"] for c in chunks})
        print(f"[ingest] {video_id} indexed: {n} chunks over {pages} pages "
              f"(attempt {attempt})")
        return {"video_id": video_id, "chunks": n, "pages": pages}
    except Exception as exc:
        db.set_status(video_id, "failed", error=f"{type(exc).__name__}: {exc}")
        raise  # Prefect marks the run Failed; full trace in the Cloud UI
    finally:
        if path:  # scratch only — durable copies live in object storage
            Path(path).unlink(missing_ok=True)
