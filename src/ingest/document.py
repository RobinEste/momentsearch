"""Stages shared by every document-shaped source (papers, decks).

Fetching a PDF and embedding its chunks are the same work whatever the document
is; only parsing differs, so only parsing lives in the per-kind modules. These
two tasks were written in paper.py first and moved here when deck.py became the
second caller — their `@task(name=...)` labels were already kind-neutral
("fetch-document", "embed-index-document"), which is the signal that this, not
paper.py, was always their home.

Nothing here imports a flow, so a per-kind module can import these without
pulling in a sibling kind.
"""
from __future__ import annotations

from prefect import task

from .. import config, db
from ..rag import vector_store
from ..rag.embeddings import embed_docs
from . import fetch as fetch_mod

_EMBED_BATCH = 64


@task(name="fetch-document", retries=2, retry_delay_seconds=[30, 120])
def t_fetch(doc_id: str, user_id: str, token: str) -> tuple[str, str | None]:
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

    Kind-neutral by construction: the duplicate check reads `kind` off the
    manifest row rather than assuming one, so identical bytes registered as a
    paper and as a deck stay two separate sources.
    """
    db.set_status(doc_id, "fetching", token=token)
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
    db.set_status(doc_id, "fetching", source_hash=source_hash, token=token)
    duplicate = db.find_duplicate(user_id, source_hash, exclude_id=doc_id,
                                  kind=row["kind"])
    if duplicate:
        path.unlink(missing_ok=True)
        db.set_status(doc_id, "skipped", error=f"duplicate of {duplicate['id']}",
                      token=token)
        return "", None
    return str(path), row.get("title")


@task(name="embed-index-document", retries=2, retry_delay_seconds=60)
def t_embed_index(doc_id: str, user_id: str, chunks: list[dict], token: str,
                  *, kind: str = "paper", ns: str = "page") -> int:
    """Batched bge embeddings -> idempotent upsert into the shared text
    collection. Deleting first drops points a previous, longer run left behind:
    deterministic ids overwrite, but they do not clean up orphans.

    The paper and deck flows differ in exactly two literals, the payload's
    `kind` and the point-id namespace, so those are arguments. The defaults
    reproduce the paper flow's original behaviour exactly — `ns` in particular
    feeds the deterministic point id, so a different default would strand every
    point already indexed under "page".
    """
    # Fenced BEFORE the purge below: a run that has lost the row must find out
    # here, not after it has deleted its successor's points.
    db.set_status(doc_id, "embedding", progress=0.0, units=len(chunks), token=token)
    vector_store.ensure_text_collection()
    vector_store.delete_video(user_id, doc_id)

    total = 0
    for start in range(0, len(chunks), _EMBED_BATCH):
        batch = chunks[start:start + _EMBED_BATCH]
        vectors = embed_docs([c["embed_text"] for c in batch])
        vector_store.upsert_chunks(
            user_id, doc_id, vectors,
            payloads=[{"user_id": user_id, "video_id": doc_id,
                       "modality": "text", "kind": kind,
                       "page": c["page"], "section": c["section"],
                       "text": c["text"],
                       "embed_version": config.TEXT_EMBED_VERSION}
                      for c in batch],
            ns=ns, start=start)
        total += len(batch)
        db.set_progress(doc_id, total / len(chunks), token=token)

    db.set_status(doc_id, "indexed", frame_count=total,
                  embed_version=config.TEXT_EMBED_VERSION, progress=1.0,
                  token=token)
    return total
