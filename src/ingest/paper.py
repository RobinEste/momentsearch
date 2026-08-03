"""Per-paper ingest pipeline — the document twin of pipeline.py.

pending -> fetching -> parsing -> embedding -> indexed | skipped | failed

Stages, mirroring the video flow rather than inventing a second shape:
  1. fetch   acquire the PDF into worker scratch (http download or bucket),
             hash the bytes, skip duplicates          — shared, see document.py
  2. parse   pages -> cleaned text -> page-scoped chunks, token-limit verified
  3. embed   bge-embed the chunks -> idempotent Qdrant upsert into the SAME
             text collection the transcript branch uses — shared, see document.py

Only parsing is paper-specific, so only parsing lives here.

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
from ..rag.embeddings import count_tokens
from .chunking import chunk_text, context_line, verify_token_limit
from .document import t_embed_index, t_fetch
from .run import ingest_run
from .pdf import parse_pdf


@task(name="parse-document")
def t_parse(doc_id: str, path: str, title: str | None, token: str) -> list[dict]:
    """PDF -> page-scoped chunks with their locator and heading path.

    A chunk never spans two pages: `page` is what a citation points at, and a
    chunk built across a page break makes that number false.
    """
    db.set_status(doc_id, "parsing", progress=0.0, token=token)
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


@flow(name="ms-ingest-paper", log_prints=True, timeout_seconds=config.FLOW_TIMEOUT_S)
def ingest_paper(video_id: str, user_id: str) -> dict:
    """Parameter is named video_id because the deployment contract is shared
    with the video flow (src/jobs.py schedules both the same way) and the
    manifest still calls its primary key that. Renaming both is queued as
    cleanup — see db/manifest.py's module docstring."""
    with ingest_run(video_id) as run:
        path, title = t_fetch(video_id, user_id, run.token)
        if not path:  # duplicate — already marked 'skipped' by t_fetch
            print(f"[ingest] {video_id} skipped (duplicate content)")
            return {"video_id": video_id, "skipped": True}
        run.scratch = path
        chunks = t_parse(video_id, path, title, run.token)
        n = t_embed_index(video_id, user_id, chunks, run.token)
        pages = len({c["page"] for c in chunks})
        print(f"[ingest] {video_id} indexed: {n} chunks over {pages} pages "
              f"(attempt {run.attempt})")
        return {"video_id": video_id, "chunks": n, "pages": pages}
