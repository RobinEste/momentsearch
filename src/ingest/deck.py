"""Per-deck ingest pipeline — the slide twin of paper.py.

pending -> fetching -> parsing -> embedding -> indexed | skipped | failed

A slide deck arrives as a PDF, so this shares the document flow's parser,
chunker, embedder and status vocabulary rather than inventing a second shape for
the same work. `t_fetch` and `t_embed_index` come from document.py; only parsing
differs, and it differs in exactly three ways, each deliberate:

  * The locator word. A slide IS a page in the file, and the payload keeps
    calling it `page` — that field is the structural signal the whole document
    branch keys on (`_fuse.same_window`, the document branch of `retrieve`,
    `isDoc()` in the UI). Introducing a third locator field would put those
    three places out of step the first time one of them was forgotten. What
    differs is the citation's locator — both its machine key (`slide` rather
    than `page`, which the grader reads) and its human label — and both are
    derived from `kind` at citation time, in one place (`_doc_locator`). What
    changes here is the context line prepended before
    embedding: "slide 7", not "page 7", because that string is read by the
    embedding model and a deck's own vocabulary retrieves better.

  * The chunk floor. Decks chunk at SLIDE_MIN_CHUNK_CHARS, not the paper
    MIN_CHUNK_CHARS; the argument is at that constant.

  * Slides with no text layer are counted and reported rather than passed over
    in silence. For a paper an empty page is a warning sign; for a deck it is an
    ordinary image-only slide, and the count is the measurement that decides
    whether the vision-caption route is worth building (see the deck entry in
    opleiding.md). A slide skipped here is skipped permanently: unlike a video
    frame it has no redundant neighbour and no second retrieval branch behind it.
"""
from __future__ import annotations

from pathlib import Path

from prefect import flow, task

from .. import db
from .chunking import SLIDE_MIN_CHUNK_CHARS, chunk_text, context_line, verify_token_limit
from ..rag.embeddings import count_tokens
from .document import t_embed_index, t_fetch
from .pdf import parse_pdf


@task(name="parse-deck")
def t_parse(doc_id: str, path: str, title: str | None) -> list[dict]:
    """PDF deck -> slide-scoped chunks with their locator.

    A chunk never spans two slides, for the same reason a paper chunk never
    spans two pages: the number is what a citation points at.
    """
    db.set_status(doc_id, "parsing", progress=0.0)
    slides = parse_pdf(Path(path).read_bytes())
    if not slides:
        raise RuntimeError("PDF has no pages.")

    # Chunk every slide first, then verify the whole deck in one call — one HTTP
    # round trip to the embedding service instead of one per slide, on the very
    # service that also answers search queries.
    candidates = [(slide, text)
                  for slide in slides
                  for text in chunk_text(slide.text, min_chars=SLIDE_MIN_CHUNK_CHARS)]
    verified = verify_token_limit([text for _, text in candidates], count_tokens)

    chunks: list[dict] = []
    for source, text in verified:
        slide = candidates[source][0]
        chunks.append({"page": slide.number, "section": slide.section, "text": text,
                       # Carries more weight here than for a paper: a slide's own
                       # text is often just a headline.
                       "embed_text": (
                           f"{context_line(title, f'slide {slide.number}', slide.section)}\n{text}"
                       ).strip()})

    # Two separate numbers on purpose: no-text-layer decides whether the vision
    # caption route is worth building, below-floor is a chunker calibration
    # question. One combined count would answer neither.
    covered = {c["page"] for c in chunks}
    imageless, dropped = 0, []
    for slide in slides:  # one pass: a blank slide is never covered, so this is exact
        if not slide.text.strip():
            imageless += 1
        elif slide.number not in covered:
            dropped.append(slide.number)
    print(f"[parse] {doc_id}: {len(slides)} slides -> {len(chunks)} chunks over "
          f"{len(covered)} slides ({imageless} with no text layer, "
          f"{len(dropped)} below the {SLIDE_MIN_CHUNK_CHARS}-char floor {dropped[:10]})")
    if not chunks:
        # Failing loudly beats an 'indexed' deck that answers nothing. Which
        # remedy to name depends on WHY there is nothing, and the two causes
        # need opposite work: no text layer at all is the vision-caption route,
        # while text that exists but did not survive means the parse dials are
        # wrong for this deck — the repeated-line stripper in pdf.py treats a
        # master-layout title as a running head, and what is left can fall under
        # the floor. Guessing "image-only" would send the reader to build the
        # wrong thing, so let the two counts we already have say it.
        raise RuntimeError(
            f"No chunks from {len(slides)} slides: {imageless} had no text layer, "
            f"{len(dropped)} had text that did not survive parsing "
            f"(repeated-line stripping and the {SLIDE_MIN_CHUNK_CHARS}-char floor). "
            + ("This deck is image-only — it needs the vision-caption route."
               if imageless == len(slides) else
               "Text was extracted but discarded; check the parse dials before "
               "reaching for captioning."))
    return chunks


@flow(name="ms-ingest-deck", log_prints=True, timeout_seconds=3600)
def ingest_deck(video_id: str, user_id: str) -> dict:
    """Parameter is named video_id because the deployment contract is shared
    with the video and paper flows (src/jobs.py schedules all three the same
    way) and the manifest still calls its primary key that. Renaming is queued
    as cleanup — see db.py's module docstring."""
    attempt = db.bump_attempts(video_id)
    path: str | None = None
    try:
        path, title = t_fetch(video_id, user_id)
        if not path:  # duplicate — already marked 'skipped' by t_fetch
            print(f"[ingest] {video_id} skipped (duplicate content)")
            return {"video_id": video_id, "skipped": True}
        chunks = t_parse(video_id, path, title)
        n = t_embed_index(video_id, user_id, chunks, kind="deck", ns="slide")
        slides = len({c["page"] for c in chunks})
        print(f"[ingest] {video_id} indexed: {n} chunks over {slides} slides "
              f"(attempt {attempt})")
        return {"video_id": video_id, "chunks": n, "slides": slides}
    except Exception as exc:
        db.set_status(video_id, "failed", error=f"{type(exc).__name__}: {exc}")
        raise  # Prefect marks the run Failed; full trace in the Cloud UI
    finally:
        if path:  # scratch only — durable copies live in object storage
            Path(path).unlink(missing_ok=True)
