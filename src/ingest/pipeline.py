"""Per-video ingest pipeline — a Prefect flow of three stage-tasks.

pending -> fetching -> sampling -> embedding -> indexed | skipped | failed

Stages:
  1. fetch    acquire the source into worker scratch (bucket download for
              uploads, yt-dlp for YouTube), hash it, skip duplicates
  2. sample   ffmpeg pipe-to-memory keyframes -> pHash dedup -> thumbnails
              batch-uploaded to object storage
  3. embed    CLIP-embed the surviving frames (batched) -> idempotent Qdrant
              upsert (deterministic IDs, user_id-tagged)

Orchestration: Prefect Cloud. The API triggers a deployment run (src/jobs.py);
worker.py serves this flow and picks runs up. Each task carries its own retry
policy — a completed stage is not re-run when a later one fails and retries.

Postgres remains the business-status source of truth: tasks update the videos
row; Prefect Cloud's UI is the operational view (logs, retries, run history).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from pathlib import Path

from prefect import flow, task

from .. import config, db, storage
from ..config import CLIP_BATCH, EMBED_VERSION
from ..rag import vector_store
from ..rag.embeddings import embed_jpegs
from . import fetch as fetch_mod
from .dedup import dedup
from .frames import Frame, sample
from .retry import retry_unless_orphaned
from .run import ingest_run

_UPLOAD_POOL = 8  # concurrent thumbnail PUTs (I/O-bound)


@task(name="fetch", retries=2, retry_delay_seconds=[30, 120],
      retry_condition_fn=retry_unless_orphaned)
def t_fetch(video_id: str, user_id: str, token: str) -> str:
    """Source video -> worker scratch file; duplicate check via source_hash.

    Returns "" when the content is a duplicate of an already-indexed video for
    this user (row marked 'skipped' — a plain outcome, not a retryable error).
    """
    db.set_status(video_id, "fetching", token=token)
    row = db.get_video(video_id)
    if row is None:
        raise ValueError(f"no manifest row for {video_id}")

    if row["source"] == "youtube":
        path, title = fetch_mod.fetch_youtube(row["url"], video_id)
        source_hash = video_id  # the YouTube id IS the content identity
        db.set_status(video_id, "fetching", title=title, source_hash=source_hash,
                      token=token)
    else:
        path = fetch_mod.fetch_upload(row["storage_key"], video_id)
        source_hash = fetch_mod.sha256_file(path)
        db.set_status(video_id, "fetching", source_hash=source_hash, token=token)

    dup = db.find_duplicate(user_id, source_hash, exclude_id=video_id, kind="video")
    if dup:
        path.unlink(missing_ok=True)
        db.set_status(video_id, "skipped", error=f"duplicate of {dup['id']}",
                      token=token)
        return ""
    return str(path)


@task(name="sample")
def t_sample(video_id: str, user_id: str, path: str, token: str) -> list[Frame]:
    """Keyframes in memory -> pHash dedup -> thumbnails to object storage."""
    # No `units`: this phase's cost is the ffmpeg decode of the whole file, and
    # the frame count says nothing about that (config.PHASE_BUDGETS_S explains).
    # Note also that nothing ticks until the uploads below start, so the stall
    # grace does not cover the decode either — the flat budget is all there is.
    db.set_status(video_id, "sampling", progress=0.0, token=token)
    frames = sample(Path(path))
    if not frames:
        raise RuntimeError("No frames could be extracted from the video.")
    kept = dedup(frames)
    print(f"[sample] {video_id}: {len(frames)} sampled -> {len(kept)} after dedup")

    # Idempotent re-run: clear any thumbnails a previous attempt left behind.
    storage.delete_prefix(storage.frame_prefix(user_id, video_id))
    done = 0
    counted = Lock()  # `done += 1` from _UPLOAD_POOL threads is a read-modify-write

    def _put(i_f: tuple[int, Frame]) -> None:
        nonlocal done
        i, f = i_f
        storage.put_bytes(storage.frame_key(user_id, video_id, i), f.jpeg, "image/jpeg")
        # Without the lock, lost updates let the counter skip 25 outright — and
        # this branch is the only place the upload phase checks it still owns the
        # row, so missing it means the whole phase runs unchecked.
        with counted:
            done += 1
            tick = done % 25 == 0
        if tick:
            db.set_progress(video_id, done / len(kept), token=token)

    with ThreadPoolExecutor(max_workers=_UPLOAD_POOL) as ex:
        list(ex.map(_put, enumerate(kept)))
    db.set_progress(video_id, 1.0, token=token)
    return kept


@task(name="embed-index", retries=2, retry_delay_seconds=60,
      retry_condition_fn=retry_unless_orphaned)
def t_embed_index(video_id: str, user_id: str, frames: list[Frame],
                  token: str) -> int:
    """Batched CLIP embeddings -> idempotent multi-tenant Qdrant upsert."""
    # Fenced BEFORE the purge below: a run that has lost the row must find out
    # here, not after it has deleted its successor's points.
    db.set_status(video_id, "embedding", progress=0.0, units=len(frames), token=token)
    vector_store.ensure_collection()
    vector_store.delete_video(user_id, video_id)  # drop stale points from prior runs

    total = 0
    for start in range(0, len(frames), CLIP_BATCH):
        batch = frames[start:start + CLIP_BATCH]
        vectors = embed_jpegs([f.jpeg for f in batch])
        vector_store.upsert_frames(
            user_id, video_id,
            ids=range(start, start + len(batch)),
            vectors=vectors,
            payloads=[{"user_id": user_id, "video_id": video_id, "ms": f.ms,
                       "idx": start + i, "modality": "frame",
                       "t_start": f.ms / 1000.0, "t_end": f.ms / 1000.0,
                       "embed_version": EMBED_VERSION}
                      for i, f in enumerate(batch)],
        )
        total += len(batch)
        db.set_progress(video_id, total / len(frames), token=token)
    return total


@task(name="transcript", retries=1, retry_delay_seconds=30,
      retry_condition_fn=retry_unless_orphaned)
def t_transcript(video_id: str, user_id: str) -> int:
    """YouTube captions -> time chunks -> bge -> text collection (the 2nd
    branch). Best-effort: uploads have no captions, some videos have none, and
    any failure just leaves the video visual-only — never fails the flow.
    Runs AFTER embed-index (whose delete clears both branches first)."""
    from ..config import ENABLE_TRANSCRIPT, TEXT_EMBED_VERSION
    from ..rag.embeddings import embed_docs
    from .transcript import chunk_cues, fetch_transcript

    if not ENABLE_TRANSCRIPT:
        return 0
    row = db.get_video(video_id) or {}
    if row.get("source") != "youtube" or not row.get("url"):
        return 0  # uploaded files have no caption track
    try:
        chunks = chunk_cues(fetch_transcript(row["url"], video_id))
        if not chunks:
            print(f"[transcript] {video_id}: no captions — visual-only")
            return 0
        vector_store.ensure_text_collection()
        vecs = embed_docs([c["text"] for c in chunks])
        vector_store.upsert_chunks(user_id, video_id, vecs, payloads=[
            {"user_id": user_id, "video_id": video_id, "modality": "text",
             "t_start": c["t_start"], "t_end": c["t_end"],
             "ms": int(c["t_start"] * 1000), "text": c["text"],
             "embed_version": TEXT_EMBED_VERSION} for c in chunks])
        print(f"[transcript] {video_id}: indexed {len(chunks)} transcript chunks")
        return len(chunks)
    except Exception as exc:
        print(f"[transcript] {video_id}: failed ({type(exc).__name__}: {exc}) — visual-only")
        return 0


@flow(name="ms-ingest-video", log_prints=True, timeout_seconds=config.FLOW_TIMEOUT_S)
def ingest_video(video_id: str, user_id: str) -> dict:
    with ingest_run(video_id) as run:
        path = t_fetch(video_id, user_id, run.token)
        if not path:  # duplicate — already marked 'skipped' by t_fetch
            print(f"[ingest] {video_id} skipped (duplicate content)")
            return {"video_id": video_id, "skipped": True}
        run.scratch = path
        frames = t_sample(video_id, user_id, path, run.token)
        n = t_embed_index(video_id, user_id, frames, run.token)
        # Transcript branch AFTER frames (embed-index's delete clears both first).
        # Its own phase, and not a courtesy: the row stays in flight through it,
        # so without one it would be running on whatever deadline the last
        # progress tick of the PREVIOUS phase happened to leave behind.
        db.set_status(video_id, "transcribing", token=run.token)
        t = t_transcript(video_id, user_id)
        # 'indexed' is written HERE and not at the end of t_embed_index: a
        # terminal status releases the row's token, and releasing it while this
        # flow is still working hands a live source to /retry and to the next
        # registration. The transcript branch would then be writing into a
        # successor's namespace, and this flow's scratch cleanup would take the
        # successor's download.
        db.set_status(video_id, "indexed", frame_count=n,
                      embed_version=EMBED_VERSION, progress=1.0, token=run.token)
        print(f"[ingest] {video_id} indexed: {n} frames + {t} transcript chunks "
              f"(attempt {run.attempt})")
        return {"video_id": video_id, "frames": n, "transcript_chunks": t}
