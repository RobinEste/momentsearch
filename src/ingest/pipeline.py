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
from pathlib import Path

from prefect import flow, task

from .. import db, storage
from ..config import CLIP_BATCH, EMBED_VERSION
from ..rag import vector_store
from ..rag.embeddings import embed_jpegs
from . import fetch as fetch_mod
from .dedup import dedup
from .frames import Frame, sample

_UPLOAD_POOL = 8  # concurrent thumbnail PUTs (I/O-bound)


@task(name="fetch", retries=2, retry_delay_seconds=[30, 120])
def t_fetch(video_id: str, user_id: str) -> str:
    """Source video -> worker scratch file; duplicate check via source_hash.

    Returns "" when the content is a duplicate of an already-indexed video for
    this user (row marked 'skipped' — a plain outcome, not a retryable error).
    """
    db.set_status(video_id, "fetching")
    row = db.get_video(video_id)
    if row is None:
        raise ValueError(f"no manifest row for {video_id}")

    if row["source"] == "youtube":
        path, title = fetch_mod.fetch_youtube(row["url"], video_id)
        source_hash = video_id  # the YouTube id IS the content identity
        db.set_status(video_id, "fetching", title=title, source_hash=source_hash)
    else:
        path = fetch_mod.fetch_upload(row["storage_key"], video_id)
        source_hash = fetch_mod.sha256_file(path)
        db.set_status(video_id, "fetching", source_hash=source_hash)

    dup = db.find_duplicate(user_id, source_hash, exclude_id=video_id)
    if dup:
        path.unlink(missing_ok=True)
        db.set_status(video_id, "skipped", error=f"duplicate of {dup['id']}")
        return ""
    return str(path)


@task(name="sample")
def t_sample(video_id: str, user_id: str, path: str) -> list[Frame]:
    """Keyframes in memory -> pHash dedup -> thumbnails to object storage."""
    db.set_status(video_id, "sampling", progress=0.0)
    frames = sample(Path(path))
    if not frames:
        raise RuntimeError("No frames could be extracted from the video.")
    kept = dedup(frames)
    print(f"[sample] {video_id}: {len(frames)} sampled -> {len(kept)} after dedup")

    # Idempotent re-run: clear any thumbnails a previous attempt left behind.
    storage.delete_prefix(storage.frame_prefix(user_id, video_id))
    done = 0

    def _put(i_f: tuple[int, Frame]) -> None:
        nonlocal done
        i, f = i_f
        storage.put_bytes(storage.frame_key(user_id, video_id, i), f.jpeg, "image/jpeg")
        done += 1
        if done % 25 == 0:
            db.set_progress(video_id, done / len(kept))

    with ThreadPoolExecutor(max_workers=_UPLOAD_POOL) as ex:
        list(ex.map(_put, enumerate(kept)))
    db.set_progress(video_id, 1.0)
    return kept


@task(name="embed-index", retries=2, retry_delay_seconds=60)
def t_embed_index(video_id: str, user_id: str, frames: list[Frame]) -> int:
    """Batched CLIP embeddings -> idempotent multi-tenant Qdrant upsert."""
    db.set_status(video_id, "embedding", progress=0.0)
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
                       "idx": start + i, "embed_version": EMBED_VERSION}
                      for i, f in enumerate(batch)],
        )
        total += len(batch)
        db.set_progress(video_id, total / len(frames))
    db.set_status(video_id, "indexed", frame_count=total,
                  embed_version=EMBED_VERSION, progress=1.0)
    return total


@flow(name="ms-ingest-video", log_prints=True, timeout_seconds=3600)
def ingest_video(video_id: str, user_id: str) -> dict:
    attempt = db.bump_attempts(video_id)
    path: str | None = None
    try:
        path = t_fetch(video_id, user_id)
        if not path:  # duplicate — already marked 'skipped' by t_fetch
            print(f"[ingest] {video_id} skipped (duplicate content)")
            return {"video_id": video_id, "skipped": True}
        frames = t_sample(video_id, user_id, path)
        n = t_embed_index(video_id, user_id, frames)
        print(f"[ingest] {video_id} indexed: {n} frames (attempt {attempt})")
        return {"video_id": video_id, "frames": n}
    except Exception as exc:
        db.set_status(video_id, "failed", error=f"{type(exc).__name__}: {exc}")
        raise  # Prefect marks the run Failed; full trace in the Cloud UI
    finally:
        if path:  # scratch only — durable copies live in object storage
            Path(path).unlink(missing_ok=True)
