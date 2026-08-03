"""Postgres (Neon) access layer — the source manifest, source of truth.

One row per (user's) source, of any kind; `status` tracks the ingest lifecycle:
pending -> fetching -> sampling -> embedding -> indexed | skipped | failed
(skipped = duplicate (user_id, source_hash); indexed = searchable in Qdrant).
The middle stages are per kind — a paper parses and chunks where a video samples
frames — so only the ends of that line are shared. What counts as "still in
flight" is therefore derived, not listed: see NOT_INFLIGHT_STATUSES.

Split into a package when it outgrew a readable file, along the seam of who is
asking: manifest.py for a source's state (API + flows), queue.py for what should
run next and what died (dispatcher + reaper), llm.py for a different table
entirely, _core.py for the pool and the SQL fragments they share. Callers keep
using `db.<name>` — this module re-exports all of it.
"""
from ._core import LostOwnership, init_schema, pool
from .llm import delete_user_llm, get_user_llm, set_user_llm
from .manifest import (delete_video, find_duplicate, get_video, heartbeat,
                       list_videos, requeue, set_flow_run, set_progress,
                       set_status, start_run, upsert_pending, videos_by_ids)
from .queue import count_inflight, reap_stale, wfq_claim

__all__ = [
    "LostOwnership", "count_inflight", "delete_user_llm", "delete_video",
    "find_duplicate", "get_user_llm", "get_video", "heartbeat", "init_schema",
    "list_videos", "pool", "reap_stale", "requeue", "set_flow_run",
    "set_progress", "set_status", "set_user_llm", "start_run",
    "upsert_pending", "videos_by_ids", "wfq_claim",
]
