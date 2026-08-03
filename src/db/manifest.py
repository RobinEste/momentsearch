"""The source manifest: one row per source, and its state.

Answers "what is this source's state" for the API and the ingest flows. The
queue's view of the same table — what should run next, what died — lives in
queue.py, because it is a different question with different callers.

The table is still named ms_videos and its id column is still the generic source
id, even though papers and decks live here too. Renaming both would touch the
search and cleanup paths that non-negotiable 6 protects, for no functional gain
during this assignment. Flagged as future cleanup.
"""
from __future__ import annotations

import uuid
from typing import Any

from ..config import (ATTEMPT_RESET_STATUSES, NOT_INFLIGHT_STATUSES,
                      PHASE_BUDGET_SCALE, PROGRESS_STALL_S, TERMINAL_STATUSES,
                      phase_budget_s)
from ._core import (_NOT_IN_FLIGHT, _OWNED_BY, _RELEASE, _STAMP_DEADLINE,
                    LostOwnership, pool)


def upsert_pending(source: dict[str, Any]) -> dict:
    """Insert a source as pending; re-submitting an existing id resets it.

    `kind` is a required key with no Python-side default, on purpose. The column
    defaults to 'video', so a caller that omits it would file a paper as a video
    silently — the loud KeyError from psycopg is the better outcome.

    ON CONFLICT deliberately leaves `kind` alone, and that is only safe because
    every id carries its kind: yt_/up_ are videos by construction, and a doc_ id
    hashes the kind together with the URI (see api/admin.py::_document_id). A
    re-submission therefore always lands on a row of the same kind. Drop that
    property and this clause starts silently ignoring a corrected kind.

    A row that is still IN FLIGHT keeps its status: no external caller may move
    one (src/reaper.py says what a second flow on one source destroys). The
    running flow itself does keep writing its own row, but only for as long as
    it still holds the token _OWNED_BY checks. The return value is then the
    CURRENT status rather than 'pending' — the 202 contract holds, but a caller
    that string-matches on "pending" sees the truth instead.

    The per-column copies of _NOT_IN_FLIGHT look like they want hoisting into
    `ON CONFLICT ... DO UPDATE ... WHERE`. They do not: that form skips the row,
    and a skipped row is absent from RETURNING, so this would hand back None for
    exactly the case it exists to describe.
    """
    with pool().connection() as conn:
        row = conn.execute(
            f"""
            INSERT INTO ms_videos (id, user_id, kind, source, url, storage_key, source_hash, title, status)
            VALUES (%(id)s, %(user_id)s, %(kind)s, %(source)s, %(url)s, %(storage_key)s,
                    %(source_hash)s, %(title)s, 'pending')
            ON CONFLICT (id) DO UPDATE SET
                url = COALESCE(EXCLUDED.url, ms_videos.url),
                storage_key = COALESCE(EXCLUDED.storage_key, ms_videos.storage_key),
                source_hash = COALESCE(EXCLUDED.source_hash, ms_videos.source_hash),
                title = COALESCE(EXCLUDED.title, ms_videos.title),
                status = CASE WHEN {_NOT_IN_FLIGHT}
                              THEN 'pending' ELSE ms_videos.status END,
                error = CASE WHEN {_NOT_IN_FLIGHT}
                             THEN NULL ELSE ms_videos.error END,
                progress = CASE WHEN {_NOT_IN_FLIGHT}
                                THEN NULL ELSE ms_videos.progress END,
                phase_deadline = CASE WHEN {_NOT_IN_FLIGHT}
                                      THEN NULL ELSE ms_videos.phase_deadline END,
                last_heartbeat_at = CASE WHEN {_NOT_IN_FLIGHT}
                                         THEN NULL ELSE ms_videos.last_heartbeat_at END,
                run_token = CASE WHEN {_NOT_IN_FLIGHT}
                                 THEN NULL ELSE ms_videos.run_token END,
                -- A human asking again is a fresh start, so the give-up counter
                -- goes back to zero with it. Guarded the same way: a run in
                -- flight keeps the count it has earned.
                attempts = CASE WHEN {_NOT_IN_FLIGHT}
                                THEN 0 ELSE ms_videos.attempts END,
                updated_at = now()
            RETURNING *
            """,
            {**source, "_not_inflight": list(NOT_INFLIGHT_STATUSES)},
        ).fetchone()
    return row


def set_status(video_id: str, status: str, *, error: str | None = None,
               title: str | None = None, frame_count: int | None = None,
               source_hash: str | None = None, embed_version: str | None = None,
               progress: float | None = None, units: int | None = None,
               token: str | None = None) -> None:
    """Write the row's phase, and with it the deadline that phase gets.

    Every flow's stage transitions come through here, so a new stage cannot
    forget to get a budget: an unbudgeted name falls back to the roomy default
    rather than to no deadline at all (config.phase_budget_s). It is not the
    only writer of phase_deadline — see _STAMP_DEADLINE for that list.

    `units` is the size of the work the phase is about to do — frames, chunks —
    where the caller knows it at that moment; omitted, the phase gets its flat
    base budget. A clean landing also clears `attempts`
    (config.ATTEMPT_RESET_STATUSES) and releases the row's token.
    """
    budget = phase_budget_s(status, units)
    with pool().connection() as conn:
        cur = conn.execute(
            f"""
            UPDATE ms_videos SET status = %s, error = %s,
                title = COALESCE(%s, title),
                frame_count = COALESCE(%s, frame_count),
                source_hash = COALESCE(%s, source_hash),
                embed_version = COALESCE(%s, embed_version),
                progress = %s,
                phase_deadline = {_STAMP_DEADLINE},
                attempts = CASE WHEN %s = ANY(%s) THEN 0 ELSE attempts END,
                -- Landing releases the row, whichever way it ended: the next run
                -- to take it gets its own token rather than inheriting this one.
                run_token = CASE WHEN %s = ANY(%s) THEN NULL ELSE run_token END,
                updated_at = now()
            WHERE id = %s AND {_OWNED_BY}
            """,
            (status, error, title, frame_count, source_hash, embed_version,
             progress, budget, status, list(ATTEMPT_RESET_STATUSES),
             status, list(TERMINAL_STATUSES), video_id, token, token),
        )
        if token is not None and cur.rowcount == 0:
            raise LostOwnership(
                f"{video_id}: cannot set {status!r}, another run owns it")


def start_run(video_id: str) -> tuple[int, str]:
    """A process is taking this row: count the attempt and restart its clock.

    This is a phase change in everything but name — ownership moves from the
    queue to a worker — so it restamps the deadline like every other one. The
    row is still 'queued' here, its first real status write a Prefect task
    submission away, and without this the 'queued' budget keeps running against
    a run that has already started. Letting it expire is precisely the reap-a-
    live-flow case src/reaper.py exists to prevent, which is why this is not
    just `attempts + 1`.
    """
    token = uuid.uuid4().hex
    with pool().connection() as conn:
        row = conn.execute(
            f"""
            UPDATE ms_videos SET attempts = attempts + 1, run_token = %s,
                   phase_deadline = {_STAMP_DEADLINE},
                   -- Vouch once, here. The beating thread's first write is a
                   -- whole interval away, and until it lands the row matches
                   -- NEITHER reaper branch: a process that dies inside that
                   -- window would hold its dispatch slot for good. Same
                   -- statement, so it costs nothing.
                   last_heartbeat_at = now(), updated_at = now()
            WHERE id = %s RETURNING attempts
            """,
            (token, phase_budget_s("queued"), video_id),
        ).fetchone()
    return (row["attempts"] if row else 0), token


def requeue(video_id: str, user_id: str) -> dict | None:
    """Put a source back in line by hand. None means it is in flight and was
    left alone, which the caller turns into a 409.

    Conditional UPDATE rather than read-then-write: the latter is a second
    implementation of the same guard, and it has a window where a row entering
    flight between the SELECT and the UPDATE is reset anyway.
    """
    with pool().connection() as conn:
        return conn.execute(
            f"""
            UPDATE ms_videos SET status = 'pending', error = NULL,
                   {_RELEASE}, attempts = 0, updated_at = now()
            WHERE id = %s AND user_id = %s AND status = ANY(%s)
            RETURNING *
            """,
            (video_id, user_id, list(NOT_INFLIGHT_STATUSES)),
        ).fetchone()


def heartbeat(source_id: str, token: str) -> bool:
    """Vouch for a running source. False means the row is gone or has finished,
    which is the beating thread's cue to stop (src/heartbeat.py).

    Deliberately narrow twice over. It only touches rows still in flight, so a
    thread that outlives its flow cannot keep a failed row looking fresh; and it
    writes ONLY last_heartbeat_at. Bumping updated_at here would turn that column
    into the liveness proxy the whole design says it must not be — and it is a
    field clients read, so an idle row would appear to change every few seconds.
    """
    with pool().connection() as conn:
        row = conn.execute(
            f"UPDATE ms_videos SET last_heartbeat_at = now() "
            f"WHERE id = %s AND status <> ALL(%s) AND {_OWNED_BY} RETURNING id",
            (source_id, list(NOT_INFLIGHT_STATUSES), token, token),
        ).fetchone()
    return row is not None


def set_progress(video_id: str, progress: float, token: str) -> None:
    """Record progress within the current phase, and buy it more time.

    Observed progress is direct evidence that the phase is not hung, which is
    the only thing its deadline is trying to detect. Without this the deadline
    measures elapsed time against an ESTIMATE of the work, and a stage that is
    visibly at 60% is reaped exactly like a frozen one — the thumbnail uploads
    inside t_sample, for instance, scale with the network rather than with the
    frame count the budget is derived from.

    Once a phase is ticking, the question becomes "how long since the last
    tick", which is a different quantity from the phase budget and has its own
    name (config.PROGRESS_STALL_S). Reaching the roomy default through a stage
    name that matches nothing would have said 900s while looking like a lookup.
    """
    with pool().connection() as conn:
        cur = conn.execute(
            f"""UPDATE ms_videos SET progress = %s, updated_at = now(),
                   phase_deadline = {_STAMP_DEADLINE}
                 WHERE id = %s AND {_OWNED_BY}""",
            (round(progress, 3), PROGRESS_STALL_S * PHASE_BUDGET_SCALE, video_id,
             token, token))
        if cur.rowcount == 0:
            raise LostOwnership(f"{video_id}: another run owns it now")


def get_video(video_id: str) -> dict | None:
    with pool().connection() as conn:
        return conn.execute("SELECT * FROM ms_videos WHERE id = %s", (video_id,)).fetchone()


def find_duplicate(user_id: str, source_hash: str, exclude_id: str,
                   kind: str) -> dict | None:
    """An already-indexed source of the SAME KIND with the same content.

    Scoped by kind because identical bytes can be two legitimate sources: the
    same PDF ingested as a paper (page locators, text route) and as a deck
    (slide locators, its own route) are different products of the same file.
    Without this filter the second one is silently marked 'skipped' as a
    duplicate of the first, and never gets indexed at all.
    """
    with pool().connection() as conn:
        return conn.execute(
            """
            SELECT * FROM ms_videos
            WHERE user_id = %s AND source_hash = %s AND id <> %s AND kind = %s
              AND status = 'indexed'
            LIMIT 1
            """,
            (user_id, source_hash, exclude_id, kind),
        ).fetchone()


def list_videos(user_id: str, status: str | None = None) -> list[dict]:
    q = "SELECT * FROM ms_videos WHERE user_id = %s"
    params: list = [user_id]
    if status:
        q += " AND status = %s"
        params.append(status)
    q += " ORDER BY created_at DESC"
    with pool().connection() as conn:
        return conn.execute(q, tuple(params)).fetchall()


def videos_by_ids(ids: list[str]) -> dict[str, dict]:
    """Metadata join for search citations (title/url/source live here, not in Qdrant)."""
    if not ids:
        return {}
    with pool().connection() as conn:
        rows = conn.execute("SELECT * FROM ms_videos WHERE id = ANY(%s)", (ids,)).fetchall()
    return {r["id"]: r for r in rows}


def set_flow_run(video_id: str, flow_run_id: str) -> None:
    """Record which Prefect run was scheduled for this row.

    Deliberately not guarded by run_token: this is written at SCHEDULING time,
    when the row is `queued` and no process owns it yet — a token guard would
    reject every write. It is a note about what was asked of Prefect, not a
    claim on the row, and the delete path is its only reader.

    Not raising when the row is gone is the point: the source can be deleted
    between enqueue() and this write, and failing here would turn a successful
    schedule into a 500 for a row that no longer matters.
    """
    with pool().connection() as conn:
        conn.execute("UPDATE ms_videos SET flow_run_id = %s WHERE id = %s",
                     (flow_run_id, video_id))


def delete_video(video_id: str) -> None:
    with pool().connection() as conn:
        conn.execute("DELETE FROM ms_videos WHERE id = %s", (video_id,))
