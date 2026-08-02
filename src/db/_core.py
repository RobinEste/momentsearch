"""Connection pool, schema bootstrap, and the SQL fragments the rest shares.

Everything in this package writes to one Postgres; these are the pieces that
would otherwise be imported across sibling modules by their private names. The
fragments in particular exist to stop wording from drifting between statements
that must agree, so they cannot themselves live in only one of those statements.
"""
from __future__ import annotations

import os

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from ..config import DATABASE_URL, HEARTBEAT_STALE_AFTER_S, NOT_INFLIGHT_STATUSES
from ..schema import SCHEMA

# Every write that moves a row's clock forward stamps the new deadline, and four
# do: set_status, start_run, set_progress and wfq_claim. (reap_stale is not one
# of them — it NULLs the deadline, because a requeued row is waiting, not late.)
# Three of the four have to be a single atomic statement, so the spelling is
# shared rather than the code: make_interval is strict, so a NULL budget yields
# a NULL deadline without a CASE around it.
_STAMP_DEADLINE = "now() + make_interval(secs => %s::float8)"

# Letting go of a row: everything that says "a run holds this" goes at once.
# Adding a column to that protocol is otherwise an edit in several places with
# nothing that fails if one is missed — the failure mode this diff already had
# to hand-rescue once (see init_schema).
_RELEASE = ("progress = NULL, phase_deadline = NULL, "
            "last_heartbeat_at = NULL, run_token = NULL")

# "Nobody is working on this row." Named because upsert_pending needs it once
# per column it guards, and seven hand-copied copies is a request to the reader
# rather than a property of the code. Named for what it MATCHES, not for what
# the CASE does with it: every use reads `WHEN <this> THEN <reset it>`.
_NOT_IN_FLIGHT = "ms_videos.status = ANY(%(_not_inflight)s)"

# The fence. Every write a running flow makes carries the token start_run gave
# it, and this clause is what makes a stale one harmless: a flow that lost the
# row (reaped while alive, then re-admitted to another run) matches no rows and
# is told so, instead of overwriting whoever owns it now. A NULL token means
# "not from inside a run" — the API and the dispatcher — and skips the check.
_OWNED_BY = "(%s::text IS NULL OR run_token = %s)"


class LostOwnership(RuntimeError):
    """A write was refused because another run owns this source now."""



_pool: ConnectionPool | None = None
_pool_pid: int | None = None


def pool() -> ConnectionPool:
    """Process-local pool. Prefect runs flows in subprocesses; a child must
    never reuse the parent's SSL connections (corrupts the TLS stream), so a
    fork gets a fresh pool."""
    global _pool, _pool_pid
    if _pool is None or _pool_pid != os.getpid():
        # check= pings each connection before lending it out — Neon silently
        # drops idle SSL connections, which otherwise 500s the first request
        # after a quiet period.
        _pool = ConnectionPool(DATABASE_URL, min_size=1, max_size=5,
                               check=ConnectionPool.check_connection,
                               kwargs={"row_factory": dict_row})
        _pool_pid = os.getpid()
    return _pool


def init_schema() -> None:
    with pool().connection() as conn:
        conn.execute(SCHEMA)
        # One-time rescue for rows already in flight when the three recovery
        # columns arrived: all are NULL, which reads as "nobody ever took this
        # row", and reap_stale then leaves them alone forever.
        # Those are exactly the rows the reaper was written for — a killed worker
        # left one on 'embedding' and nothing brought it back — so without this
        # the fix is a no-op for its own motivating case, while they keep
        # charging a dispatch slot. No live process can be behind one of them
        # (their code predates the heartbeat), so stamping a beat and a past
        # deadline sends them on the next tick rather than a stale window later.
        #
        # Safe to leave here permanently: any row that enters flight after this
        # deploy gets all three columns from start_run and wfq_claim, so it can
        # never match again.
        conn.execute(
            "UPDATE ms_videos SET last_heartbeat_at = now() - make_interval("
            "  secs => %s), phase_deadline = now() "
            "WHERE status <> ALL(%s) AND last_heartbeat_at IS NULL "
            "AND phase_deadline IS NULL AND run_token IS NULL "
            # during a rolling restart the OLD worker is still running flows,
            # and ITS code writes none of these three columns -- so a live row
            # looks exactly like an abandoned one. Age is the only thing that
            # tells them apart: a row untouched for minutes has no live writer.
            "AND updated_at < now() - interval '5 minutes'",
            (HEARTBEAT_STALE_AFTER_S + 1, list(NOT_INFLIGHT_STATUSES)),
        )
