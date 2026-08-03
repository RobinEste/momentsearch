"""The manifest read as a work queue: what to admit, and what died.

Called by src/dispatcher.py and src/reaper.py and by nothing else. Same table as
manifest.py, different question: not "what is this source's state" but "what
should run next, and what is no longer running at all".
"""
from __future__ import annotations

from psycopg import Connection

from ..config import NOT_INFLIGHT_STATUSES, phase_budget_s
from ._core import _RELEASE, _STAMP_DEADLINE, pool

# Names the fleet's dispatch lock. An advisory lock refers to nothing — it is a
# name Postgres tracks a holder for, not a lock on any row or table — so any
# 64-bit constant does. All that matters is that every dispatcher picks the
# same one, which is why it is a module constant and not a caller's argument.
_DISPATCH_LOCK_KEY = 4_071_982_026

# Shared so the decision and the report of it cannot drift: claim_within_capacity
# decides against this count and src/dispatcher.py prints it. Two spellings of
# "in flight" would let the log line contradict the choice it exists to explain.
_COUNT_INFLIGHT = "SELECT count(*) AS n FROM ms_videos WHERE status <> ALL(%s)"


def count_inflight() -> int:
    """How many sources currently occupy execution capacity (scheduled/running).

    Asked as "not waiting and not finished" rather than "in one of these stages",
    so a stage name this function has never heard of still counts. Equivalent to
    the old enumeration for every existing video status; it differs only for
    names that did not exist yet.
    """
    with pool().connection() as conn:
        row = conn.execute(
            _COUNT_INFLIGHT, (list(NOT_INFLIGHT_STATUSES),)).fetchone()
    return row["n"] if row else 0


def claim_within_capacity(max_inflight: int,
                          kinds: tuple[str, ...]) -> list[dict]:
    """Claim fairly-chosen pending sources, up to what free FLEET capacity allows.

    Counting the free slots and claiming against that count is ONE transaction
    here, behind one lock, and that is the whole reason this function exists.
    The halves used to be two separate calls in src/dispatcher.py, and every
    worker runs a dispatcher: N of them read the same "0 in flight" before any
    of them had claimed, so each admitted a full DISPATCH_MAX_INFLIGHT and the
    fleet admitted N times its own ceiling. Measured at 8 in flight against a
    limit of 4, with two replicas.

    The atomic claim in _claim_fairly does NOT prevent that, and was never meant
    to. Its promise is that no source is handed out twice, and it kept it — the
    second dispatcher simply claimed four OTHER rows. Handing out each row once
    and admitting the right NUMBER of rows are different guarantees, and only
    the first is a property of a single statement.

    What over-admitting costs is fairness, not throughput. Everything above the
    fleet's execution capacity sits inside Prefect in submitted order, where
    _claim_fairly's round-robin can no longer reach it, so the next user to
    arrive queues behind rows that can no longer be reordered. That is also why
    DISPATCH_MAX_INFLIGHT belongs at replicas x concurrency and not above it:
    the difference between the two IS the depth of that unreachable buffer.

    try_ rather than a blocking wait: a dispatcher that loses the race does
    nothing this tick and tries again in DISPATCH_INTERVAL_S, which is the same
    outcome waiting would have given it (no free slots, nothing to claim)
    without parking a thread on a lock. The lock is transaction-scoped, so the
    commit below releases it, and so does a rollback or the connection dying —
    there is no path where a crashed dispatcher keeps the fleet from admitting.

    jobs.enqueue is deliberately NOT called from in here: it is a network round
    trip to Prefect, and holding the fleet's dispatch lock across it would put
    every worker's dispatcher behind Prefect's latency.
    """
    if max_inflight <= 0 or not kinds:
        return []
    with pool().connection() as conn:
        got = conn.execute("SELECT pg_try_advisory_xact_lock(%s) AS ok",
                           (_DISPATCH_LOCK_KEY,)).fetchone()
        if not (got and got["ok"]):
            return []  # another dispatcher holds the tick; ours comes round again
        row = conn.execute(
            _COUNT_INFLIGHT, (list(NOT_INFLIGHT_STATUSES),)).fetchone()
        slots = max_inflight - (row["n"] if row else 0)
        if slots <= 0:
            return []
        return _claim_fairly(conn, slots, kinds)


def _claim_fairly(conn: Connection, limit: int,
                  kinds: tuple[str, ...]) -> list[dict]:
    """Atomically claim up to `limit` pending sources in FAIR (round-robin across
    users) order, flipping them pending -> queued. Returns the claimed rows.

    Takes a connection instead of opening one, and is private, for the same
    reason: this claim and the capacity check that sized `limit` have to be one
    transaction. Claiming without reserving that capacity first is exactly the
    bug claim_within_capacity exists to remove, so a public entry point here
    would be an invitation to put it back.

    Fairness: rank each user's pending sources by age (row_number partitioned by
    user_id), then order by that rank first — so we take everyone's oldest, then
    everyone's 2nd, ... A user who dumped 50 sources only gets one slot per round,
    exactly like the others. The UPDATE ... WHERE status='pending' RETURNING is
    the atomic claim: if two dispatchers race, each row is handed out once.

    Fairness spans kinds: a bulk of papers and a queue of videos take turns in one
    line, which is the point of admitting them from the same manifest. `kinds`
    narrows that to the types an ingest flow actually exists for — a kind left out
    stays `pending` instead of being handed to the wrong flow, and joins the line
    for real the moment its flow lands.

    Known limit, left in on purpose: the tiebreak between users at the SAME rank
    is `id`, a TEXT key of the form doc_… / up_… / yt_… (src/schema.py), so at
    equal rank a document beats an upload beats a YouTube add. Nobody starves —
    the winner leaves the pending pool, so the loser takes the next tick — but
    it is arbitrary where ordering on created_at would be defensible.
    """
    picked = conn.execute(
        """
        SELECT id FROM (
            SELECT id, row_number() OVER (
                PARTITION BY user_id ORDER BY created_at, id) AS rn
            FROM ms_videos WHERE status = 'pending' AND kind = ANY(%s)
        ) t
        ORDER BY rn, id
        LIMIT %s
        """,
        (list(kinds), limit),
    ).fetchall()
    ids = [r["id"] for r in picked]
    if not ids:
        return []
    return conn.execute(
        f"""
        UPDATE ms_videos SET status = 'queued', updated_at = now(),
               phase_deadline = {_STAMP_DEADLINE},
               -- Admitting is a fresh start: whatever process vouched for
               -- this row last time is long gone, and leaving its beat
               -- or its token behind would speak for it after the fact.
               last_heartbeat_at = NULL, run_token = NULL
        WHERE id = ANY(%s) AND status = 'pending'
        RETURNING id, user_id, kind
        """,
        (phase_budget_s("queued"), ids),
    ).fetchall()


def reap_stale(stale_after_s: float, max_attempts: int) -> list[dict]:
    """Requeue sources whose worker went away; give up on the ones that keep dying.

    One statement, and that is what makes several reapers safe: every worker
    runs one, and whichever UPDATE lands first moves the row out of the WHERE
    clause ('pending' and 'failed' are both in NOT_INFLIGHT_STATUSES), so the
    others match nothing rather than resetting it twice.

    Deliberately NOT behind claim_within_capacity's dispatch lock, and the
    difference is worth stating because the two loops otherwise look alike. A
    dispatcher tick is a read (how much room is left) followed by a write that
    spends what the read found, and two of those interleave into over-admission.
    A reaper tick is a single conditional write with no read to go stale — the
    WHERE clause re-evaluates against the committed row — so concurrency costs
    it duplicate queries and duplicate log lines, never a wrong outcome.

    Two triggers. A heartbeat that STOPPED means the process is gone. An expired
    deadline means nobody got anywhere in the time this phase was given, whether
    a process ever took the row or not.

    That last clause was once an exception: a row nobody had vouched for was
    left alone, because requeueing it does not cancel the Prefect run already
    scheduled for it, and a second run beside the first is the thing this module
    exists to prevent. Measured on a real crash, the exception costs more than
    it saves. Kill a worker between the claim and the moment Prefect delivers
    the run, and the row sits in 'queued' with no token and no beat, matching
    neither trigger and holding a dispatch slot for good — the exact failure
    this module was written for, while its 900s queued budget, written for
    exactly that case, could never fire.

    What changed is the fence. Two runs on one source are no longer a
    corruption: whichever calls start_run last owns the row, and the other is
    refused at its first write, which comes before it downloads anything. So the
    deadline may speak for a row nobody ever took, and MAX_ATTEMPTS bounds how
    often that repeats.

    Both NULL cases still fall out for free: `NULL < now()` is NULL and NULL only
    weakens an OR, so a row with neither a budget nor a beat is left alone.

    Expressions in SET read the row's OLD values, so `status` inside them is the
    phase it died in, not the one being written. The error message re-states the
    WHERE's heartbeat branch: change one and the other lies about why a row was
    reaped, so they are kept adjacent rather than factored apart.
    """
    with pool().connection() as conn:
        return conn.execute(
            f"""
            UPDATE ms_videos SET
                status = CASE WHEN attempts >= %(max_attempts)s THEN 'failed'
                              ELSE 'pending' END,
                -- Built from the OLD row on purpose: after this statement the
                -- phase, the beat and the progress are gone, so the only record
                -- of how far it got and how long it was silent is this string.
                error = CASE
                    WHEN attempts >= %(max_attempts)s
                        THEN 'reaper: gave up after ' || attempts
                             || ' attempt(s), last stuck in ' || status
                             || coalesce(' at ' || round(progress * 100) || '%%', '')
                    WHEN last_heartbeat_at < now() - make_interval(secs => %(stale)s)
                        THEN 'reaper: no heartbeat for '
                             || round(extract(epoch FROM now() - last_heartbeat_at))
                             || 's while ' || status
                             || coalesce(' at ' || round(progress * 100) || '%%', '')
                             || ', requeued'
                    ELSE 'reaper: '
                         || round(extract(epoch FROM now() - phase_deadline))
                         || 's over the time budget for ' || status
                         || coalesce(' at ' || round(progress * 100) || '%%', '')
                         || ', requeued'
                END,
                -- Whoever held this row no longer does (_RELEASE): a write
                -- still coming from that run now matches nothing instead of
                -- landing on its successor.
                {_RELEASE},
                updated_at = now()
            WHERE status <> ALL(%(not_inflight)s)
              AND (last_heartbeat_at < now() - make_interval(secs => %(stale)s)
                   OR phase_deadline < now())
            RETURNING id, user_id, kind, status, attempts, error
            """,
            {"max_attempts": max_attempts, "stale": stale_after_s,
             "not_inflight": list(NOT_INFLIGHT_STATUSES)},
        ).fetchall()
