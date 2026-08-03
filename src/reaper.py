"""Crash recovery — the loop that puts a dead run back in the queue.

Why it has to exist: the dispatcher only ever claims 'pending' (db/queue.py), while
count_inflight charges a dispatch slot for every row that is neither pending nor
finished. A worker that dies mid-run therefore leaves a row that nothing will
ever run again AND that holds a slot for good: with DISPATCH_MAX_INFLIGHT=2, two
crashes close the queue for every user. That is measured, not feared — a killed
worker left a source on 'embedding' and nothing brought it back.

It runs as a daemon thread in worker.py, next to the dispatcher. Same shape (a
periodic tick over the manifest) and the reaper's entire purpose is to hand work
back to the dispatcher, so they belong side by side. Several workers means
several reapers, which is safe because the reset is one conditional UPDATE.

Deliberately NOT in the API process: if every worker is down there is nothing to
reap for, since nothing can run anyway, and the first worker to come back does
the cleaning. Move it if the API ever needs the slots freed without a worker.

Two independent signals decide that a row is dead, because neither covers both
failure modes:

  heartbeat  a running flow vouches for its row every HEARTBEAT_INTERVAL_S
             (src/heartbeat.py). Silence past HEARTBEAT_STALE_AFTER_S means the
             PROCESS is gone. It needs its own column: `updated_at` moves for
             four reasons unrelated to life, so a healthy 80s download is
             indistinguishable from a corpse by that measure.
  deadline   every phase carries a budget (config.PHASE_BUDGETS_S) stamped when
             the phase is written. Catches a process that is alive but no longer
             getting anywhere, which a heartbeat by construction cannot see.

Their timescales differ on purpose. A crash is the common case and has to be
noticed in seconds; a hang is rare and gets minutes.

THE ERROR THAT MUST NEVER HAPPEN, and the reason for every conservative choice
below: reaping a row whose flow is still alive. The dispatcher then admits a
second run on the same source, the two share one scratch file, and the delete at
the top of one lands between the upserts of the other inside a try/except pass.
The result is a half-empty index that calls itself 'indexed', with no error
anywhere. Every budget is therefore set above the phase's legitimate worst case
(including retry backoff), ownership transfer restamps the clock (db.start_run),
and nothing outside this module may move an in-flight row.
"""
from __future__ import annotations

from . import config, db, ticker


def reap_once() -> list[dict]:
    """Requeue every source whose worker went away. Returns the rows it moved."""
    rows = db.reap_stale(config.HEARTBEAT_STALE_AFTER_S, config.MAX_ATTEMPTS)
    for row in rows:
        # Loud on purpose, one line per row: this is the only trace that a run
        # was lost and re-offered, and it is what a resilience harness reads.
        # user_id is here so the row can actually be looked up afterwards:
        # /admin/sources is tenant-scoped and will not show it without one.
        print(f"[reap] {row['id']} ({row['kind']}, user {row['user_id']}) "
              f"-> {row['status']} [attempt {row['attempts']}] {row['error']}")
    return rows


def start_in_background() -> None:
    """Start the reaper as a daemon thread (no-op when disabled)."""
    ticker.start(
        "reap", reap_once, config.REAP_INTERVAL_S,
        enabled=config.REAP_ENABLED,
        on_start=(f"[reap] crash recovery on — no heartbeat for "
                  f"{config.HEARTBEAT_STALE_AFTER_S:.0f}s or past its phase budget, "
                  f"tick {config.REAP_INTERVAL_S}s, giving up after "
                  f"{config.MAX_ATTEMPTS} consecutive failures"),
        on_disabled="[reap] crash recovery disabled — a dead run keeps its slot")
