"""When a retry is worth its delay, and when it only holds a worker slot.

Prefect's `retries=` answers "how often", never "for which failure". The
difference matters most for the failure this module exists for: every task in an
ingest flow writes through db.set_status(..., token=...), and that raises
LostOwnership the moment the row is gone or belongs to another run. No retry can
un-delete a row, so the delay is paid for a failure that is already decided.

The cost is not the failed run, it is the SLOT. retry_delay_seconds keeps the
run alive on a worker while it waits — [30, 120] for the fetch tasks — so one
orphaned run occupies a quarter of a two-worker fleet for ~150 seconds.

Measured: benchmark/bench.py registers 30 probe sources for the accept-latency
gate and deletes them seconds later. Those whose runs had already started spent
the fleet's capacity on retry backoff, and the eight real documents behind them
did not begin for ~480s. The throughput gate read 0.72 chunks/s; the documents,
once running, indexed at roughly 7.

Deliberately narrow. Anything that is not provably permanent keeps retrying,
including an unreadable state — the failure direction to prefer is a wasted
retry, not a lost document.
"""
from __future__ import annotations

from .. import db


def retry_unless_orphaned(task, task_run, state) -> bool:
    """Prefect retry_condition_fn: False only when this run lost its row."""
    exc = state.result(raise_on_failure=False)
    return not isinstance(exc, db.LostOwnership)
