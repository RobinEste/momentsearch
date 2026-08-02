"""Liveness signal for a running ingest flow.

The reaper has to tell "this worker died" apart from "this stage is just slow",
and updated_at alone cannot: it moves only when a status or a progress batch is
written, so an 80-second download looks exactly like a corpse by that measure.
So a flow says it itself, every HEARTBEAT_INTERVAL_S, from a daemon thread that
dies with the process it is vouching for. That is the whole trick: the signal
cannot outlive its subject.

What it does NOT cover, and must not be sold as covering: a process that is
alive but no longer getting anywhere keeps beating happily. Detecting that is
the job of the per-phase deadline (config.PHASE_BUDGETS_S), which is a different
question asked of a different column.
"""
from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator

from . import config, db


@contextlib.contextmanager
def beating(source_id: str, token: str | None = None) -> Iterator[None]:
    """Vouch for `source_id` until the block exits, the row leaves flight, or
    another run takes it over — `token` is what tells those last two apart."""
    stop = threading.Event()

    def _loop() -> None:
        # wait() returns True when it is set, False on timeout: one beat per
        # interval, and an immediate exit on stop instead of sleeping it out.
        while not stop.wait(config.HEARTBEAT_INTERVAL_S):
            try:
                if not db.heartbeat(source_id, token):
                    # Finished, failed, deleted — or reaped and handed to
                    # someone else. Either way this process no longer speaks
                    # for the row.
                    return
            except Exception as exc:  # noqa: BLE001
                # Keep beating. A thread that dies on one bad round trip takes
                # the liveness signal with it, and the reaper would then requeue
                # a run that is alive: the expensive failure, caused by the
                # cheap one.
                print(f"[heartbeat] {source_id}: {type(exc).__name__}: {exc}")

    thread = threading.Thread(target=_loop, daemon=True, name=f"heartbeat-{source_id}")
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=2.0)
