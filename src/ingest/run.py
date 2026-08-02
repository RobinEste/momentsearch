"""The shell every ingest flow runs inside — ownership, liveness, cleanup.

The three flows differ in their middle: a video samples frames, a paper parses
pages, a deck parses slides. Everything around that middle was already the same
in all three (count the attempt, mark 'failed' on the way out, delete the scratch
file), and crash recovery adds two more things that must be identical everywhere:
taking the row, and letting go of it. Written once here rather than four times,
because a fourth source kind should get crash recovery by existing, not by
someone remembering.

The one rule worth stating on its own: a flow that has LOST the row writes
nothing further. The reaper can requeue a run that is still alive (a frozen
machine, a phase that outran its budget), the dispatcher then admits a second
run, and from that moment the first one is a ghost. Its status writes would land
on its successor's row and its `finally` would delete its successor's download.
So `db.LostOwnership` propagates untouched: no 'failed' status, no unlink, and
Prefect records the run as failed, which it is.
"""
from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .. import db, heartbeat


@dataclass
class Run:
    """One attempt at one source. `token` proves this process owns the row and
    rides along on every write; `scratch` is whatever this attempt downloaded,
    set by the flow once it exists so the shell can clean it up."""
    source_id: str
    attempt: int
    token: str
    scratch: str | None = None


@contextlib.contextmanager
def ingest_run(source_id: str) -> Iterator[Run]:
    attempt, token = db.start_run(source_id)
    run = Run(source_id=source_id, attempt=attempt, token=token)
    # Says "this process is still here" every few seconds, so the reaper can
    # tell a dead worker from a slow download (src/heartbeat.py).
    with heartbeat.beating(source_id, token):
        try:
            yield run
        except db.LostOwnership as exc:
            # Stand down completely. Marking 'failed' here would stamp it on the
            # run that owns the row now, and the scratch file is at a path
            # derived from the source id — the successor's download, not ours.
            # Clearing it is what keeps the `finally` below from taking it:
            # `finally` runs on this path too, which is exactly the mistake this
            # line exists to prevent.
            run.scratch = None
            print(f"[ingest] {source_id}: {exc} — standing down")
            raise
        except Exception as exc:
            try:
                db.set_status(source_id, "failed",
                              error=f"{type(exc).__name__}: {exc}", token=token)
            except db.LostOwnership:
                # Same proof as the branch above, so the same consequence: the
                # scratch path is derived from the source id, and the run that
                # owns the row now is downloading to it. Missing this line here
                # while having it there is the whole bug in one asymmetry.
                run.scratch = None
                print(f"[ingest] {source_id}: failed, but another run owns the row")
            raise  # Prefect marks the run Failed; full trace in the Cloud UI
        finally:
            if run.scratch:  # scratch only — durable copies live in object storage
                Path(run.scratch).unlink(missing_ok=True)
