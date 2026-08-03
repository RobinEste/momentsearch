"""Fair dispatcher — the WFQ scheduler that sits in front of Prefect.

Why this exists: if the API enqueued every video to Prefect at register-time,
Prefect would run them in submitted order (FIFO) — one user who uploads 50
videos blocks everyone behind them. Instead, videos wait `pending` in Postgres
and THIS loop admits them:

  every DISPATCH_INTERVAL_S:
    slots = DISPATCH_MAX_INFLIGHT - (videos currently queued/running)
    claim up to `slots` pending videos in FAIR order (round-robin across users)
    schedule a Prefect run for each

Because only ~capacity videos are ever handed to Prefect at once, the *waiting
line lives in our DB, fairly ordered* (db.claim_within_capacity) rather than
FIFO inside Prefect. No user can starve the others. Set ENABLE_FAIR_DISPATCH=false
to fall back to immediate FIFO enqueue (useful for A/B teaching the difference).

Runs as a background thread in worker.py, so there is one dispatcher per worker
replica and the ceiling has to hold across all of them. It does not hold by
itself: reading the free slots and claiming against that reading are two steps,
and N dispatchers interleave them into N times the ceiling. db has both steps in
one locked transaction for that reason — see db.claim_within_capacity, which is
where the whole argument lives.
"""
from __future__ import annotations

from . import config, db, jobs, ticker


def dispatch_once() -> int:
    """Admit as many fairly-chosen pending sources as free FLEET capacity allows.
    Returns how many were dispatched this tick.

    Only kinds with an ingest flow are claimed (jobs.INGEST_DEPLOYMENTS); the
    rest keep waiting as `pending`, which costs no capacity and loses nothing.

    Sizing the claim and making it is one call on purpose. As two — a
    count_inflight() here, a claim there — every dispatcher in the fleet sized
    against the same stale count and the ceiling was enforced per worker rather
    than across them.

    Enqueueing stays out here, after the claim has committed: it is a network
    round trip per row, and the claim holds a fleet-wide lock.
    """
    claimed = db.claim_within_capacity(config.DISPATCH_MAX_INFLIGHT,
                                       jobs.dispatchable_kinds())
    for row in claimed:
        try:
            jobs.enqueue(row["id"], row["user_id"], row["kind"])
        except Exception as exc:
            # Couldn't reach Prefect — put it back so it's retried next tick.
            db.set_status(row["id"], "pending", error=f"dispatch: {exc}")
    if claimed:
        print(f"[dispatch] admitted {', '.join(r['id'] for r in claimed)} "
              f"({db.count_inflight()}/{config.DISPATCH_MAX_INFLIGHT} in flight)")
    return len(claimed)


def start_in_background() -> None:
    """Start the dispatcher as a daemon thread (no-op if fair dispatch is off)."""
    ticker.start(
        "dispatch", dispatch_once, config.DISPATCH_INTERVAL_S,
        enabled=config.ENABLE_FAIR_DISPATCH,
        on_start=(f"[dispatch] fair scheduler on — max in-flight "
                  f"{config.DISPATCH_MAX_INFLIGHT}, tick {config.DISPATCH_INTERVAL_S}s"),
        on_disabled="[dispatch] fair dispatch disabled — FIFO (immediate enqueue)")
