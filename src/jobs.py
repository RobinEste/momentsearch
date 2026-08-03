"""Prefect Cloud trigger layer — the API schedules runs, workers execute them.

One flow per source kind ("ms-ingest-video", "ms-ingest-paper", "ms-ingest-deck"
— the "ms-" prefix keeps them distinct from the digital-twin-akash flow living
in the same Prefect workspace), each served as an "ingest" deployment by
worker.py; the authoritative list is INGEST_DEPLOYMENTS below. The API
never imports a pipeline or its heavy deps (torch, ffmpeg) — it just asks
Prefect Cloud to schedule a run; any live worker picks it up. Retries/backoff
live on the flows' tasks (src/ingest/); failed runs are visible + retryable in
the Prefect Cloud UI.

INGEST_DEPLOYMENTS is the single source of truth for which kinds can *run*. The
dispatcher reads it to decide what to admit, so a kind whose flow does not exist
yet waits as `pending` instead of being handed to the wrong flow.

It is not the source of truth for everything else about a kind: registration
validates against DOC_KINDS (api/admin.py), the chunk floor lives with the
chunker, and the citation locator with the search layer. Adding "deck" touched
all four. A kinds table would collapse them, and the reason there isn't one is
that at three kinds the guarded duplicates fail loudly (worker.py's
_check_deployments_match, a 400 at registration) — only the chunk floor and the
locator fail silently. Revisit when a fourth kind arrives.
"""
from __future__ import annotations

from prefect.client.orchestration import get_client
from prefect.client.schemas.filters import (FlowRunFilter, FlowRunFilterName,
                                            FlowRunFilterState,
                                            FlowRunFilterStateType)
from prefect.client.schemas.objects import StateType
from prefect.deployments import run_deployment
from prefect.exceptions import ObjectNotFound
from prefect.states import Cancelled

from . import db

# What "still worth cancelling" excludes. Written as the complement so a state
# type Prefect adds later is treated as live — cancelling something already
# finished is a wasted call, missing something live is a wasted worker slot.
_TERMINAL_RUN_STATES = (StateType.COMPLETED, StateType.FAILED,
                        StateType.CANCELLED, StateType.CRASHED)

INGEST_DEPLOYMENTS = {
    "video": "ms-ingest-video/ingest",
    "paper": "ms-ingest-paper/ingest",
    "deck": "ms-ingest-deck/ingest",
}


def dispatchable_kinds() -> tuple[str, ...]:
    """Kinds an ingest flow exists for — what the dispatcher may claim."""
    return tuple(INGEST_DEPLOYMENTS)


def enqueue(source_id: str, user_id: str, kind: str) -> str:
    """Schedule the ingest flow for one source. Returns the Prefect flow-run id."""
    deployment = INGEST_DEPLOYMENTS.get(kind)
    if deployment is None:
        raise ValueError(f"no ingest deployment for kind {kind!r}")
    flow_run = run_deployment(
        name=deployment,
        # The parameter is still called video_id: both flows take it, and the
        # manifest's primary key has not been renamed yet (see src/db/).
        parameters={"video_id": source_id, "user_id": user_id},
        timeout=0,  # fire-and-forget: don't block the API waiting for the run
        flow_run_name=f"ingest-{source_id}",
    )
    # Remember what we asked for, here rather than in each caller: four call
    # sites schedule runs (dispatcher, both register paths, retry) and the one
    # that forgot would leave an uncancellable run behind — the exact failure
    # this column exists to prevent.
    #
    # Swallowing a failure here is deliberate, and it is the safer direction of
    # the two. The run is ALREADY scheduled by this point; letting a database
    # hiccup propagate would make enqueue() look failed to the dispatcher, which
    # puts the row back to `pending` and admits it again — a second run on a
    # source that already has one. Losing the note costs a cancel we cannot make
    # (and cancel_ingest can still find the run by name); losing the row costs a
    # duplicate ingest.
    try:
        db.set_flow_run(source_id, str(flow_run.id))
    except Exception as exc:  # noqa: BLE001 — see above
        print(f"[jobs] scheduled {flow_run.id} for {source_id} but could not "
              f"record it: {type(exc).__name__}: {exc}")
    return str(flow_run.id)


def cancel(flow_run_id: str) -> bool:
    """Call off a scheduled run. True if Prefect accepted the transition.

    Cancelling is a request, not a guarantee: a run already executing keeps
    going until it next checks its state, and one that has finished cannot be
    cancelled at all. Both are fine here — the point is the run that has not
    started yet, which is the one still holding a worker slot for a source that
    no longer exists. The row is deleted right after this, so a run that does
    slip through still dies on LostOwnership, as it did before.

    Never raises. This runs inside DELETE, where the user asked for the source
    to be gone; an unreachable Prefect Cloud must not turn that into a 500.
    """
    try:
        with get_client(sync_client=True) as client:
            client.set_flow_run_state(flow_run_id, Cancelled())
        return True
    except ObjectNotFound:
        return False  # already gone from Prefect's side — nothing to call off
    except Exception as exc:  # noqa: BLE001 — see docstring: delete must survive
        print(f"[jobs] could not cancel {flow_run_id}: {type(exc).__name__}: {exc}")
        return False


def cancel_ingest(source_id: str, flow_run_id: str | None = None) -> int:
    """Call off whatever is scheduled for this source. Returns runs cancelled.

    Two handles, because the recorded id has a hole: enqueue() can only write it
    AFTER run_deployment() returns, and a source deleted inside that window (the
    dispatcher claims, the network round trip runs, the delete lands) leaves a
    run nobody wrote down. The run NAME is deterministic and exists from the
    moment Prefect knows the run, so it closes that window — at the cost of a
    query, which is why it is the fallback and not the default path.

    Looking up by name also catches more than one live run for a source, which
    is what a reap-then-readmit leaves behind: the new run is recorded on the
    row, the old one is only reachable by name.
    """
    if flow_run_id:
        return 1 if cancel(flow_run_id) else 0
    try:
        with get_client(sync_client=True) as client:
            runs = client.read_flow_runs(flow_run_filter=FlowRunFilter(
                name=FlowRunFilterName(any_=[f"ingest-{source_id}"]),
                state=FlowRunFilterState(
                    type=FlowRunFilterStateType(not_any_=list(_TERMINAL_RUN_STATES))),
            ))
            for run in runs:
                client.set_flow_run_state(run.id, Cancelled())
            return len(runs)
    except Exception as exc:  # noqa: BLE001 — same reason as cancel()
        print(f"[jobs] could not look up runs for {source_id}: "
              f"{type(exc).__name__}: {exc}")
        return 0
