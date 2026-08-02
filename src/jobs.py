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

from prefect.deployments import run_deployment

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
    return str(flow_run.id)
