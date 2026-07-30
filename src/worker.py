"""Ingest worker entrypoint — serves the Prefect flows.

    python -m src.worker

serve() registers the "ms-ingest-video/ingest" and "ms-ingest-paper/ingest"
deployments in Prefect Cloud (idempotent) and long-polls for scheduled runs —
outbound HTTPS only, no ports. Scale horizontally by running more replicas of
this process; each executes up to WORKER_CONCURRENCY runs at once, shared
across both deployments — capacity belongs to the machine, not to a source type.

Sample seeding is NOT done here — it's a one-shot startup gate (seed.py /
src/seeding.py) that the whole stack waits on, so the app never serves a
half-indexed corpus. This worker only handles user uploads + YouTube adds.

Embedding goes to the warm CLIP service when CLIP_SERVICE_URL is set
(docker-compose default); unset, each run loads the model in-process.
"""
import os
import time

from prefect import serve
from prefect.deployments.runner import EntrypointType

from . import jobs
from .db import init_schema
from .ingest.paper import ingest_paper
from .ingest.pipeline import ingest_video

# The flows themselves cannot live in jobs.py: the API imports that module, and
# importing a flow drags torch and pypdfium2 into the API process. So the kind ->
# flow mapping lives here and the kind -> deployment-name mapping lives there,
# and _check_deployments_match() below makes a disagreement between them a loud
# boot failure. Without that check, a kind added to jobs.INGEST_DEPLOYMENTS but
# forgotten here would be claimed by the dispatcher, fail to enqueue, get put
# back to `pending`, and be claimed again — a silent loop every tick.
FLOWS = {"video": ingest_video, "paper": ingest_paper}


def _check_deployments_match() -> None:
    served = {f"{flow.name}/ingest" for flow in FLOWS.values()}
    declared = set(jobs.INGEST_DEPLOYMENTS.values())
    if served != declared or set(FLOWS) != set(jobs.INGEST_DEPLOYMENTS):
        raise RuntimeError(
            "worker.FLOWS and jobs.INGEST_DEPLOYMENTS disagree: "
            f"serving {sorted(served)} for kinds {sorted(FLOWS)}, "
            f"but jobs declares {sorted(declared)} for kinds "
            f"{sorted(jobs.INGEST_DEPLOYMENTS)}")


def main():
    _check_deployments_match()
    init_schema()  # make sure migrations ran before consuming runs
    from .rag import vector_store
    vector_store.ensure_collection()  # up front, not mid-first-ingest
    # Fair scheduler (WFQ): admits pending videos round-robin across users so
    # one bulk uploader can't starve everyone else (src/dispatcher.py).
    from . import dispatcher
    dispatcher.start_in_background()
    limit = int(os.getenv("WORKER_CONCURRENCY", "2"))
    # serve() talks to Prefect Cloud on startup; a transient outage (e.g. a 503)
    # used to crash the worker permanently and stop the machine. Self-heal:
    # retry forever so a blip pauses ingest instead of killing the worker.
    while True:
        try:
            print("[worker] serving deployments 'ms-ingest-video/ingest' and "
                  f"'ms-ingest-paper/ingest' (shared concurrency {limit})")
            # MODULE_PATH, not the default FILE_PATH. serve() re-imports the
            # flow in the run subprocess, and a file-path entrypoint executes
            # src/ingest/paper.py as a standalone script — at which point
            # `from .. import config, db` raises "attempted relative import
            # beyond top-level package". A module entrypoint
            # (src.ingest.paper:ingest_paper) keeps the package context.
            serve(*[flow.to_deployment(name="ingest",
                                       entrypoint_type=EntrypointType.MODULE_PATH)
                    for flow in FLOWS.values()],
                  limit=limit, print_starting_message=False)
            break  # clean shutdown
        except KeyboardInterrupt:
            break
        except Exception as exc:
            print(f"[worker] serve crashed: {type(exc).__name__}: {exc} — retrying in 15s")
            time.sleep(15)


if __name__ == "__main__":
    main()
