"""CLIP inference service — one warm model behind a URL.

    uvicorn src.clip_service:app --host 0.0.0.0 --port 8001

The model loads ONCE at boot and stays hot; workers and the API send batches
instead of each flow-run subprocess paying a fresh torch import + weight load
(~15-30s per video). This is the standard model-serving pattern (TEI / Triton
/ OpenAI-embeddings-shaped): inference is a URL, so scaling embedding means
scaling THIS one service — today a CPU container, later the same container on
a GPU machine — while workers stay cheap and stateless.

Wire-up: set CLIP_SERVICE_URL=http://clip:8001 on api + worker (docker-compose
does this by default). Unset, they embed in-process — simple mode, no service.

Endpoints:
  POST /embed/images  {"jpegs_b64": [...]}  -> {"vectors": [[...], ...]}
  POST /embed/text    {"text": "..."}       -> {"vector": [...]}
  GET  /healthz                             -> {"ok", "model", "dim"}
"""
from __future__ import annotations

import base64
import contextlib
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from . import config
from .config import CLIP_MODEL
from .rag import embeddings

# Batch work yields to interactive work.
#
# Every endpoint here is a sync `def`, so FastAPI runs it in a threadpool, and
# torch inside it helps itself to every core. One ingest flow posting a whole
# document's chunks is one long op holding those cores; four flows are four,
# oversubscribing a 12-core box, and a search needing a 64ms query embed then
# waits for a scheduling slice rather than for compute. Measured here with a
# 20-document backfill in flight: embed_text on the read path went from a 64ms
# idle median to 2788ms (43x), peaking at 20397ms (318x). A 318x factor is not
# CPU sharing, it is queueing.
#
# The textbook fix is a second replica for the read path, and it does not fit:
# this container holds ~4.0GiB of the 7.65GiB Docker has here, so a twin OOMs.
# Capacity cannot be bought, only allocated — hence a queue discipline instead
# of a copy. The batch endpoints take a slot; /embed/text and /embed/query never
# do, so an interactive request is admitted while ingest waits.
#
# A fairness knob, not a throughput one: it bounds how much of the service
# ingest may hold at once, and the ingest gate is what says whether that cost is
# acceptable. Re-measure BOTH sides when changing it.
_BATCH_SLOTS = (threading.Semaphore(config.CLIP_BATCH_CONCURRENCY)
                if config.CLIP_BATCH_CONCURRENCY > 0 else None)


def _batch_slot():
    """Hold a batch slot for a block (a no-op when the limit is disabled)."""
    return _BATCH_SLOTS if _BATCH_SLOTS is not None else contextlib.nullcontext()


@asynccontextmanager
async def lifespan(app: FastAPI):
    dim = embeddings.embedding_dim()  # load CLIP NOW, not on first request
    print(f"[clip] {CLIP_MODEL} warm (dim {dim})")
    # Only warm the local bge model when it's actually the text provider; with
    # TEXT_EMBED_PROVIDER=openai the transcript branch calls OpenAI directly and
    # never touches this service.
    if config.ENABLE_TRANSCRIPT and config.TEXT_EMBED_PROVIDER != "openai":
        embeddings.embed_docs_local(["warmup"])  # load the bge text model too
        print(f"[clip] text model {config.TEXT_EMBED_MODEL} warm")
    yield


app = FastAPI(title="MomentSearch CLIP service", lifespan=lifespan)


class ImagesRequest(BaseModel):
    jpegs_b64: list[str]


class TextRequest(BaseModel):
    text: str


class DocsRequest(BaseModel):
    texts: list[str]


@app.get("/healthz")
def healthz():
    return {"ok": True, "model": CLIP_MODEL, "dim": embeddings.embedding_dim()}


# ── Batch endpoints (ingest) — these take a slot ─────────────────────────────

@app.post("/embed/images")
def embed_images(req: ImagesRequest):
    jpegs = [base64.b64decode(j) for j in req.jpegs_b64]
    with _batch_slot():
        return {"vectors": embeddings.embed_jpegs_local(jpegs).tolist()}


# ── Interactive endpoints (the read path) — never blocked by a batch ─────────

@app.post("/embed/text")
def embed_text(req: TextRequest):
    return {"vector": embeddings.embed_text_local(req.text).tolist()}


# ── Transcript branch (bge semantic text) ────────────────────────────────────

@app.post("/embed/docs")
def embed_docs(req: DocsRequest):
    with _batch_slot():
        return {"vectors": embeddings.embed_docs_local(req.texts).tolist()}


@app.post("/embed/query")
def embed_query(req: TextRequest):
    return {"vector": embeddings.embed_query_local(req.text).tolist()}


@app.post("/count_tokens")
def count_tokens(req: DocsRequest):
    """Token counts for the text model, so the chunker can stay under its limit.

    This lives here because this service owns the model, and the model owns the
    limit: bge truncates at 512 tokens without raising, so a chunk sized by a
    character budget can lose its tail in silence. Measured on real papers, the
    chars-per-token ratio swings from 5.2 (prose) to 2.0 (tables and formulas),
    which is why no character cap can guarantee that invariant.
    """
    return {"counts": embeddings.count_tokens_local(req.texts),
            "limit": embeddings.text_token_limit()}
