"""CLIP embeddings — the heart of the *visual* search. "Embedding is a URL."

A single CLIP model encodes both video frames (images) and text queries into
the same vector space, so a natural-language question can be matched directly
against what is *seen* on screen. No transcription, no audio.

Two modes, switched by CLIP_SERVICE_URL:

  set    -> remote: batches go to the warm clip_service.py container (model
            loaded ONCE at its boot). Workers and the API stay light — no
            torch import, no per-video model reload. Scaling embedding =
            scaling that one service; point the URL at a GPU machine later.
  unset  -> local: the model loads lazily in this process (simple mode — no
            extra service; fine for quickstart and single-machine dev).

The *_local functions are the actual inference; clip_service.py serves them.
"""
from __future__ import annotations

import base64
import io
import json
import threading
import time
import urllib.error
import urllib.request
from functools import lru_cache

import numpy as np

from .. import config

_lock = threading.Lock()


# ── Local inference (used in-process, and by clip_service.py) ────────────────

@lru_cache
def _model():
    # Imported lazily so processes in remote mode never drag in torch.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.CLIP_MODEL)


@lru_cache
def embedding_dim() -> int:
    return int(_model().get_sentence_embedding_dimension())


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def embed_jpegs_local(jpegs: list[bytes]) -> np.ndarray:
    """Encode in-memory JPEGs into L2-normalized CLIP vectors (local model)."""
    from PIL import Image

    if not jpegs:
        return np.zeros((0, embedding_dim()), dtype=np.float32)
    images = [Image.open(io.BytesIO(b)).convert("RGB") for b in jpegs]
    try:
        with _lock:  # sentence-transformers models are not thread-safe
            vecs = _model().encode(images, convert_to_numpy=True, batch_size=32,
                                   show_progress_bar=False)
    finally:
        for img in images:
            img.close()
    return _normalize(np.asarray(vecs, dtype=np.float32))


def embed_text_local(text: str) -> np.ndarray:
    """Encode a text query into the shared CLIP space (local model)."""
    with _lock:
        vec = _model().encode([text], convert_to_numpy=True, show_progress_bar=False)
    return _normalize(np.asarray(vec, dtype=np.float32))[0]


# ── Remote inference (CLIP_SERVICE_URL set) ──────────────────────────────────

def _post(path: str, payload: dict, timeout: int = 600) -> dict:
    """POST to the clip service, retrying while it warms up at boot (the model
    load takes ~30s; workers may start first)."""
    url = config.CLIP_SERVICE_URL + path
    body = json.dumps(payload).encode()
    last: Exception | None = None
    for _ in range(12):  # up to ~60s of patience for a cold service
        try:
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.URLError as exc:
            last = exc
            time.sleep(5)
    raise RuntimeError(f"CLIP service unreachable at {config.CLIP_SERVICE_URL}: {last}")


# ── Public API (mode dispatch) ────────────────────────────────────────────────

def embed_jpegs(jpegs: list[bytes]) -> np.ndarray:
    if config.CLIP_SERVICE_URL:
        if not jpegs:
            with urllib.request.urlopen(config.CLIP_SERVICE_URL + "/healthz",
                                        timeout=60) as resp:
                dim = json.loads(resp.read())["dim"]
            return np.zeros((0, dim), dtype=np.float32)
        vecs = _post("/embed/images", {
            "jpegs_b64": [base64.b64encode(j).decode() for j in jpegs]})["vectors"]
        return np.asarray(vecs, dtype=np.float32)
    return embed_jpegs_local(jpegs)


def embed_text(text: str) -> np.ndarray:
    if config.CLIP_SERVICE_URL:
        vec = _post("/embed/text", {"text": text}, timeout=60)["vector"]
        return np.asarray(vec, dtype=np.float32)
    return embed_text_local(text)
