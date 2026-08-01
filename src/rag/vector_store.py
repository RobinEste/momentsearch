"""Qdrant — one shared multi-tenant collection, one point per kept frame.

Multi-tenancy: every point carries user_id; the field has a tenant payload
index and every search / upsert / delete is user_id-filtered. NOT
collection-per-user (collection explosion); a huge tenant can graduate to a
dedicated collection later.

Memory profile (the frame-scale levers, all env flags, default ON):
  QDRANT_ON_DISK        original float vectors live on disk
  QDRANT_QUANTIZATION   int8 copies pinned in RAM (~4x smaller) do the search;
                        queries rescore the top candidates from the originals
  QDRANT_HNSW_ON_DISK   the HNSW graph lives on disk too

Point IDs are uuid5 of "{video_id}:{frame_idx}" — deterministic, so re-runs
overwrite instead of duplicating. Payloads are trimmed to filter/display
fields (user_id, video_id, ms, idx, embed_version); titles and URLs live in
Postgres and are joined at answer time.
"""
from __future__ import annotations

import uuid
from typing import Any, Iterable

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from ..config import (
    CLIP_DIM,
    CLIP_MODEL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_HNSW_ON_DISK,
    QDRANT_LOCAL_PATH,
    QDRANT_ON_DISK,
    QDRANT_QUANTIZATION,
    QDRANT_URL,
    TEXT_COLLECTION,
    TEXT_EMBED_DIM,
)

_client: QdrantClient | None = None

# Dimensions of the stock sentence-transformers CLIP checkpoints — lets the
# API create the collection at boot without pulling in torch or downloading
# the model. Unknown/custom models: set CLIP_DIM, or the model gets loaded.
_KNOWN_DIMS = {
    "clip-ViT-B-32": 512,
    "clip-ViT-B-16": 512,
    "clip-ViT-L-14": 768,
    "clip-ViT-L-14-336": 768,
}


def _dim() -> int:
    if CLIP_DIM:
        return CLIP_DIM
    if CLIP_MODEL in _KNOWN_DIMS:
        return _KNOWN_DIMS[CLIP_MODEL]
    from .embeddings import embedding_dim  # last resort — loads the model

    return embedding_dim()


def client() -> QdrantClient:
    global _client
    if _client is None:
        if QDRANT_URL:
            _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None,
                                   timeout=60)
        else:  # embedded local instance — dev only, single-process
            _client = QdrantClient(path=QDRANT_LOCAL_PATH)
    return _client


def point_id(video_id: str, frame_idx: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{video_id}:{frame_idx}"))


def _user_filter(user_id: str, video_id: str | None = None,
                 video_ids: list[str] | None = None) -> qm.Filter:
    must: list[qm.FieldCondition] = [
        qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id))]
    if video_id:  # single-video scope (kept for /transcript-style calls)
        must.append(qm.FieldCondition(key="video_id", match=qm.MatchValue(value=video_id)))
    elif video_ids:  # multi-select scope — query only the chosen videos
        must.append(qm.FieldCondition(key="video_id", match=qm.MatchAny(any=video_ids)))
    return qm.Filter(must=must)


def _ensure(collection: str, dim: int, range_fields: tuple[str, ...] = ()) -> None:
    """Create a collection (low-RAM profile) + tenant/video payload indexes.

    `range_fields` get a FLOAT index. Qdrant does not merely run a range filter
    slower without one — it refuses with a 400 ("Index required but not found"),
    so a range condition and its index are one decision, not two."""
    c = client()
    if not c.collection_exists(collection):
        c.create_collection(
            collection_name=collection,
            vectors_config=qm.VectorParams(
                size=dim,
                distance=qm.Distance.COSINE,
                on_disk=QDRANT_ON_DISK,
            ),
            hnsw_config=qm.HnswConfigDiff(on_disk=QDRANT_HNSW_ON_DISK),
            quantization_config=(
                qm.ScalarQuantization(scalar=qm.ScalarQuantizationConfig(
                    type=qm.ScalarType.INT8, always_ram=True))
                if QDRANT_QUANTIZATION else None
            ),
        )
    # Tenant index on user_id: co-locates a tenant's points so per-user
    # searches touch a small slice of the index. video_id for delete/filter.
    try:
        c.create_payload_index(
            collection_name=collection, field_name="user_id",
            field_schema=qm.KeywordIndexParams(type=qm.KeywordIndexType.KEYWORD,
                                               is_tenant=True))
    except Exception:  # older server without is_tenant, or index already exists
        try:
            c.create_payload_index(collection_name=collection, field_name="user_id",
                                   field_schema=qm.PayloadSchemaType.KEYWORD)
        except Exception:
            pass
    try:
        c.create_payload_index(collection_name=collection, field_name="video_id",
                               field_schema=qm.PayloadSchemaType.KEYWORD)
    except Exception:
        pass
    for field in range_fields:
        try:
            c.create_payload_index(collection_name=collection, field_name=field,
                                   field_schema=qm.PayloadSchemaType.FLOAT)
        except Exception as exc:  # noqa: BLE001
            # Not silent: "already exists" is routine, a rejected schema is not,
            # and the two look identical from here. The consequence of the second
            # one is a feature that quietly never works, so it gets a line.
            print(f"[qdrant] payload index {collection}.{field}: {str(exc)[:100]}")


def ensure_collection() -> None:
    """Visual (CLIP frame) collection."""
    _ensure(QDRANT_COLLECTION, _dim())


def ensure_text_collection() -> None:
    """Transcript (bge text) collection — the second branch.

    `t_start`/`t_end` are indexed here and not on the frame collection: the
    time-window lookup (`text_covering`) only ever reads transcript chunks, and
    an index nobody filters on is memory for nothing."""
    _ensure(TEXT_COLLECTION, TEXT_EMBED_DIM, range_fields=("t_start", "t_end"))


def upsert_frames(user_id: str, video_id: str, ids: Iterable[int],
                  vectors: np.ndarray, payloads: list[dict[str, Any]]) -> None:
    points = [
        qm.PointStruct(id=point_id(video_id, idx), vector=vec.tolist(), payload=payload)
        for idx, vec, payload in zip(ids, vectors, payloads)
    ]
    if points:
        client().upsert(collection_name=QDRANT_COLLECTION, points=points, wait=True)


def search(vector: np.ndarray, user_id: str, *, top_k: int,
           video_id: str | None = None,
           video_ids: list[str] | None = None) -> list[dict[str, Any]]:
    try:
        hits = client().query_points(
            collection_name=QDRANT_COLLECTION,
            query=vector.tolist(),
            limit=top_k,
            query_filter=_user_filter(user_id, video_id, video_ids),
            with_payload=True,
            search_params=qm.SearchParams(
                # Quantized search is lossy; rescore re-reads the full-precision
                # vectors from disk for the top candidates.
                quantization=qm.QuantizationSearchParams(rescore=True)
                if QDRANT_QUANTIZATION else None,
            ),
        ).points
    except Exception as exc:
        # Empty deployment (collection not created yet) is a "no results"
        # situation, not a 500 — the UI shows "no moments found".
        if "doesn't exist" in str(exc) or "Not found" in str(exc):
            return []
        raise
    return [{"score": float(h.score), **(h.payload or {})} for h in hits]


# ── Transcript (text) branch ─────────────────────────────────────────────────

def upsert_chunks(user_id: str, video_id: str, vectors: np.ndarray,
                  payloads: list[dict[str, Any]], *, ns: str = "text",
                  start: int = 0) -> None:
    """Text chunks into the text collection. IDs are uuid5 of
    '<video_id>:<ns>:<i>' so re-runs overwrite, and never collide with frame ids.

    `ns` keeps the source types that share this collection in separate id
    namespaces: transcript chunks are 'text', document pages are 'page'.

    `start` offsets the index for callers that upsert in batches. Without it
    every batch would restart at 0 and silently overwrite the previous one —
    the kind of bug that shows up as "my document only has 64 chunks".
    """
    points = [
        qm.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL,
                                         f"{video_id}:{ns}:{start + i}")),
                       vector=vec.tolist(), payload=payload)
        for i, (vec, payload) in enumerate(zip(vectors, payloads))
    ]
    if points:
        client().upsert(collection_name=TEXT_COLLECTION, points=points, wait=True)


def search_text(vector: np.ndarray, user_id: str, *, top_k: int,
                video_id: str | None = None,
                video_ids: list[str] | None = None) -> list[dict[str, Any]]:
    try:
        hits = client().query_points(
            collection_name=TEXT_COLLECTION,
            query=vector.tolist(),
            limit=top_k,
            query_filter=_user_filter(user_id, video_id, video_ids),
            with_payload=True,
            search_params=qm.SearchParams(
                quantization=qm.QuantizationSearchParams(rescore=True)
                if QDRANT_QUANTIZATION else None,
            ),
        ).points
    except Exception as exc:
        if "doesn't exist" in str(exc) or "Not found" in str(exc):
            return []
        raise
    return [{"score": float(h.score), **(h.payload or {})} for h in hits]


def text_covering(user_id: str, video_id: str, seconds: float) -> str | None:
    """The transcript chunk whose span contains `seconds`, or None.

    Filter-only, no vector: "which chunk covers this instant" is a range
    question, not a similarity one. It serves a frame-only citation — one the
    visual branch found and the text branch did not — the words that were
    actually being spoken there, so the citation rests on indexed text instead
    of on nothing. `limit=1`: chunk spans do not overlap, so there is at most one.

    Documents share this collection but cannot match: they carry no `t_start` /
    `t_end`, so the range conditions exclude them.
    """
    flt = qm.Filter(must=[
        qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id)),
        qm.FieldCondition(key="video_id", match=qm.MatchValue(value=video_id)),
        qm.FieldCondition(key="t_start", range=qm.Range(lte=seconds)),
        qm.FieldCondition(key="t_end", range=qm.Range(gte=seconds)),
    ])
    try:
        points, _ = client().scroll(collection_name=TEXT_COLLECTION,
                                    scroll_filter=flt, limit=1, with_payload=True)
    except Exception as exc:
        # Degrade, never take the search down with it: this lookup only ever
        # enriches a citation that would otherwise carry no words. A missing
        # collection, or a collection indexed before `t_start`/`t_end` were,
        # costs a borrowed passage -- not the answer.
        msg = str(exc)
        if "doesn't exist" in msg or "Not found" in msg or "Index required" in msg:
            print(f"[text_covering] skipped for {video_id}: {msg[:120]}")
            return None
        raise
    return ((points[0].payload or {}).get("text") or None) if points else None


def delete_video(user_id: str, video_id: str) -> None:
    """Purge a video from BOTH branches (frames + transcript)."""
    sel = qm.FilterSelector(filter=_user_filter(user_id, video_id))
    for coll in (QDRANT_COLLECTION, TEXT_COLLECTION):
        try:
            client().delete(collection_name=coll, points_selector=sel, wait=True)
        except Exception:
            pass  # text collection may not exist if transcript is disabled


def collection_ready() -> bool:
    try:
        return client().collection_exists(QDRANT_COLLECTION)
    except Exception:
        return False
