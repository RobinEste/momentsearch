"""Read path: question -> retrieve -> gate -> cited answer (or honest abstain).

Retrieval is milliseconds; the multimodal LLM call is seconds and dominates
cost. So the shape is a confidence funnel: fetch KNN_K candidates, collapse
temporal near-duplicates, trim to TOP_K, and — Gate 1 — if even the best
score is below CONFIDENCE_THRESHOLD, abstain WITHOUT calling the LLM. That
one free check kills most hallucination risk. Generated answers get their
[n] citations validated; invented references are stripped.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .. import db, llm, storage
from ..config import CONFIDENCE_THRESHOLD, KNN_K, TOP_K
from . import vector_store
from .embeddings import embed_text

# Two hits from the same video within this window are the same moment.
_NEAR_MS = 5000

ABSTAIN = ("I couldn't find that in your videos — nothing indexed looks "
           "visually related to the question.")


def _seconds(ms: int) -> str:
    s = ms // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


def _dedupe_moments(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the best-scoring hit per (video, ~timestamp) — the cheap 'rerank'
    that decides which few frames are worth multimodal-LLM money."""
    kept: list[dict[str, Any]] = []
    for h in hits:  # hits arrive score-descending
        dup = any(k["video_id"] == h["video_id"] and abs(k["ms"] - h["ms"]) < _NEAR_MS
                  for k in kept)
        if not dup:
            kept.append(h)
    return kept


def _deeplink(video: dict | None, video_id: str, ms: int) -> str:
    secs = ms // 1000
    if video and video.get("source") == "youtube" and video.get("url"):
        sep = "&" if "?" in video["url"] else "?"
        return f"{video['url']}{sep}t={secs}"
    return f"/api/video/{video_id}#t={secs}"


def _thumb_url(user_id: str, video_id: str, idx: int) -> str:
    """Browser-facing thumbnail URL. Presigned GET straight to the bucket when
    the provider supports it (an <img> tag can't send auth headers); the API
    serves the bytes itself only in local-dev mode."""
    if storage.presign_capable():
        return storage.presign_get(storage.frame_key(user_id, video_id, idx))
    return f"/api/frame/{video_id}/{idx:06d}.jpg?u={user_id}"


def _media_url(video: dict | None, user_id: str, video_id: str) -> str | None:
    """Playback URL for uploaded videos (YouTube plays via its own URL)."""
    if not video or video.get("source") != "upload" or not video.get("storage_key"):
        return None
    if storage.presign_capable():
        return storage.presign_get(video["storage_key"])
    return f"/api/video/{video_id}?u={user_id}"


def retrieve(question: str, user_id: str, *, top_k: int | None = None,
             video_id: str | None = None,
             video_ids: list[str] | None = None) -> list[dict[str, Any]]:
    """Top moments for a question, as numbered citations (metadata from Postgres).

    video_ids scopes the search to a chosen subset (the UI's select/unselect) —
    e.g. unselect the samples to query only your own uploads."""
    k = top_k or TOP_K
    qvec = embed_text(question)
    hits = vector_store.search(qvec, user_id, top_k=max(KNN_K, k),
                               video_id=video_id, video_ids=video_ids)
    hits = _dedupe_moments(hits)[:k]
    videos = db.videos_by_ids(sorted({h["video_id"] for h in hits}))
    citations = []
    for i, h in enumerate(hits, 1):
        vid = h["video_id"]
        meta = videos.get(vid)
        ms = int(h.get("ms", 0))
        idx = int(h.get("idx", 0))
        citations.append({
            "n": i,
            "video_id": vid,
            "title": (meta or {}).get("title") or vid,
            "url": (meta or {}).get("url"),
            "source": (meta or {}).get("source"),
            "ms": ms,
            "timestamp": _seconds(ms),
            "idx": idx,
            "thumbnail": _thumb_url(user_id, vid, idx),
            "media_url": _media_url(meta, user_id, vid),
            "deeplink": _deeplink(meta, vid, ms),
            "score": round(h.get("score", 0.0), 4),
        })
    return citations


def _fallback_answer(citations: list[dict[str, Any]]) -> str:
    """No-LLM summary: rank the visually-closest moments. Honest about being
    similarity, not synthesis."""
    top = citations[0]
    where = f"{top['title']} at {top['timestamp']}" if top.get("title") else top["timestamp"]
    others = ", ".join(f"{c['timestamp']} [{c['n']}]" for c in citations[1:4])
    msg = f"Closest visual match: {where} [{top['n']}] (similarity {top['score']})."
    if others:
        msg += f" Other relevant moments: {others}."
    return msg


_CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _validate_citations(answer: str, n_frames: int) -> str:
    """Strip invented [n] references the model has no frame for."""
    def fix(m: re.Match) -> str:
        nums = [int(x) for x in re.split(r"\s*,\s*", m.group(1))]
        valid = [str(x) for x in nums if 1 <= x <= n_frames]
        return f"[{', '.join(valid)}]" if valid else ""
    return _CITE_RE.sub(fix, answer)


def _fetch_frames(user_id: str, citations: list[dict[str, Any]]) -> list[bytes]:
    keys = [storage.frame_key(user_id, c["video_id"], c["idx"]) for c in citations]
    with ThreadPoolExecutor(max_workers=6) as ex:
        return list(ex.map(storage.get_bytes, keys))


def resolve_llm(user_id: str) -> tuple[llm.LLMConfig | None, str]:
    """Which model answers for this tenant: their own hosted endpoint
    (ms_user_llms — e.g. a vLLM server) first, the server-wide LLM_* env
    config as fallback. Returns (config, source) with source in
    {"user", "server", "none"}."""
    row = db.get_user_llm(user_id)
    if row and row.get("model"):
        return llm.from_row(row), "user"
    cfg = llm.env_config()
    return (cfg, "server") if cfg else (None, "none")


def ask(question: str, user_id: str, *, top_k: int | None = None,
        video_id: str | None = None,
        video_ids: list[str] | None = None) -> dict[str, Any]:
    citations = retrieve(question, user_id, top_k=top_k, video_id=video_id,
                         video_ids=video_ids)
    result: dict[str, Any] = {"question": question, "citations": citations}

    if not citations:
        result.update(answer="No relevant moments were found. Try ingesting a video first.",
                      llm_used=False, abstained=True)
        return result

    # Gate 1 — retrieval confidence. Below threshold, no LLM call at all.
    if CONFIDENCE_THRESHOLD and citations[0]["score"] < CONFIDENCE_THRESHOLD:
        result.update(answer=ABSTAIN, llm_used=False, abstained=True)
        return result

    cfg, source = resolve_llm(user_id)
    if cfg is None:
        # CLIP is an embedding model — it can't write prose — so instead of
        # inventing an answer we summarize the best matches by similarity.
        result.update(answer=_fallback_answer(citations), llm_used=False,
                      note=("Visual-similarity results only. Connect your own "
                            "model (vLLM/Ollama/API) in settings, or set "
                            "LLM_API_KEY on the server, for a synthesized, "
                            "frame-grounded answer."))
        return result

    frames = _fetch_frames(user_id, citations)
    result["answer"] = _validate_citations(llm.answer(question, frames, cfg),
                                           len(citations))
    result["llm_used"] = True
    result["llm_source"] = source          # "user" = their own hosted model
    result["llm_model"] = cfg.model
    return result
