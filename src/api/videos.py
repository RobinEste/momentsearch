"""Video registration API — the write path's front door (Bearer auth).

Upload flow (gigabytes never touch this process):
  1. POST /api/videos/presign   -> scoped, time-limited PUT URL (server picks
                                   the key: uploads/{user}/{id}.{ext})
  2. browser PUTs the file straight to object storage
  3. POST /api/videos           -> HEAD-verify the object, insert a pending
                                   Postgres row, schedule a Prefect run, 202

YouTube flow: POST /api/videos {"url": ...} — the worker downloads it.

Every request is tenant-scoped by the X-User-Id header (default "default");
swap that for real per-user auth later — keys, rows and vectors are already
user_id-tagged.
"""
from __future__ import annotations

import re
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel

from .. import config, db, jobs, storage
from ..samples import is_sample
from ..config import (
    ADMIN_TOKEN,
    ALLOWED_UPLOAD_TYPES,
    DEFAULT_USER_ID,
    MAX_UPLOAD_MB,
    UPLOAD_KEY_PREFIX,
)
from ..rag import vector_store

router = APIRouter(prefix="/api/videos", tags=["videos"])

_USER_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_EXT_RE = re.compile(r"^\.[A-Za-z0-9]{1,8}$")
_YT_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?v=|shorts/|live/|embed/)|youtu\.be/)([A-Za-z0-9_-]{11})")


def require_auth(authorization: str | None = Header(default=None)) -> None:
    if not ADMIN_TOKEN:  # dev convenience — set ADMIN_TOKEN in any real deploy
        return
    if authorization != f"Bearer {ADMIN_TOKEN}":
        raise HTTPException(401, "Missing or invalid bearer token.")


def user_id(x_user_id: str | None = Header(default=None)) -> str:
    uid = (x_user_id or DEFAULT_USER_ID).strip()
    if not _USER_RE.match(uid):
        raise HTTPException(400, "Invalid X-User-Id.")
    return uid


# ── Presign ───────────────────────────────────────────────────────────────────

class PresignRequest(BaseModel):
    filename: str
    content_type: str
    size: int


@router.post("/presign", dependencies=[Depends(require_auth)])
def presign(req: PresignRequest, uid: str = Depends(user_id)):
    if req.size > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds the {MAX_UPLOAD_MB}MB limit.")
    if not any(req.content_type.startswith(t) for t in ALLOWED_UPLOAD_TYPES):
        raise HTTPException(415, "Only video uploads are accepted.")
    ext = Path(req.filename or "video.mp4").suffix.lower() or ".mp4"
    if not _EXT_RE.match(ext):
        ext = ".mp4"
    video_id = f"up_{uuid.uuid4().hex[:10]}"
    key = storage.upload_key(uid, video_id, ext)
    if not storage.presign_capable():
        # local-dev fallback: the API accepts the bytes itself
        return {"mode": "direct", "video_id": video_id, "key": key,
                "url": f"/api/videos/{video_id}/content?key={key}",
                "headers": {"Content-Type": req.content_type}}
    signed = storage.presign_put(key, req.content_type)
    return {"mode": "presigned", "video_id": video_id, "key": key, **signed}


@router.put("/{video_id}/content", dependencies=[Depends(require_auth)])
async def upload_direct(video_id: str, key: str, request: Request,
                        uid: str = Depends(user_id)):
    """Dev-only direct upload (STORAGE_PROVIDER=local can't presign)."""
    if storage.presign_capable():
        raise HTTPException(400, "Use the presigned URL to upload.")
    if not key.startswith(f"{UPLOAD_KEY_PREFIX}{uid}/{video_id}"):
        raise HTTPException(403, "Key does not belong to this upload.")
    dest = storage.local_path(key)
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with dest.open("wb") as out:
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 1024 * 1024:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"File exceeds the {MAX_UPLOAD_MB}MB limit.")
            out.write(chunk)
    return {"ok": True, "key": key, "size": size}


# ── Register (returns 202 instantly; a worker does the heavy lifting) ─────────

class RegisterRequest(BaseModel):
    url: str | None = None        # YouTube
    video_id: str | None = None   # upload (from /presign)
    key: str | None = None        # upload (from /presign)
    title: str | None = None


@router.post("", status_code=202, dependencies=[Depends(require_auth)])
def register(req: RegisterRequest, uid: str = Depends(user_id)):
    if req.url:
        # The scheme is checked separately because _YT_RE is not anchored: it
        # searches, so "javascript:alert(1)//youtube.com/watch?v=dQw4w9WgXcQ"
        # matches on the tail while the whole string — javascript: prefix and
        # all — is what gets stored, handed to _deeplink() and finally set as an
        # href by the UI. One click then runs it. Same 400 as before, so the
        # contract does not change; only input that could never be a video URL
        # is rejected.
        if urlsplit(req.url).scheme not in ("http", "https"):
            raise HTTPException(400, "Not a recognizable YouTube URL.")
        m = _YT_RE.search(req.url)
        if not m:
            raise HTTPException(400, "Not a recognizable YouTube URL.")
        video_id = f"yt_{m.group(1)}"
        row = db.upsert_pending({"id": video_id, "user_id": uid, "kind": "video",
                                 "source": "youtube",
                                 "url": req.url, "storage_key": None,
                                 "source_hash": video_id, "title": req.title})
    elif req.video_id and req.key:
        # Never trust the client's key: it must be the one WE minted for them.
        if not req.key.startswith(f"{UPLOAD_KEY_PREFIX}{uid}/{req.video_id}"):
            raise HTTPException(403, "Key does not belong to this user/upload.")
        meta = storage.head(req.key)
        if meta is None:
            raise HTTPException(404, "Object not found — did the upload finish?")
        if meta["size"] > MAX_UPLOAD_MB * 1024 * 1024:
            storage.delete_key(req.key)
            raise HTTPException(413, f"Object exceeds the {MAX_UPLOAD_MB}MB limit.")
        title = req.title or Path(req.key).stem
        row = db.upsert_pending({"id": req.video_id, "user_id": uid, "kind": "video",
                                 "source": "upload",
                                 "url": None, "storage_key": req.key,
                                 "source_hash": None, "title": title})
    else:
        raise HTTPException(400, "Provide either url (YouTube) or video_id+key (upload).")

    # `status` stays the documented "accepted for ingest"; `current_status` is
    # what the row actually says, which can differ now that upsert_pending leaves
    # an in-flight row alone (see admin.py, same shape).
    accepted = {"video_id": row["id"], "status": "pending",
                **({"current_status": row["status"]}
                   if row["status"] != "pending" else {})}
    # Fair dispatch (WFQ): leave it `pending` — the dispatcher admits it in fair
    # order (src/dispatcher.py). FIFO mode: enqueue to Prefect immediately, but
    # NOT when the guard just refused to reset it, or that is a second run on a
    # source a worker is already holding.
    if config.ENABLE_FAIR_DISPATCH or row["status"] not in config.NOT_INFLIGHT_STATUSES:
        return accepted
    return {**accepted, "flow_run_id": jobs.enqueue(row["id"], uid, row["kind"])}


# ── Status / lifecycle ─────────────────────────────────────────────────────────

_PUBLIC_FIELDS = ("id", "kind", "source", "url", "title", "status", "error",
                  "frame_count", "progress", "attempts", "created_at", "updated_at")


def _public(row: dict) -> dict:
    out = {k: row.get(k) for k in _PUBLIC_FIELDS}
    # Samples are protected: unselectable-yes, deletable-no. The UI hides the ✕
    # on these and the delete endpoint refuses them.
    out["is_sample"] = is_sample(row["id"])
    return out


@router.get("")
def list_videos(uid: str = Depends(user_id), status: str | None = None):
    return {"videos": [_public(r) for r in db.list_videos(uid, status=status)]}


@router.get("/{video_id}")
def get_video(video_id: str, uid: str = Depends(user_id)):
    row = db.get_video(video_id)
    if row is None or row["user_id"] != uid:
        raise HTTPException(404, "Video not found.")
    return _public(row)


@router.post("/{video_id}/retry", status_code=202, dependencies=[Depends(require_auth)])
def retry(video_id: str, uid: str = Depends(user_id)):
    # Re-queueing something a worker is busy with would start a SECOND flow on
    # the same source: two runs sharing one scratch file, and the delete at the
    # top of one landing between the upserts of the other. Only the reaper may
    # move an in-flight row, and only once it is provably dead. db.requeue
    # decides and writes in one statement, so there is no window between the two.
    row = db.requeue(video_id, uid)
    if row is None:
        # Only now pay for a read, to tell "not yours / no such id" (404) from
        # "yours, but busy" (409). The happy path stays at one round trip.
        existing = db.get_video(video_id)
        if existing is None or existing["user_id"] != uid:
            raise HTTPException(404, "Video not found.")
        raise HTTPException(
            409, f"Already in progress (status {existing['status']}). If its "
                 "worker went away it is requeued automatically.")
    if config.ENABLE_FAIR_DISPATCH:
        return {"video_id": video_id, "status": "pending"}  # dispatcher re-admits it fairly
    flow_run_id = jobs.enqueue(video_id, uid, row["kind"])
    return {"video_id": video_id, "status": "pending", "flow_run_id": flow_run_id}


@router.delete("/{video_id}", dependencies=[Depends(require_auth)])
def delete(video_id: str, uid: str = Depends(user_id)):
    """Deleting a video purges everything: vectors, thumbnails, the raw upload,
    and the manifest row — batch calls where the provider supports them.
    Sample videos are protected (unselect them from a query instead)."""
    if is_sample(video_id):
        raise HTTPException(403, "Sample videos can't be deleted — unselect it "
                                 "from your query instead.")
    row = db.get_video(video_id)
    if row is None or row["user_id"] != uid:
        raise HTTPException(404, "Video not found.")
    # Call off the scheduled run FIRST, and only while it could still do work.
    # First, because a run that is already executing may otherwise write points
    # after vector_store.delete_video() has run, leaving orphans behind the row
    # that would have explained them.
    #
    # NOT_INFLIGHT_STATUSES, not TERMINAL_STATUSES: `pending` has to be excluded
    # too. A pending row was never handed to Prefect, so there is nothing to
    # cancel, and asking anyway costs a round trip — measured at ~0.9s per
    # delete across bench.py's 30-probe cleanup, which is time the dispatcher
    # spends admitting more of those probes.
    if row["status"] not in config.NOT_INFLIGHT_STATUSES:
        jobs.cancel_ingest(video_id, row.get("flow_run_id"))
    vector_store.delete_video(uid, video_id)
    storage.delete_prefix(storage.frame_prefix(uid, video_id))
    if row.get("storage_key"):
        storage.delete_key(row["storage_key"])
    db.delete_video(video_id)
    return {"ok": True, "video_id": video_id}
