"""Document registration API — /admin, the document twin of /api/videos.

Why a second router with its own prefix: the video endpoints live under
/api/videos and non-negotiable 6 freezes their shape, while the assignment's
contract and its self-verify curls call POST /admin/documents and
GET /admin/sources. So this adds the document side plus a unified read, without
touching the video paths.

The async contract mirrors registering a video: validate the SHAPE of the
request, insert a pending row, return 202. The source itself is never touched
here — fetching and parsing belong on the queue. That is also why an external
URL is checked for form only: benchmark/bench.py posts 30 URLs that do not exist
and requires a 202 for each within 300ms, which is precisely the behaviour a
reachability check would break.

require_auth and user_id are imported from .videos on purpose: one auth rule
should live in one place, so a change to it cannot apply to half the write path.
"""
from __future__ import annotations

import hashlib
import re
from urllib.parse import urlsplit, urlunsplit

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .. import db, storage
from .videos import require_auth, user_id

router = APIRouter(prefix="/admin", tags=["admin"])

DOC_KINDS = ("paper", "deck")

_STORAGE_SCHEME = "storage://"
_MAX_URI_LEN = 2048
# A client-chosen storage key ends up as DATA / key inside storage.head(), so it
# has to be contained before it is used: a known document prefix, and segments
# that cannot climb out. Unlike an upload key (which the server mints and then
# recognises), this key is chosen by whoever drops the file in the bucket.
_DOC_KEY_PREFIXES = ("papers/", "decks/", "docs/")
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class DocumentRequest(BaseModel):
    uri: str
    kind: str
    title: str | None = None


def _normalize_uri(uri: str) -> str:
    """Scheme and host are case-insensitive, the path is not, and a fragment
    never identifies a different document."""
    parts = urlsplit(uri.strip())
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(),
                       parts.path, parts.query, ""))


def _document_id(normalized_uri: str) -> tuple[str, str]:
    """Return (id, source_hash), both derived from the URI.

    Deterministic on purpose: re-registering the same URI then updates the same
    row instead of adding a near-duplicate, exactly like yt_<video id> does for
    YouTube. The content hash would be the better dedup key, but it is unknowable
    here — reading the bytes is the queue's job, not this endpoint's.
    """
    digest = hashlib.sha256(normalized_uri.encode()).hexdigest()
    return f"doc_{digest[:12]}", digest


def _validate_http_url(uri: str) -> None:
    parts = urlsplit(uri)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise HTTPException(400, "uri must be an http(s) URL or a storage:// key.")


def _validate_storage_key(key: str) -> str:
    if not any(key.startswith(prefix) for prefix in _DOC_KEY_PREFIXES):
        raise HTTPException(
            400, f"storage key must start with one of {_DOC_KEY_PREFIXES}.")
    for segment in key.split("/"):
        # Rejects "", ".", ".." and anything starting with a dot, so the key
        # cannot escape the storage root or hide as a dotfile.
        if not _SEGMENT_RE.match(segment):
            raise HTTPException(400, "storage key has an unsafe path segment.")
    return key


@router.post("/documents", status_code=202, dependencies=[Depends(require_auth)])
def register_document(req: DocumentRequest, uid: str = Depends(user_id)):
    kind = req.kind.strip().lower()
    if kind not in DOC_KINDS:
        raise HTTPException(
            400, f"kind must be one of {DOC_KINDS}; register videos via POST /api/videos.")

    uri = req.uri.strip()
    if not uri or len(uri) > _MAX_URI_LEN:
        raise HTTPException(400, "uri is empty or too long.")

    normalized = _normalize_uri(uri)

    # Case-insensitive, because urlsplit lowercases the scheme everywhere else
    # in this module: without it "STORAGE://..." would fall through to the http
    # check and be rejected, while its lowercase twin registers fine.
    if uri[:len(_STORAGE_SCHEME)].lower() == _STORAGE_SCHEME:
        key = _validate_storage_key(uri[len(_STORAGE_SCHEME):])
        # An object in OUR OWN storage may be checked: it is one local call, and
        # a missing object is a client mistake worth reporting immediately. An
        # external URL is a different case — see the module docstring.
        if storage.head(key) is None:
            raise HTTPException(404, "Object not found in storage.")
        source, url, storage_key = "upload", None, key
    else:
        _validate_http_url(uri)
        # The normalized form is what gets stored, not the raw submission. The id
        # is derived from it, so storing the raw one would let two submissions of
        # the same document agree on their id while disagreeing on their url —
        # and that url is what a citation links to.
        source, url, storage_key = "url", normalized, None

    doc_id, source_hash = _document_id(normalized)
    row = db.upsert_pending({"id": doc_id, "user_id": uid, "kind": kind,
                             "source": source, "url": url,
                             "storage_key": storage_key,
                             "source_hash": source_hash, "title": req.title})

    # Left `pending` deliberately, in both dispatch modes: the dispatcher admits
    # it in fair order once an ingest flow for this kind exists
    # (jobs.INGEST_DEPLOYMENTS). Enqueueing here instead would let a bulk of
    # documents jump the queue and starve the video users.
    return {"id": row["id"], "kind": row["kind"], "status": row["status"]}


# ── Unified read (videos + documents) ────────────────────────────────────────

_SOURCE_FIELDS = ("id", "kind", "source", "url", "title", "status", "error",
                  "attempts", "created_at", "updated_at")


def _source(row: dict) -> dict:
    out = {k: row.get(k) for k in _SOURCE_FIELDS}
    # The contract asks for pct (0-100); the manifest stores progress as 0..1.
    progress = row.get("progress")
    out["pct"] = None if progress is None else round(progress * 100)
    return out


@router.get("/sources")
def list_sources(uid: str = Depends(user_id), kind: str | None = None,
                 status: str | None = None):
    """Every source this user owns, whatever its kind.

    No auth, matching the existing read endpoints (GET /api/videos). Worth being
    explicit about what that means: X-User-Id is not authenticated, so anyone who
    guesses a user id can read that user's list. Tenancy is tagged everywhere,
    identity is not enforced — the same gap PRODUCT_EVAL.md has to state plainly.
    """
    rows = db.list_videos(uid, status=status)
    if kind:
        # Filtered here rather than in SQL: without a query that needs it, an
        # index on kind would be speculation.
        rows = [r for r in rows if r.get("kind") == kind]
    return {"sources": [_source(r) for r in rows]}
