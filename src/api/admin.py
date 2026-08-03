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

from .. import config, db, jobs, storage
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


def _document_id(normalized_uri: str, kind: str, user_id: str) -> tuple[str, str]:
    """Return (id, source_hash) for this user's registration of this URI+kind.

    Deterministic on purpose: re-registering the same URI then updates the same
    row instead of adding a near-duplicate, exactly like yt_<video id> does for
    YouTube. The content hash would be the better dedup key, but it is unknowable
    here — reading the bytes is the queue's job, not this endpoint's.

    `kind` is part of the id, not just of the row, because it decides how the
    source is parsed and what its citations point at. Without it the same PDF
    registered as a paper and as a deck would land on one id, and since the
    upsert deliberately leaves `kind` alone, the second registration would
    silently keep the first one's kind. Two ids is the honest answer: they are
    two different ingests of the same bytes.

    `user_id` is in there for a blunter reason: the id is the primary key. Leave
    it out and two tenants who happen to register the same public URL collide on
    one row — the second registration resets the first one's indexed document to
    `pending` and re-ingests it, while the second user never sees the document at
    all, because the row still belongs to someone else.

    source_hash stays URI-only, so the same document offered under two kinds is
    still recognised as duplicate CONTENT later in the flow, where the bytes
    exist.
    """
    digest = hashlib.sha256(normalized_uri.encode()).hexdigest()
    scoped = hashlib.sha256(f"{user_id}:{kind}:{normalized_uri}".encode()).hexdigest()
    return f"doc_{scoped[:12]}", digest


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
        meta = storage.head(key)
        if meta is None:
            raise HTTPException(404, "Object not found in storage.")
        # The byte cap has to be checked here: unlike the URL path, which stops
        # streaming once it goes over, fetch_upload pulls the whole object down
        # and t_parse then reads it into memory in one go.
        if meta["size"] > config.MAX_DOCUMENT_MB * 1024 * 1024:
            raise HTTPException(413, f"Object exceeds the {config.MAX_DOCUMENT_MB}MB limit.")
        source, url, storage_key = "upload", None, key
    else:
        _validate_http_url(uri)
        # The normalized form is what gets stored, not the raw submission. The id
        # is derived from it, so storing the raw one would let two submissions of
        # the same document agree on their id while disagreeing on their url —
        # and that url is what a citation links to.
        source, url, storage_key = "url", normalized, None

    doc_id, source_hash = _document_id(normalized, kind, uid)
    row = db.upsert_pending({"id": doc_id, "user_id": uid, "kind": kind,
                             "source": source, "url": url,
                             "storage_key": storage_key,
                             "source_hash": source_hash, "title": req.title})

    # With fair dispatch on, leave it `pending`: the dispatcher admits it in fair
    # order, and enqueueing here would let a bulk of documents jump the queue and
    # starve the video users. With fair dispatch OFF there is no dispatcher
    # thread at all (dispatcher.start_in_background returns early), so the same
    # code path would leave documents pending forever while videos sail past —
    # /api/videos has this branch, and without it the two write paths disagree
    # about what "registered" means.
    # `status` is the documented word for "accepted for ingest", not a reading of
    # the row: the README's contract is {id, status:"pending", kind}. Since
    # upsert_pending now leaves an in-flight row alone, the two can differ, so
    # the row's real state rides alongside instead of overwriting the promise.
    accepted = {"id": row["id"], "kind": row["kind"], "status": "pending",
                **({"current_status": row["status"]}
                   if row["status"] != "pending" else {})}
    # Already busy: the guard kept it, so enqueueing here would put a second run
    # on a source a worker is holding.
    if row["status"] not in config.NOT_INFLIGHT_STATUSES:
        return accepted
    if not config.ENABLE_FAIR_DISPATCH and kind in jobs.dispatchable_kinds():
        return {**accepted, "flow_run_id": jobs.enqueue(row["id"], uid, kind)}
    return accepted


# ── Unified read (videos + documents) ────────────────────────────────────────

_SOURCE_FIELDS = ("id", "kind", "source", "url", "title", "status", "error",
                  "attempts", "created_at", "updated_at",
                  # How much this source actually produced: frames for a video,
                  # chunks for a document. Without it the unified view can say a
                  # source is 'indexed' without saying how much of it there is,
                  # and a benchmark measuring chunks per second reads eight
                  # zeroes and reports a throughput of nought. Additive, so what
                  # this endpoint already promised is unchanged.
                  "frame_count")


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
