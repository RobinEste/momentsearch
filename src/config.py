"""Central env-driven config — every knob in one place.

Same conventions as the digital-twin-akash service: module-level constants,
provider-neutral STORAGE_* credentials with AWS_* fallbacks, Prefect Cloud
read straight from PREFECT_API_URL / PREFECT_API_KEY by the SDK.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"  # local-provider storage root (dev only)


def _envbool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


# --- Database (Neon Postgres) — videos manifest, source of truth ------------
DATABASE_URL = os.getenv("DATABASE_URL", "")

# --- API auth ----------------------------------------------------------------
# Bearer token required on every mutating endpoint (presign, register, delete,
# retry). The tenant is the X-User-Id header (default "default") — swap this
# for real per-user auth (JWT/Clerk) later without touching the data model:
# every bucket key, Postgres row, and Qdrant point is already user_id-tagged.
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "default")

# --- Object storage (videos + frame thumbnails) ------------------------------
# STORAGE_PROVIDER: local | aws | gcp | gcp_native | flyio
# aws/gcp/flyio share one boto3 S3 client (different endpoints); gcp_native
# uses Google's SDK + service-account JSON; local writes under ./data (dev).
STORAGE_PROVIDER = os.getenv("STORAGE_PROVIDER", "local").strip().lower()
STORAGE_BUCKET = (os.getenv("STORAGE_BUCKET", "")
                  or os.getenv("BUCKET_NAME", "")            # injected by `fly storage create`
                  or os.getenv("GCS_BUCKET_NAME", "")        # gcp_native conventions
                  or os.getenv("GOOGLE_CLOUD_BUCKET_NAME", ""))
STORAGE_ACCESS_KEY_ID = os.getenv("STORAGE_ACCESS_KEY_ID", "").strip() or os.getenv("AWS_ACCESS_KEY_ID", "").strip()
STORAGE_SECRET_ACCESS_KEY = os.getenv("STORAGE_SECRET_ACCESS_KEY", "").strip() or os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
AWS_REGION = os.getenv("STORAGE_REGION", "").strip() or os.getenv("AWS_REGION", "auto")
_PROVIDER_ENDPOINTS = {
    "aws": None,  # boto3 default
    "flyio": "https://fly.storage.tigris.dev",
    "gcp": "https://storage.googleapis.com",
}
STORAGE_ENDPOINT = os.getenv("AWS_ENDPOINT_URL_S3", "").strip() or _PROVIDER_ENDPOINTS.get(STORAGE_PROVIDER)


def gcs_service_account_info() -> dict:
    """Service-account JSON for STORAGE_PROVIDER=gcp_native, rebuilt from the
    GOOGLE_CLOUD_* env vars (the standard exploded-JSON convention)."""
    key = os.getenv("GOOGLE_CLOUD_PRIVATE_KEY", "").strip()
    # dotenv strips surrounding quotes locally, but `fly secrets import` keeps
    # them literally — strip defensively so the PEM is valid in both places.
    if len(key) >= 2 and key[0] == key[-1] and key[0] in "\"'":
        key = key[1:-1]
    return {
        "type": "service_account",
        "project_id": os.getenv("GOOGLE_CLOUD_PROJECT_ID", ""),
        "private_key_id": os.getenv("GOOGLE_CLOUD_PRIVATE_KEY_ID", ""),
        "private_key": key.replace("\\n", "\n"),  # dotenv keeps \n literal inside quotes
        "client_email": os.getenv("GOOGLE_CLOUD_CLIENT_EMAIL", ""),
        "client_id": os.getenv("GOOGLE_CLOUD_CLIENT_ID", ""),
        "auth_uri": os.getenv("GOOGLE_CLOUD_AUTH_URI", "https://accounts.google.com/o/oauth2/auth"),
        "token_uri": os.getenv("GOOGLE_CLOUD_TOKEN_URI", "https://oauth2.googleapis.com/token"),
        "auth_provider_x509_cert_url": os.getenv(
            "GOOGLE_CLOUD_AUTH_PROVIDER_X509_CERT_URL", "https://www.googleapis.com/oauth2/v1/certs"),
        "client_x509_cert_url": os.getenv("GOOGLE_CLOUD_CLIENT_X509_CERT_URL", ""),
        "universe_domain": os.getenv("GOOGLE_CLOUD_UNIVERSE_DOMAIN", "googleapis.com"),
    }


# Bucket key layout — every key is user-scoped (tenant isolation at the path level):
#   uploads/{user_id}/{video_id}.{ext}      raw uploaded video (presigned PUT target)
#   frames/{user_id}/{video_id}/NNNNNN.jpg  downscaled frame thumbnails (citations)
UPLOAD_KEY_PREFIX = "uploads/"
FRAME_KEY_PREFIX = "frames/"

# --- Presigned uploads (browser -> bucket, bypassing the API) -----------------
PRESIGN_EXPIRY_S = _int("PRESIGN_EXPIRY_S", 900)          # presigned PUT lifetime
PRESIGN_GET_EXPIRY_S = _int("PRESIGN_GET_EXPIRY_S", 3600)  # thumbnails / playback
MAX_UPLOAD_MB = _int("MAX_UPLOAD_MB", 2048)                # register rejects bigger objects
# Documents are two orders of magnitude smaller than video, and this cap is the
# only thing standing between "a URL a user pasted" and the worker's disk.
MAX_DOCUMENT_MB = _int("MAX_DOCUMENT_MB", 64)
ALLOWED_UPLOAD_TYPES = ("video/",)                         # content-type must start with

# --- Video ingest lifecycle ---------------------------------------------------
# pending  = registered, waiting in our fair queue (not yet sent to Prefect)
# queued   = the dispatcher picked it and scheduled a Prefect run
# fetching = acquiring the source file; sampling = frames + dedup + thumbnails;
# embedding = CLIP + Qdrant upsert; skipped = duplicate (user_id, source_hash).
VIDEO_STATUSES = ("pending", "queued", "fetching", "sampling", "embedding",
                  "indexed", "skipped", "failed")
# Other source kinds bring their own stage names (a paper parses and chunks, it
# does not sample frames), so the tuple above is the VIDEO vocabulary, not the
# whole set. Capacity accounting must not depend on knowing every stage name.
#
# Terminal = the row is done, whichever way it ended. `pending` is waiting, so
# it costs nothing either. Everything else occupies execution capacity, by
# definition rather than by enumeration: a new per-kind stage is then counted
# and reapable the moment it exists, instead of silently escaping both. The
# failure direction flips with it — an unknown status now over-counts (admit a
# little less) instead of under-counting (over-admit, and a stuck row that no
# reaper can see).
TERMINAL_STATUSES = ("indexed", "skipped", "failed")
NOT_INFLIGHT_STATUSES = ("pending",) + TERMINAL_STATUSES

# --- Fair scheduling (WFQ) ----------------------------------------------------
# FIFO (default off): register enqueues to Prefect immediately -> Prefect runs
# them in submitted order, so one user with 50 videos blocks everyone behind
# them. Fair dispatch (WFQ, on): videos wait `pending` in Postgres and a
# dispatcher admits them round-robin ACROSS users, keeping only
# DISPATCH_MAX_INFLIGHT running at once — so the waiting line is fairly ordered
# in OUR DB, not FIFO inside Prefect. No user can starve the others.
ENABLE_FAIR_DISPATCH = _envbool("ENABLE_FAIR_DISPATCH", True)
# Max videos executing at once. Set to your total capacity:
# (worker machines) x WORKER_CONCURRENCY — anything above that would just pile
# up FIFO inside Prefect and defeat the fairness.
DISPATCH_MAX_INFLIGHT = _int("DISPATCH_MAX_INFLIGHT", _int("WORKER_CONCURRENCY", 2))
DISPATCH_INTERVAL_S = _float("DISPATCH_INTERVAL_S", 3.0)  # how often the dispatcher tops up

# --- Crash recovery: heartbeat + reaper ---------------------------------------
# Why any of this exists, and how the two signals divide the work: see the
# module docstring of src/reaper.py. That is the canonical account; what lives
# here are the numbers and where they come from.
REAP_ENABLED = _envbool("REAP_ENABLED", True)
REAP_INTERVAL_S = _float("REAP_INTERVAL_S", 10.0)
HEARTBEAT_INTERVAL_S = _float("HEARTBEAT_INTERVAL_S", 10.0)
HEARTBEAT_STALE_AFTER_S = _float("HEARTBEAT_STALE_AFTER_S", 45.0)  # 4.5 missed beats
# A source that keeps killing its worker would otherwise be requeued forever,
# burning a slot each round. `attempts` counts starts since the last success
# (db.start_run bumps it, set_status clears it on a clean landing), so this is
# consecutive failures and not a lifetime total.
MAX_ATTEMPTS = _int("MAX_ATTEMPTS", 3)
# Landing on one of these clears the counter: the source got somewhere on its
# own. Derived from TERMINAL_STATUSES minus 'failed' rather than listed, so a
# future clean-landing status resets it the moment it exists.
#
# 'pending' is deliberately NOT here even though it is also not-in-flight. The
# dispatcher writes 'pending' through set_status when it cannot reach Prefect
# (dispatcher.py), and that is neither a success nor a human asking again — it
# would hand the give-up counter an endless supply of fresh starts. The two
# paths where a human really does ask again (registration, retry) zero the
# counter explicitly in their own statement instead.
ATTEMPT_RESET_STATUSES = tuple(s for s in TERMINAL_STATUSES if s != "failed")

# Per-phase budgets as (base seconds, seconds per unit of work), from measured
# rates with roughly a 9x margin rather than guessed:
#   sampling   55 ms/frame  (400 sampled + dedup in 22s)
#   embedding  55 ms/item   (one CLIP batch of 128 in ~7s, warm clip-service)
#   parsing    2.0 ms/page  — even a 2000-page PDF is seconds, so it stays flat
# `units` is the size of the work the phase is about to do, passed by the caller
# that knows it: the chunk or frame count for embedding, the configured ceiling
# for sampling. Parsing and fetching learn their size only BY doing the work, so
# they are flat.
#
# The base is not slack, it is the legitimate idle a phase may contain. A task
# with @task(retries=n, retry_delay_seconds=d) sleeps between attempts while its
# flow is alive and beating, and the deadline is restamped only when the next
# attempt starts — so a base below that backoff reaps a healthy run mid-retry,
# which is the one failure this whole mechanism must never cause. Hence:
#   embedding  t_embed_index retries with a 60s delay -> 60s work slack + 120s
#   fetching   t_fetch retries with [30, 120] -> 1800s covers it many times over
# Generous where the work is unpredictable, tight where it was measured: a
# download is network-bound, and a crash there is already caught in 45s by the
# heartbeat, so its budget only has to bound a hang.
PHASE_BUDGETS_S: dict[str, tuple[float, float]] = {
    # Admission to the flow's first line: a round trip to Prefect Cloud, its
    # scheduling, a free worker slot, a subprocess spawn and a ~15-30s torch
    # import. Nothing here is under our control and being early is the expensive
    # direction, so it is roomy. It is also the only thing that can speak for a
    # run Prefect never picks up at all: a worker killed between admission and
    # delivery leaves a row no heartbeat will ever vouch for.
    "queued":    (900.0, 0.0),
    "fetching":  (1800.0, 0.0),
    "parsing":   (300.0, 0.0),
    # NOT indexed to the frame count, however tempting: sample() runs one ffmpeg
    # pass with an fps filter, which decodes the ENTIRE source and then emits
    # ~MAX_FRAMES of it. The measured 55 ms/frame is the emit, not the decode,
    # and the decode tracks the video's duration — which nothing here knows at
    # phase entry. A budget derived from the wrong variable reaped a healthy
    # two-hour lecture; flat and roomy is the honest shape.
    "sampling":  (1800.0, 0.0),
    "embedding": (180.0, 0.5),
    # The caption branch: a network fetch plus a bge embed of the whole
    # transcript, with no progress ticks and nothing at phase entry that knows
    # how many cues there will be. It exists as a phase at all because the row
    # must stay in flight while this runs — releasing it early handed a live
    # source to /retry — and a phase that is in flight needs a budget it can
    # actually make.
    "transcribing": (600.0, 0.0),
}
# An unrecognised phase gets this, and it is deliberately the roomy end: a stage
# name nobody remembered to budget must not be reaped early. Same failure
# direction as NOT_INFLIGHT_STATUSES above — err towards leaving work alone.
DEFAULT_PHASE_BUDGET_S = (900.0, 0.0)
# One knob for "this machine is slower than the one those rates came from". They
# were measured on a laptop in Docker with a warm CLIP service; a small cloud
# machine can be several times slower, and a budget that feels roomy locally
# eats live runs there, where it is hardest to see.
PHASE_BUDGET_SCALE = _float("PHASE_BUDGET_SCALE", 1.0)
# How long a phase that reports progress may go WITHOUT reporting any. Once a
# stage is ticking, "it stopped ticking" is a sharper reading of hung than "it
# outran an estimate of its total size", so db.set_progress restamps with this
# instead of with the phase budget. Deliberately shorter than most of those
# budgets: a stalled tick is better evidence, so it may act sooner.
PROGRESS_STALL_S = _float("PROGRESS_STALL_S", 300.0)
# The hard ceiling Prefect puts on a whole run. It scales with the same knob so
# that raising PHASE_BUDGET_SCALE does not silently push the phase budgets past
# it — but the two are NOT provably ordered: the embedding and sampling budgets
# grow with the work, task retries multiply a phase's budget by the attempt
# count, and setting FLOW_TIMEOUT_S explicitly overrides the scaling entirely.
# So this is a backstop, not a coordinated bound. When it does fire first the
# run process ends, its heartbeat stops, and the reaper requeues the row within
# HEARTBEAT_STALE_AFTER_S — a worse diagnosis than an expired phase budget, not
# a lost source.
FLOW_TIMEOUT_S = _float("FLOW_TIMEOUT_S", 7200.0 * PHASE_BUDGET_SCALE)


def phase_budget_s(phase: str, units: int | None = None) -> float | None:
    """How long `phase` may last before the reaper calls it hung.

    None means no deadline: the row is waiting or finished, so it is not holding
    anyone up.
    """
    if phase in NOT_INFLIGHT_STATUSES:
        return None
    base, per_unit = PHASE_BUDGETS_S.get(phase, DEFAULT_PHASE_BUDGET_S)
    return (base + per_unit * (units or 0)) * PHASE_BUDGET_SCALE

# --- Frame sampling (the biggest scaling lever) --------------------------------
# interval: one frame every FRAME_INTERVAL_SEC (widened to respect MAX_FRAMES).
# scene:    one frame per detected cut (ffmpeg scene filter).
FRAME_STRATEGY = os.getenv("FRAME_STRATEGY", "interval").strip().lower()
FRAME_INTERVAL_SEC = _float("FRAME_INTERVAL_SEC", 2.0)
SCENE_THRESHOLD = _float("SCENE_THRESHOLD", 0.4)
MAX_FRAMES = _int("MAX_FRAMES", 400)
THUMB_WIDTH = _int("THUMB_WIDTH", 480)   # frames are downscaled in the ffmpeg pass
THUMB_QUALITY = _int("THUMB_QUALITY", 3)  # ffmpeg -q:v (2 best .. 31 worst)

# Perceptual-hash dedup — drop visually-identical neighbours BEFORE embedding.
DEDUP_ENABLED = _envbool("DEDUP_ENABLED", True)
DEDUP_MAX_DISTANCE = _int("DEDUP_MAX_DISTANCE", 4)  # Hamming distance on 64-bit dHash

# --- CLIP embeddings ------------------------------------------------------------
# One model encodes frames and text queries into a shared space. Runs on CPU
# inside the worker today; EMBED_VERSION is stamped on every Qdrant point so a
# future re-embed (or an external GPU CLIP service) can replace stale vectors
# without guessing.
CLIP_MODEL = os.getenv("CLIP_MODEL", "clip-ViT-B-32").strip()
CLIP_BATCH = _int("CLIP_BATCH", 128)   # frames per embed call (inner batch is 32)
# Inference-service URL ("embedding is a URL"). Set -> api/worker send batches
# to the warm clip_service.py container instead of loading the model in-process
# (which costs each Prefect run subprocess a fresh ~15-30s torch load). Unset
# -> in-process embedding (simple mode, no extra service). Point it at a GPU
# machine later — nothing else changes.
CLIP_SERVICE_URL = os.getenv("CLIP_SERVICE_URL", "").strip().rstrip("/")
# How many BATCH embeds (ingest) the clip service runs at once. The interactive
# endpoints the read path uses are never held by this — that asymmetry is the
# whole point, see src/clip_service.py for what it is worth in milliseconds.
# 0 disables the limit and restores the old free-for-all.
CLIP_BATCH_CONCURRENCY = _int("CLIP_BATCH_CONCURRENCY", 2)
# Vector dimension override. 0 = auto: known CLIP models resolve from a table
# (so the API can create the collection at boot WITHOUT loading the model);
# unknown models load the model to measure. Set explicitly for custom models.
CLIP_DIM = _int("CLIP_DIM", 0)
EMBED_VERSION = os.getenv("EMBED_VERSION", f"{CLIP_MODEL}-v1")

# --- Multimodal: transcript (text) branch (Path 1) -----------------------------
# The visual branch is CLIP frames (above). This adds a SECOND branch: YouTube
# captions, chunked by time, embedded with a small semantic text model (bge via
# fastembed — CPU, free), in a separate Qdrant collection. At query time both
# branches run and fuse by RANK (RRF) — CLIP scores (~0.3) and text scores
# (~0.7) live on different scales, so raw-score comparison is meaningless.
# Uploaded files have no captions, so this is YouTube-only; a video with no
# captions just indexes visually (the branch is skipped, never fatal).
ENABLE_TRANSCRIPT = _envbool("ENABLE_TRANSCRIPT", True)
TEXT_COLLECTION = os.getenv("TEXT_COLLECTION", "moments_text")
# Transcript-branch embedding PROVIDER — env decides the model:
#   fastembed (default) -> bge via fastembed: CPU, free, NO API key (keeps search
#                          working keyless for a fresh cloner). Dim 384.
#   openai              -> OpenAI (or any OpenAI-compatible) embeddings API, e.g.
#                          text-embedding-3-small. Hosted, stronger retrieval,
#                          costs per call + needs a key. Reuses the LLM_* key /
#                          base_url by default (override with TEXT_EMBED_API_KEY /
#                          TEXT_EMBED_BASE_URL). Setting the provider alone flips
#                          the default model+dim to 3-small / 1536.
# The model & dim MUST match between indexing and querying, so switching provider
# means RE-SEEDING the transcript collection (its vector dim changes). The two
# branches fuse by RANK (RRF), so the text model is independent of CLIP.
TEXT_EMBED_PROVIDER = os.getenv("TEXT_EMBED_PROVIDER", "fastembed").strip().lower()
_TE_OPENAI = TEXT_EMBED_PROVIDER == "openai"
TEXT_EMBED_MODEL = os.getenv(
    "TEXT_EMBED_MODEL",
    "text-embedding-3-small" if _TE_OPENAI else "BAAI/bge-small-en-v1.5")
TEXT_EMBED_DIM = _int("TEXT_EMBED_DIM", 1536 if _TE_OPENAI else 384)
# openai provider: falls back to the LLM key/base_url in embeddings.py so ONE
# OpenAI key can power both the answer and the text embeddings.
TEXT_EMBED_API_KEY = os.getenv("TEXT_EMBED_API_KEY", "").strip()
TEXT_EMBED_BASE_URL = os.getenv("TEXT_EMBED_BASE_URL", "").strip()
TEXT_EMBED_VERSION = os.getenv("TEXT_EMBED_VERSION", f"{TEXT_EMBED_MODEL}-v1")
# Transcript chunking: group caption cues into ~CHUNK_SECONDS windows so a chunk
# is a coherent spoken passage with a real t_start/t_end, not one tiny cue.
TRANSCRIPT_CHUNK_SECONDS = _float("TRANSCRIPT_CHUNK_SECONDS", 20.0)
TRANSCRIPT_LANGS = [c.strip() for c in
                    os.getenv("TRANSCRIPT_LANGS", "en,en-US,en-GB").split(",") if c.strip()]

# --- Fusion (multimodal retrieval) ---------------------------------------------
# RRF: rank-based fusion across branches (score-agnostic). rrf = 1/(K + rank).
RRF_K = _int("RRF_K", 60)
# Hits from either branch within this many seconds are the SAME moment.
FUSION_WINDOW_S = _float("FUSION_WINDOW_S", 15.0)
# When a window has BOTH a frame and a transcript hit, multiply its score —
# two independent modalities agreeing is the strongest relevance signal.
CROSS_MODAL_BOOST = _float("CROSS_MODAL_BOOST", 1.5)
# Per-branch candidates fetched before fusion.
BRANCH_TOP_K = _int("BRANCH_TOP_K", 20)

# --- YouTube download hardening ---------------------------------------------------
# YouTube increasingly answers yt-dlp's default web client with "Sign in to
# confirm you're not a bot". Mitigations, in order of reliability:
#   YT_COOKIES_FILE  path to a Netscape cookies.txt exported from a logged-in
#                    browser (yt-dlp wiki: "Exporting YouTube cookies").
#                    In docker-compose, drop it at ./data/cookies.txt and set
#                    YT_COOKIES_FILE=/app/data/cookies.txt
#   YT_PROXY_URL     route YouTube traffic through a (residential) proxy,
#                    e.g. http://user:pass@host:port
#   YT_RETRY_CLIENTS on a bot-check error, automatically retry once with these
#                    alternate YouTube player clients (comma-separated).
# Cookies make yt-dlp an authenticated client — the one fix that works from
# ANY IP (home OR datacenter). Two ways to supply them, so the same code works
# local and deployed:
#   YT_COOKIES_FILE  path to a mounted cookies.txt   (easy locally)
#   YT_COOKIES_B64   base64 of cookies.txt as a secret (Fly/cloud: no file mount
#                    needed — the worker writes it to a temp file at runtime)
YT_COOKIES_FILE = os.getenv("YT_COOKIES_FILE", "").strip()
YT_COOKIES_B64 = os.getenv("YT_COOKIES_B64", "").strip()
YT_PROXY_URL = os.getenv("YT_PROXY_URL", "").strip()
# Player clients. Default EMPTY = let yt-dlp pick (best, once a JS runtime is
# present — see below). Forcing tv/android used to help pre-JS-runtime, but now
# those clients hand back media URLs that 403 on download, so we only fall back
# to them if the default attempt fails outright.
YT_PLAYER_CLIENTS = [c.strip() for c in
                     os.getenv("YT_PLAYER_CLIENTS", "").split(",") if c.strip()]
YT_FALLBACK_CLIENTS = [c.strip() for c in
                       os.getenv("YT_FALLBACK_CLIENTS", "tv,android,ios").split(",") if c.strip()]
# yt-dlp 2025+ needs a JavaScript runtime to compute YouTube signatures, plus
# its EJS challenge-solver component — WITHOUT these every video fails with
# "This video is not available" / "Requested format is not available". The
# Dockerfile installs Node; these tell yt-dlp to use it + fetch the solver.
# (For bare-process dev, install node or deno yourself.)
YT_JS_RUNTIMES = [c.strip() for c in
                  os.getenv("YT_JS_RUNTIMES", "node").split(",") if c.strip()]
YT_REMOTE_COMPONENTS = [c.strip() for c in
                        os.getenv("YT_REMOTE_COMPONENTS", "ejs:github").split(",") if c.strip()]

# --- Sample corpus ------------------------------------------------------------
# On boot the worker auto-ingests the four "Deep Dive into LLMs" sample talks
# (src/samples.py) if they aren't indexed yet — a fresh clone is queryable on
# the / page without running anything by hand. Set false to skip.
SEED_SAMPLE_VIDEOS = _envbool("SEED_SAMPLE_VIDEOS", True)

# --- Work orchestration (Prefect Cloud) ----------------------------------------
# The SDK reads PREFECT_API_URL / PREFECT_API_KEY from the environment directly.
# WORKER_CONCURRENCY is read by worker.py; retries live on the flow's tasks.

# --- Qdrant ----------------------------------------------------------------------
# One shared multi-tenant collection: every point carries user_id (tenant payload
# index) and every search/upsert/delete is user_id-filtered.
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip() or os.getenv("QDRANT_TOKEN", "").strip()
QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH", str(DATA / "qdrant"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "moments")
# Low-RAM profile: original vectors on disk, int8-quantized copies pinned in
# RAM (~4x smaller), HNSW graph on disk; queries rescore against the originals.
# Frames balloon vector counts fast, so these default ON.
QDRANT_QUANTIZATION = _envbool("QDRANT_QUANTIZATION", True)
QDRANT_ON_DISK = _envbool("QDRANT_ON_DISK", True)
QDRANT_HNSW_ON_DISK = _envbool("QDRANT_HNSW_ON_DISK", True)

# --- Retrieval / faithfulness ------------------------------------------------------
TOP_K = _int("TOP_K", 6)                 # frames fed to the multimodal LLM (3-8)
KNN_K = _int("KNN_K", 24)                # candidates fetched before trimming to TOP_K
# Ceiling on a caller-supplied top_k. The read path is reachable without a token,
# and every extra citation is a bucket fetch, a JPEG decode and one more image in
# the prompt -- so an unclamped `?top_k=999` hands a cost multiplier to anyone
# with the URL. Clamped in `retrieve`, so both /api/ask and /ask_stream inherit it.
MAX_TOP_K = _int("MAX_TOP_K", 12)
# Gate 1: abstain WITHOUT calling the LLM if BOTH branches' best raw score is
# below their threshold. Fusion scores are RRF (tiny), so the gate uses each
# branch's own raw cosine. CLIP text->image cosines run low (~0.2-0.35); bge
# text-text cosines run higher (~0.5-0.7 for real matches). 0 disables.
CONFIDENCE_THRESHOLD = _float("CONFIDENCE_THRESHOLD", 0.2)              # visual (CLIP)
TEXT_CONFIDENCE_THRESHOLD = _float("TEXT_CONFIDENCE_THRESHOLD", 0.35)  # transcript (bge)

# --- Multimodal LLM (answer synthesis only — retrieval works without it) -----------
# LLM_PROVIDER: openai | nvidia | anthropic ("openai" also covers any
# OpenAI-compatible server via LLM_BASE_URL: Ollama, vLLM, OpenRouter, ...).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
LLM_MAX_TOKENS = _int("LLM_MAX_TOKENS", 1024)
LLM_IMAGE_MAX_PX = _int("LLM_IMAGE_MAX_PX", 512)  # frames are downscaled again before the LLM


def llm_configured() -> bool:
    # Local OpenAI-compatible servers often need no key, so a base_url alone counts.
    return bool(LLM_API_KEY or LLM_BASE_URL)
