# MomentSearch

**Ask questions about your videos and get answers grounded in the exact moments — by what's _seen_ on screen.**

🌐 **Live app:** [momentsearch.fly.dev](https://momentsearch.fly.dev/get-started)

MomentSearch is an open-source, production-shaped stack for **visual** video
search and RAG. Users upload videos (or paste YouTube URLs); background workers
sample keyframes, dedup them, embed them with CLIP and index them per-user in
[Qdrant](https://qdrant.tech). Ask a question and it retrieves the most
relevant moments and (optionally) has **your own vision LLM** read those
frames and write a cited answer — or honestly abstain when the evidence isn't
there.

> **Visual, not audio.** MomentSearch understands the *picture* — it never
> transcribes speech. That means it works on silent footage, screen recordings,
> sports, surveillance, b-roll, slides, demos, and anything where what you're
> looking for is something you can *see*.

- 🎥 **Presigned uploads** — the browser PUTs straight to object storage; gigabytes never flow through the API
- ⚙️ **Queue + stateless workers** — the API answers `202` instantly; Prefect-orchestrated workers do the heavy lifting
- 🔍 **Visual retrieval** — CLIP embeddings, runs locally, no API key needed to search
- 👥 **Multi-tenant & private** — every bucket key, Postgres row and Qdrant point is `user_id`-tagged and filtered
- 🛡️ **Confidence gate** — below-threshold retrievals abstain *before* the LLM is ever called
- 💬 **Cited answers** — bring your own vision LLM (OpenAI-compatible, NVIDIA, or Anthropic)
- 🏠 **Per-user models** — each tenant can plug in a model *they* host (vLLM, Ollama, any OpenAI-compatible endpoint) and their answers run on it
- 🔓 **Apache 2.0**

## Architecture

The design rule (same as its sibling, `digital-twin-akash`): **stateful =
rented managed service, stateless = this repo's code.** Every API box and
worker is disposable; durable state lives in object storage, Qdrant and
Postgres — "nothing on local."

Two paths that scale in opposite directions and never share a request:

```
WRITE PATH (slow, background)                       READ PATH (fast, ms retrieval)
                                                    
 browser ──1. POST /api/videos/presign──► API        question ──► POST /api/ask
 browser ──2. PUT video───► Object storage                │
 browser ──3. POST /api/videos (register)─► API           ▼
                │ insert row (pending)                CLIP text embed ──► clip service
                │ schedule flow run                        │
                ▼                    ▼                     ▼
          Neon Postgres      Prefect Cloud           Qdrant kNN (user_id-filtered,
          (manifest,         (queue, retries,        int8-quantized, rescored)
           status, hashes)    run dashboard)               │
                                   │ polls (HTTPS)         ▼
                                   ▼                  temporal dedup → top-k frames
                          Worker (src/worker.py)           │
   fetch → sample (ffmpeg→memory) → pHash dedup       Gate 1: score < threshold?
   → thumbnails → embed batches ──► CLIP service      └─ yes → abstain, no LLM call
   → Qdrant   (seed gate indexes the 4 talks first)        │ no
                                                           ▼
   pending → fetching → sampling → embedding         vision LLM reads the frames →
          → indexed | skipped | failed               cited answer ([n] validated)

           CLIP service (src/clip_service.py) — ONE warm model behind a URL:
           api + workers send batches to CLIP_SERVICE_URL; scale/GPU it alone
```

The whole system is **one Docker image** with four entrypoints (the command
picks which): the API, the ingest worker, the CLIP service, and a one-shot
seed gate. All application code lives under [`src/`](src/); the repo root holds
only build/config files.

| Piece | Where it lives | You… |
|---|---|---|
| API + worker + CLIP service (this repo, one image) | Fly.io / Docker / bare python | deploy it |
| Raw videos + thumbnails | S3 / GCS / Tigris (or local disk in dev) | rent it |
| Postgres — manifest + status | [Neon](https://neon.tech) | rent it |
| Work queue + run dashboard | [Prefect Cloud](https://app.prefect.cloud) (free tier) | rent it |
| Vector index | [Qdrant Cloud](https://cloud.qdrant.io) (or the compose container) | rent it |
| Vision LLM | OpenAI / NVIDIA / Anthropic / any OpenAI-compatible — env-switched | rent it |

## Quickstart (Docker)

```bash
git clone https://github.com/traversaal-ai/momentsearch.git
cd momentsearch
cp .env.example .env    # fill in: DATABASE_URL, PREFECT_API_URL/KEY
                        # (storage=local + compose Qdrant work out of the box;
                        #  LLM key optional — search works without it;
                        #  ADMIN_TOKEN optional — set it on public deploys)
docker compose up --build
# API + UI:       http://localhost:8000
# Queue/run view: https://app.prefect.cloud → Runs
```

Two pages, one app:

| Page | What it is |
|---|---|
| **`/`** | **Sample project — "A Deep Dive into LLMs."** Four LLM talks, pre-indexed, read-only. |
| **`/get-started`** | **Bring your own videos.** Add a YouTube URL or upload a file, then ask. |

**The sample corpus is a startup gate.** A one-shot `seed` service indexes the
four talks and must finish before `api`/`worker` start — so when
`http://localhost:8000` first answers, the samples are already queryable, never
half-done. First run takes a few minutes (model download + 4 videos); watch it
with `docker compose logs -f seed`. It's durable (Qdrant Cloud) and idempotent,
so every later `up` finds them indexed and starts in seconds. Set
`SEED_SAMPLE_VIDEOS=false` to skip the gate; `python examples/quickstart.py`
is the manual route (also runs sample queries in the terminal).

Bare processes instead of compose (each is `python -m` / uvicorn on the `src.`
module — run in separate terminals):
```
uvicorn src.app:app --port 8000          # API + UI
python -m src.worker                      # ingest worker
uvicorn src.clip_service:app --port 8001  # CLIP service (optional; else set CLIP_SERVICE_URL empty)
python -m src.seed                        # one-shot: index the 4 samples
```

## The write path — upload to searchable vectors

1. **Presign** — `POST /api/videos/presign {filename, content_type, size}`
   (Bearer auth when `ADMIN_TOKEN` is set). The server picks the key (`uploads/{user}/{id}.mp4` — never
   trusted from the client), caps size and type, and returns a time-limited
   PUT URL. With `STORAGE_PROVIDER=local` it returns a direct-upload URL
   instead (dev fallback).
2. **Upload** — the browser PUTs the file straight to the bucket.
3. **Register** — `POST /api/videos {video_id, key}`. The API HEAD-verifies
   the object (exists, size, key prefix belongs to this user), writes a
   `pending` row, schedules a Prefect run, returns `202` instantly.
4. **Worker** (per video, `WORKER_CONCURRENCY` at a time):
   - **fetch** — stream from the bucket (or yt-dlp for YouTube), `sha256` it;
     a duplicate `(user_id, source_hash)` marks the row `skipped` and stops.
   - **sample** — one ffmpeg pass decodes, samples (interval or scene-cut),
     downscales and pipes JPEGs to memory — no write-then-reopen. *The
     biggest scaling lever: sampling is what stops thousands of videos
     becoming billions of near-identical vectors.*
   - **dedup** — perceptual hash (dHash + luminance) drops visually-identical
     neighbours **before** they cost CLIP compute; thumbnails batch-upload to
     `frames/{user}/{id}/NNNNNN.jpg`.
   - **embed + index** — batches of `CLIP_BATCH` frames go to the **warm CLIP
     service** (no per-video model load), then upsert to Qdrant with
     deterministic IDs (`uuid5(video_id:frame_idx)` — re-runs overwrite,
     never duplicate), tagged `user_id`, `video_id`, `ms`, `embed_version`.

Poll `GET /api/videos` (or watch the UI chips) until `indexed`.

## The read path — question to answer-or-abstain

`POST /api/ask {question, video_id?}`:

1. **Retrieve** — CLIP text embedding (via the warm clip service) → Qdrant
   HNSW filtered by `user_id`
   (private *and* fast: the tenant index means a user's search touches only
   their slice), quantization-rescored. Milliseconds.
2. **Trim** — temporal dedup (two hits from the same video within 5s are one
   moment) → `TOP_K` frames. This count is how many costly frames reach the LLM.
3. **Gate 1 — confidence**: best score below `CONFIDENCE_THRESHOLD` → abstain
   now ("I couldn't find that in your videos"), **no LLM call**. Kills most
   hallucination risk for free.
4. **Generate** — the few best frames, downscaled to `LLM_IMAGE_MAX_PX`, go to
   the vision LLM: answer only from these frames, cite `[n]`, or say so.
   Citations are validated; invented references are stripped.
5. **Answer** — with clickable thumbnails + timestamps (presigned GETs straight
   from the bucket), or the honest refusal.

The cost fact that drives this shape: retrieval is ~10-30ms; the multimodal
LLM call is seconds and dominates cost. Optimize there — few frames,
downscaled, gated — not the vector store.

## Bring your own model (per user)

Which model writes the answer is resolved **per tenant**, in this order:

1. **The user's own hosted model** — saved via `PUT /api/llm` (or the "Use
   your own model" card in the UI): a **vLLM** / Ollama / LM Studio / Together
   / OpenRouter endpoint (anything OpenAI-compatible) via `base_url`, NVIDIA
   NIM, or Anthropic. The model must be **vision-capable** — it is shown the
   actual frames (e.g. `Qwen/Qwen2.5-VL-7B-Instruct` or
   `llava-hf/llava-v1.6-mistral-7b-hf` on vLLM).
2. **The server default** — the `LLM_*` env config, used when the user hasn't
   attached one.
3. **No model** — retrieval still works; answers degrade to honest
   visual-similarity summaries.

```bash
# attach your hosted vLLM to your account
# (drop the Authorization header when ADMIN_TOKEN is unset — the dev default)
curl -X PUT localhost:8000/api/llm \
  -H "Authorization: Bearer $ADMIN_TOKEN" -H "Content-Type: application/json" \
  -d '{"provider":"openai","model":"Qwen/Qwen2.5-VL-7B-Instruct",
       "base_url":"http://my-vllm-host:8000/v1"}'

curl -X POST -H "Authorization: Bearer $ADMIN_TOKEN" localhost:8000/api/llm/test
#  -> sends one tiny image through your model; fails fast if it isn't vision-capable

curl -X DELETE -H "Authorization: Bearer $ADMIN_TOKEN" localhost:8000/api/llm
#  -> back to the server default
```

Settings live in Postgres (`ms_user_llms`, one row per user); API keys are
write-only (masked on read, blank on update keeps the stored key). `/api/ask`
responses include `llm_source: "user" | "server"` so the UI can show whose
model answered. **Ops note:** user `base_url`s make your API box call
user-chosen hosts — on a hosted deployment, egress-restrict the API container
or allowlist hosts; self-hosted single-tenant setups don't care.

## Qdrant at frame scale

One shared collection, multi-tenant by `user_id` (tenant payload index) — not
collection-per-user. Frames balloon vector counts fast (a 1h video at 2s
sampling ≈ 1,800 candidate frames), so the low-RAM profile defaults **on**:

| Flag | Effect | Default |
|---|---|---|
| `QDRANT_ON_DISK` | original float vectors on disk | on |
| `QDRANT_QUANTIZATION` | int8 copies pinned in RAM (~4× smaller) do the search; queries rescore from the originals | on |
| `QDRANT_HNSW_ON_DISK` | the HNSW graph on disk too | on |

For scale intuition: 200M frame vectors ≈ 600GB float32 vs ≈ 150GB int8.
On a big-RAM node flip these off to trade memory back for speed. Payloads are
trimmed to filter/display fields; titles/URLs live in Postgres and join at
answer time. `embed_version` on every point means a future CLIP upgrade can
re-embed in the background without breaking the live index.

## What's scaled, and why

The design separates the four processes precisely so each scales on its **own**
bottleneck, independently — that's the whole point of splitting them out:

| Component | Scales by | Because its bottleneck is… | How |
|---|---|---|---|
| **API** (`src/app.py`) | replicas (horizontal) | request concurrency (all I/O, no heavy compute) | stateless; auto-stops when idle on Fly |
| **Worker** (`src/worker.py`) | replicas (horizontal) | ingest throughput — download + ffmpeg per video | `fly scale count worker=N` / `--scale worker=N`; workers only dial out, zero coordination |
| **CLIP service** (`src/clip_service.py`) | vertically → **GPU** | embedding FLOPs (the compute-heavy step) | one warm model behind `CLIP_SERVICE_URL`; move it to a GPU box, change only the URL |
| **Qdrant** | memory profile → shards | vector count (frames balloon fast) | int8 + on-disk + rescore by default; shard when one node is outgrown |

The two axes that matter pull in opposite directions: **ingest** (many cheap
CPU workers, scale out) vs. **embedding** (one hot model, scale up/GPU). Coupling
them — the naive "CLIP inside the worker" — would force you to pay for GPUs on
every worker or starve embedding on every scale-out. Splitting them is what lets
you add cheap workers for a backfill while a single GPU handles all their embeds.

## Scaling — the details

**Workers.** The API must answer `202` instantly, but a video takes minutes —
workers pull runs from Prefect Cloud and execute them, `WORKER_CONCURRENCY`
at a time. Runs bottleneck on different resources (fetch = network, sampling
= CPU, embedding = CPU/GPU), so concurrent runs overlap. One worker machine
full? `fly scale count worker=3` or `docker compose up --scale worker=3` —
workers only dial out, so replicas need zero coordination.

**CLIP (the usual bottleneck) — "embedding is a URL".** Inference runs in a
dedicated service ([clip_service.py](clip_service.py)): one warm model loaded
once at boot, api + workers send batches over HTTP (`CLIP_SERVICE_URL`).
That's what makes workers cheap and stateless — no torch, no ~15-30s model
reload per video — and it's the standard model-serving pattern (TEI / Triton /
OpenAI-embeddings-shaped). Scaling embedding = scaling that one service: CPU
container today, the same container on a GPU machine later, with nothing but
the URL changing. Unset `CLIP_SERVICE_URL` and everything embeds in-process —
the zero-service simple mode for cloners.

**Deletes purge everything** — `DELETE /api/videos/{id}` removes the vectors
(by filter), thumbnails + raw upload (batch delete), and the manifest row.

**Fair scheduling (WFQ).** The queue is fair, not FIFO. If it enqueued every
video to Prefect at register time, Prefect would run them in submitted order —
one user who uploads 50 videos blocks everyone behind them. Instead videos wait
`pending` in Postgres and a **dispatcher** ([src/dispatcher.py](src/dispatcher.py))
admits them **round-robin across users**, keeping only `DISPATCH_MAX_INFLIGHT`
running at once. So the waiting line lives in *our* DB, fairly ordered
([`db.wfq_claim`](src/db.py) ranks each user's videos by age and takes
everyone's oldest first, then everyone's second, …) — no user can starve the
others. Set `ENABLE_FAIR_DISPATCH=false` to fall back to plain FIFO and see the
difference. `DISPATCH_MAX_INFLIGHT` should equal your real capacity
(`worker machines × WORKER_CONCURRENCY`); anything above that would just pile up
FIFO inside Prefect and defeat the fairness.

```
FIFO:  user A ▓▓▓▓▓▓▓▓▓▓ (50)  then→  user B ▓   ← B waits for all of A
WFQ:   A▓ B▓ A▓ B▓ A▓ B▓ …            ← interleaved; B is served immediately
```

**Later, under real load** (design room exists, not built): per-tenant *quotas*
and weights (the dispatcher's round-robin extends to weighted shares),
backpressure on queue depth, Redis query cache, OCR/transcript hybrid search.

## Deploy (Fly.io)

Full step-by-step guide: **[DEPLOYMENT.md](DEPLOYMENT.md)**. The short version —
one image, **three process groups** from [fly.toml](fly.toml) — `api`, `worker`,
and `clip` (each on its own machine size, each scaled by its own bottleneck; the
"what's scaled and why" table above explains the split):

```powershell
fly launch --no-deploy --copy-config          # create the app (once)
fly storage create                            # Tigris bucket; injects AWS_* secrets
Get-Content .env | Where-Object { $_ -match '^[A-Z_]+=.+' -and $_ -notmatch '^FLY_' } | fly secrets import
fly secrets set STORAGE_PROVIDER=flyio
fly deploy --ha=false                         # build image, start api/worker/clip
fly scale count worker=2                      # more ingest throughput, anytime
```

On every deploy, fly.toml's `release_command` runs the **seed gate** first
(`python -m src.seed`); if the four samples can't be indexed the deploy aborts
and the previous version keeps serving. The API machine auto-stops when idle;
worker + clip stay up (scale both to 0 between ingest sessions — queued runs
just wait). Set a CORS rule on the bucket for your site's origin (see
`.env.example`) or browser uploads fail. Need GPU-speed embedding later? Run
the same clip container on a GPU machine and point `CLIP_SERVICE_URL` at it —
nothing else changes.

### Continuous deployment (GitHub Actions)

[`.github/workflows/fly-deploy.yml`](.github/workflows/fly-deploy.yml) deploys
to Fly on every push to `main`. One-time setup — create a deploy token and add
it as the `FLY_API_TOKEN` repo secret (Settings → Secrets and variables →
Actions):

```bash
fly tokens create deploy -x 999999h
```

## API

Auth is **optional**: with `ADMIN_TOKEN` unset (the local-dev default) no
endpoint needs a header — drop the `Authorization` lines below. Set it on any
public deploy and mutating endpoints start requiring it. The tenant is the
`X-User-Id` header (default `default`); swap in real per-user auth later —
the data model is already tenant-scoped everywhere.

```bash
# 1) presign
curl -X POST localhost:8000/api/videos/presign \
  -H "Authorization: Bearer $ADMIN_TOKEN" -H "Content-Type: application/json" \
  -d '{"filename":"demo.mp4","content_type":"video/mp4","size":123456789}'
# 2) PUT the file to the returned url, then 3) register:
curl -X POST localhost:8000/api/videos \
  -H "Authorization: Bearer $ADMIN_TOKEN" -H "Content-Type: application/json" \
  -d '{"video_id":"up_ab12cd34ef","key":"uploads/default/up_ab12cd34ef.mp4","title":"Demo"}'

# YouTube instead:
curl -X POST localhost:8000/api/videos \
  -H "Authorization: Bearer $ADMIN_TOKEN" -H "Content-Type: application/json" \
  -d '{"url":"https://youtu.be/VIDEO_ID"}'

# status / retry / delete
curl localhost:8000/api/videos
curl -X POST -H "Authorization: Bearer $ADMIN_TOKEN" localhost:8000/api/videos/up_ab12cd34ef/retry
curl -X DELETE -H "Authorization: Bearer $ADMIN_TOKEN" localhost:8000/api/videos/up_ab12cd34ef

# ask
curl -X POST localhost:8000/api/ask -H "Content-Type: application/json" \
  -d '{"question":"a diagram of the attention mechanism"}'
```

Public: `GET /` (sample UI) · `GET /get-started` · `GET /api/config` ·
`GET /api/health`.

## Layout

Repo root holds only build/config/docs; **all Python lives under `src/`**, with
the four entrypoints as top-level modules in the package.

```
├── Dockerfile               one image, four entrypoints (command selects which)
├── docker-compose.yml       local dev: clip + seed gate + api + worker
├── fly.toml                 Fly.io: api/worker/clip process groups + seed release_command
├── requirements.txt
├── .env.example             every env knob, documented inline
├── .github/
│   └── workflows/
│       └── fly-deploy.yml   CI: deploy to Fly on push to main
├── ui/
│   └── index.html           single-file web UI (presigned upload, status poll, player)
├── examples/
│   └── quickstart.py        manual in-process seed + terminal query demo
└── src/                     ── entrypoints ──────────────────────────────────
    ├── app.py               unified FastAPI app — videos + search routers, one port
    ├── worker.py            Prefect worker — serves "ms-ingest-video/ingest"
    ├── clip_service.py      CLIP inference service — one warm model behind a URL
    ├── seed.py              startup gate — indexes the 4 samples, then exits
    │                        ── core ──────────────────────────────────────────
    ├── config.py            every env knob in one place
    ├── db.py                Neon Postgres: manifest + status + per-user LLM rows
    ├── jobs.py              Prefect Cloud trigger (API-side run_deployment)
    ├── storage.py           object storage (aws|gcp|gcp_native|flyio|local)
    │                        + presigned PUT/GET, HEAD verify, batch delete
    ├── llm.py               provider-agnostic vision-LLM answer (frames downscaled)
    ├── samples.py           the four-sample "Deep Dive into LLMs" corpus
    ├── seeding.py           blocking seed-to-completion logic (used by seed.py)
    ├── api/
    │   ├── videos.py        write path: presign, register, status, retry, delete
    │   └── search.py        read path: /api/ask, /api/llm, config, media, UI
    ├── ingest/
    │   ├── fetch.py         source acquisition (bucket download | yt-dlp) + sha256
    │   ├── frames.py        ffmpeg pipe-to-memory sampling (interval | scene)
    │   ├── dedup.py         perceptual-hash dedup (before CLIP spends compute)
    │   └── pipeline.py      the Prefect flow: fetch → sample → embed/index
    └── rag/
        ├── embeddings.py    CLIP image+text — in-process or remote (CLIP_SERVICE_URL)
        ├── vector_store.py  multi-tenant Qdrant: tenant index, int8/on-disk, UUID5 ids
        └── search.py        retrieve → temporal dedup → confidence gate → cited answer
```

## Security notes (presigned uploads)

- On a public deploy, set `ADMIN_TOKEN` so the presign endpoint is authed —
  otherwise anyone can mint upload URLs. (Unset = open, fine for local dev.)
- The **server** generates the key (`uploads/{user}/{uuid}`), never the client;
  register re-checks the prefix, so users can't claim others' objects.
- Size and content-type are capped at presign time and re-verified via HEAD.
- Keep the bucket **private**; thumbnails/playback go out via presigned GETs.
- ffmpeg/yt-dlp parse untrusted input — run workers in containers, not on the
  API box.
- Prompt-injection: frames are pixels (low risk), but treat any future
  OCR/transcript text as data, never instructions.

## Known limits

- **YouTube downloads.** Modern yt-dlp (2025+) needs a **JavaScript runtime +
  its EJS challenge-solver** to extract YouTube formats at all — without them
  every video fails "This video is not available." The Docker image installs
  **Node** and the worker fetches the solver automatically, so this works out
  of the box; for bare-process dev, install `node` or `deno`. **Cookies** then
  get past sign-in/bot-checks and work **everywhere** (home and datacenter):
  export a `cookies.txt` from a logged-in browser and supply it via
  `YT_COOKIES_FILE` (mounted file, local) or `YT_COOKIES_B64` (base64 secret,
  e.g. `fly secrets set YT_COOKIES_B64="$(base64 -w0 data/cookies.txt)"` on
  cloud). Cookies expire in a few weeks — re-export when it starts failing.
  Uploads are never affected by any of this.
- The embedded local Qdrant (`QDRANT_URL` empty) can't be shared by API and
  worker concurrently — single-process dev only; compose runs a real Qdrant.
- Faithfulness ceiling is *near*-zero, not zero — the gate + citations remove
  most of it; a vision-verifier pass is a future, costlier layer.

## License

Apache 2.0.
