"""Table definitions and additive migrations for the source manifest.

Split out of the db package because it is read as one piece: the CREATE describes a fresh
database and the ALTERs bring an existing one to the same shape, and the pair has
to stay in sync by hand. CREATE TABLE IF NOT EXISTS is a no-op on a table that
already exists — it does NOT reconcile the definition (measured on a throwaway
table, not assumed). A column added to the CREATE alone reaches the grader's
empty database and nothing else; added to the ALTER alone it reaches every
existing database and not a fresh one.

No Alembic: the repo speaks raw SQL through psycopg and has no ORM models, so
autogenerate has nothing to read and it would buy a dependency plus a startup
step. init_schema() on every boot of api, worker and seed is what makes "fresh
clone -> docker compose up" true. Outgrow it the moment a change has to rewrite
existing rows or be reversible; that is numbered .sql files plus a
schema_migrations table, about twenty lines, still without a dependency.
"""
from __future__ import annotations

SCHEMA = """
CREATE TABLE IF NOT EXISTS ms_videos (
    -- yt_<youtube id> | up_<uuid> | doc_<hash of kind + normalized uri>. The
    -- document form is deterministic on purpose: re-registering the same source
    -- updates its row instead of adding a near-duplicate.
    id           TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    kind         TEXT NOT NULL DEFAULT 'video',  -- video | paper | deck
    source       TEXT NOT NULL,              -- youtube | upload
    url          TEXT,                       -- YouTube URL (source=youtube)
    storage_key  TEXT,                       -- uploads/<user>/<id>.<ext> (source=upload)
    source_hash  TEXT,                       -- sha256 of the file / yt video id
    title        TEXT,
    status       TEXT NOT NULL DEFAULT 'pending',
    error        TEXT,
    frame_count  INT,                         -- also the chunk count for documents
    progress     REAL,                       -- 0..1 within the current stage
    attempts     INT NOT NULL DEFAULT 0,
    embed_version TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- When the CURRENT phase has to be done, or the reaper calls it hung. NULL
    -- while pending or finished, because neither occupies a dispatch slot.
    phase_deadline TIMESTAMPTZ,
    -- Which run owns this row right now. Written by db.start_run when a process
    -- takes the row, cleared the moment it lands or is requeued. Every write a
    -- running flow makes carries it, so a flow that has ALREADY lost the row
    -- (reaped while alive, then re-admitted to someone else) finds its updates
    -- matching nothing instead of overwriting the new owner. Liveness and
    -- deadlines say WHETHER someone is working; this says WHO.
    run_token TEXT,
    -- Liveness, and deliberately not updated_at: four writers bump that column
    -- for reasons unrelated to whether a process still exists, so "written
    -- recently" is not evidence of life. NULL is a third state that matters —
    -- no process has EVER vouched for this row (scheduled, not yet started) —
    -- and silence cannot be read against it. Written only by src/heartbeat.py,
    -- cleared wherever the row is admitted or requeued.
    last_heartbeat_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS ms_videos_user_idx   ON ms_videos (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS ms_videos_status_idx ON ms_videos (status);
CREATE INDEX IF NOT EXISTS ms_videos_hash_idx   ON ms_videos (user_id, source_hash);

-- Additive migrations, for databases that predate a column (the module docstring
-- says why they have to exist alongside the CREATE). NOT NULL together with
-- DEFAULT is what makes one safe: it backfills the rows already there in the
-- same statement, which is the data migration one would otherwise hand-write.
ALTER TABLE ms_videos ADD COLUMN IF NOT EXISTS kind TEXT NOT NULL DEFAULT 'video';
-- These two are nullable with no default, so there is nothing to backfill, and
-- NULL is the safe reading in both: no budget to be late for, and nobody has
-- vouched for it. An old row is left alone rather than reaped on a deadline it
-- never had.
ALTER TABLE ms_videos ADD COLUMN IF NOT EXISTS phase_deadline TIMESTAMPTZ;
ALTER TABLE ms_videos ADD COLUMN IF NOT EXISTS last_heartbeat_at TIMESTAMPTZ;
ALTER TABLE ms_videos ADD COLUMN IF NOT EXISTS run_token TEXT;

-- Bring-your-own-model: a tenant's hosted LLM endpoint (vLLM / Ollama / any
-- OpenAI-compatible server, NVIDIA NIM, or Anthropic). When a row exists the
-- read path answers with THIS model instead of the server's LLM_* env config.
CREATE TABLE IF NOT EXISTS ms_user_llms (
    user_id    TEXT PRIMARY KEY,
    provider   TEXT NOT NULL DEFAULT 'openai',  -- openai | nvidia | anthropic
    model      TEXT NOT NULL,
    base_url   TEXT,                            -- e.g. http://my-vllm:8000/v1
    api_key    TEXT,                            -- optional (vLLM often has none)
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""
