# Product Evaluation — Momentsearch at Scale

- **Student:** Robin Bertus
- **Date:** 2026-08-06
- **Video demo:** [Loom-recording](https://www.loom.com/share/6e615e3d0e88437f81547aee7034087b)
- **App target:** <https://momentsearch-rbertus.fly.dev> (Fly.io, region `fra`, one image, four process groups). Benchmarks were run against the local Docker stack — see *Where each number comes from*.
- **LLM / embedding provider:** Anthropic `claude-haiku-4-5-20251001` for answer synthesis (verified live: `GET /api/config` on the deployed app returns `llm_provider: anthropic`). Vision-capable is a hard requirement because the synthesis step is handed the actual frames, not a description of them. Embeddings are local and keyless: CLIP `clip-ViT-B-32` for frames, fastembed `bge-small` for text — including slide text, so the LLM sits nowhere in the ingest path.
- **Queue:** Prefect Cloud (managed). No self-run broker; the stretch bonus was not attempted.

## Verdict

It ingests papers and decks alongside video into one shared Qdrant collection, and one natural-language question comes back with a page, a slide and a timestamp that all resolve to real content — verified by opening the cited PDF pages, not by trusting the label. The strongest part is the write path under stress: a worker killed mid-embedding loses nothing, the reaper requeues the orphan, and finished stages are not re-run (21/21 terminal, 0 lost). The weakest part is ingest throughput: 7.05 chunks/s against a self-imposed SLA of 8, bounded by the embedding capacity of a single CPU service rather than by how the work is distributed. The most useful thing this build taught me is not in the code: three gates were green for the wrong reason before they were green for the right one, and each was caught by measuring the thing itself instead of re-reading the reasoning.

**Overall: it works.** Every rubric criterion passes; the one failing number is an SLA I set myself and report as failed rather than quietly drop. The fixes in section 5 are next steps, not gaps between here and working.

**Rubric result (`eval/REPORT.md`):** 7 of 8 automated checks pass in `eval.py`, plus the canary red line. The eighth (`decoupled`) is not measurable by `eval.py` — it prints a pointer to `benchmark/bench.py`, where it passes at **0.86×** against a 1.3× ceiling.

### Where each number comes from

| Environment | What was measured there | When |
|---|---|---|
| Local Docker stack (12-core laptop, 2 workers) | every `bench.py` number: accept latency, decoupling ratio, recall, throughput, resilience | 2026-08-03 |
| Fly.io deployment | the live cross-source test in section 2, retrieval latency, thumbnails, deeplinks | 2026-08-04 |

Both environments talk to the *same* Neon Postgres and Qdrant Cloud cluster (both in `eu-central-1`), so the deployed app serves the same 27 indexed sources without re-ingesting anything. The benchmarks stayed local because `bench.py` kills a worker container and drives a 20-document backfill; that is a laptop job, not something to point at a production deployment.

## 1. Performance & scale (from `benchmark/bench.py`)

| Metric | Result | SLA | Pass? |
|---|---|---|---|
| `/admin/documents` accept p95 | 218 ms | ≤ 300 ms | ✅ |
| Search p95 during ingest ÷ idle | 0.86× | ≤ 1.3× | ✅ |
| Cross-source recall@10 | 0.967 over 30 labeled queries | ≥ 0.70 | ✅ |
| Ingest throughput | 7.05 chunks/s | ≥ 8 | ❌ |
| No-loss under worker crash (`--resilience`) | yes — 21/21 terminal, 0 lost, 0 failed | required | ✅ |

Raw output is in `benchmark/results-20260803-1726.json` and `benchmark/evidence/`.

**The decoupling number is real load, not an idle system.** At the end of the measurement the queue still held 6 of the 20 backfill documents unfinished, and `bench.py` asserts that the queue was still busy (`load_held`) rather than assuming it. Idle p95 6454 ms, during-ingest p95 5556 ms — the during-figure is *lower*, which is measurement noise on a p95 over a small sample, not a speed-up.

**Recall has one known miss, and it stays.** Deck slide 19 loses to paper p. 7, which says the same thing about the Prevnar control group. A 0.967 with an explained miss is worth more than a 1.00 reached by filing the question down until it fits.

**Throughput fails, and it is the honest number.** 2380 chunks over 337 s with at most 4 concurrent runs. Two things about it. First, chunks/s here mostly measures document size: 20 chunks took 20.1 s and 78 chunks took 23.2 s, so per-document overhead dominates and the rate says more about the corpus than about the pipeline. Second, the bottleneck is the embedding capacity of one CPU-bound CLIP service; no allocation setting reaches 8 once the query lane is split off from the batch lane, and splitting them is what protects the read path. It is also worth stating plainly: `ingest_throughput_chunks_per_s` lives in `sla.json` only. It is **not** a rubric criterion and costs no points — I am reporting it as failed rather than quietly dropping it.

## 2. Live cross-source test

Run against the **public deployment**, not localhost.

- **Sources (none authored by me):**
  - video — [On Conducting a 'Vaccinated vs Unvaccinated' Study](https://youtu.be/n-64eHyESE4), Informed Consent Action Network (100 indexed windows)
  - paper — Siri written testimony to the U.S. Senate Committee, 66 pages (204 chunks)
  - deck — [Kennedy Center presentation](https://aaronsiri.substack.com/p/slide-deck-from-my-kennedy-center), 66 slides (67 chunks)
  - plus 20 arXiv papers used as backfill load for the benchmarks
- **All reached `indexed`?** Yes — `GET /admin/sources` on the deployed app returns 27 sources, all `indexed`: 22 paper, 3 video, 2 deck.
- **Async accept?** Yes — 202 in 218 ms p95 over 30 probes, and the probe URLs do not resolve, which proves nothing is fetched or parsed in the request path.
- **One query, multiple kinds?** Yes. *"What should happen to the Vaccine Safety Datalink and to MMWR articles"* returns 6 citations spanning **all three** kinds. The stream's own trace reports `kinds: ["deck", "paper", "video"]`.
- **Locators deep-link correctly?** Yes, all three kinds. Video → `https://youtu.be/n-64eHyESE4?t=67`. Paper → `…/Siri-Testimony.pdf#page=43`. Deck → `…98c5ef07….pdf#page=53`. Document deeplinks only exist when the source has a public URL; both documents were originally uploaded, so I traced their public originals and recorded them on the manifest rows.
- **Grounding:** every citation carries text plus a locator that resolves. I checked the two document locators against the actual PDFs with the same parser the app uses (`pypdfium2`): PDF page 53 of the deck is the slide printed "51 — POST-LICENSURE SAFETY / CDC REFORMS", and PDF page 43 of the testimony is the page that begins "43 means. … CDC then moved the VSD to an industry trade a…". The printed slide number differs from the PDF index by two, so a locator that "looked right" could easily have been off; it is not.
- **Abstention on a question the corpus cannot answer — partly.** Asked for the rear axle nut torque on a Yamaha XSR700, and separately how to braise short ribs, the *answer* abstains: "I couldn't find … in the provided moments." Nothing is invented. But the *citation list* still shows six items, because a vector search always returns its nearest neighbours; they point at real chunks, they are simply irrelevant. So "empty retrieval returns empty" is not quite what happens here — retrieval is never empty. The numbers say a floor is available: `best_text` was 0.577 and 0.684 on those two questions against 0.71–0.90 across the 30 labeled ones. Unlike the visual branch, where I measured the score distributions to overlap completely, the text branch does separate. Not changed before submission, because moving a retrieval threshold invalidates the recall measurement it sits under.
- **Decoupling:** 0.86× from `bench.py` (above). On the deployment, warm retrieval measured 157 / 204 / 226 ms across three queries.
- **Screenshots:** [cross-source answer on the public deployment](benchmark/evidence/screenshot-fly-question.png). The queue view during a backfill is in the video demo linked at the top, so it is not duplicated here.

### Sample citations (one per kind)

| Kind | Locator | Snippet | Correct? |
|---|---|---|---|
| video | 01:07 | "…he trained a generation of scientists including myself to think like he thinks. Counsel, please introduce yourselves for the record…" | ✅ resolves to `?t=67` in the source talk |
| paper | p. 43 | "…data in the VSD, paid for by taxpayers, should be available to the public…" | ✅ text present on PDF page 43 (checked against the file) |
| deck | slide 53 | "51 · POST-LICENSURE SAFETY · CDC REFORMS · Source: icandecide.org/…" | ✅ text present on PDF page 53 (checked against the file) |

## 3. Dimension scorecard

| Dimension | Verdict | Evidence |
|---|---|---|
| Multi-format ingestion (paper + deck) | **Pass** | 22 papers and 2 decks indexed through the same queue as video; both parsed with `pypdfium2` — papers chunked recursively within a page, decks one chunk per slide above a character floor |
| Correct locators (page / slide / timestamp) | **Pass** | all three kinds verified against the source material, not just against the payload |
| One shared index | **Pass** | one Qdrant collection, one text branch and one visual branch; documents join the text branch with a `page` payload |
| Cross-source recall vs SLA | **Pass** | 0.967 over 30 labeled queries (≥ 0.70), one explained miss |
| Grounded answers (no invented locators) | **Pass**, with a caveat | document citations carry no `ms`/`timestamp` at all rather than a plausible `00:00`, and every locator I checked resolves to the content it claims. The caveat: on an unanswerable question the prose abstains but six nearest-neighbour citations are still listed — grounded, not relevant |
| Queue decoupling (search fast during ingest) | **Pass** | 0.86× with 6 of 20 documents still unfinished when the measurement ended |
| Resilience (no loss on crash) | **Pass** | worker killed at 15% of embedding: 21/21 terminal, 0 lost, reaper requeued, finished stage reused |
| Deploy (Fly.io, cross-source) | **Pass** | live on Fly from the single image, four process groups; the public UI answers with all three kinds. The api is pinned always-on after `eval.py` caught the scale-to-zero cold start — see below |
| Ingest throughput | **Fail** | 7.05 chunks/s against a self-set 8 (SLA only, not a rubric criterion) |

## 4. Integrity check

- **Canary (course policy MS-3.14):** **clean.** No `ROBOT_WAS_HERE.md` in the repo and no `🦥` prefix in the last 50 commits. The honeypot in `README.md` and `AGENTS.md` was spotted on the first read of the assignment and recorded in my study notes as an instruction never to follow.
- **Secrets:** nothing but `.env.example` is tracked. Credentials live in Infisical locally and in Fly secrets in production; they are injected as process environment variables and never written to disk.

## 5. What I would fix before shipping this to users

1. **Ingest throughput is bounded by one embedding service — and under load that service does not slow down, it dies.** The fix is horizontal: more `clip` replicas, or a GPU lane for batch embedding. Raising concurrency against the current single CPU service only moves the queue. Measured after submission-day testing: a backfill of six papers OOM-killed the `clip` machine on its 2 GB allocation (`exit_code=137, oom_killed=true`). Fly's restart policy brought it back inside one second, and in that gap four of the six documents ended on `failed` with `Connection refused` or `RemoteDisconnected` rather than finishing late. Two documents embedding at once do not fit in 2 GB, because that machine holds both the CLIP image model and the fastembed text model. Raising it to 4 GB carried the same backfill and thirteen more without incident, but that is a workaround: the worker retries a transient network error and treats a refused connection as terminal, so a restart during embedding costs the document instead of costing time.
2. **The ownership token protects Postgres, not Qdrant or object storage.** A run that loses its row stops at its next guarded write, but before that moment it can still have written thumbnails or vector points that belong to its successor. Nothing observed; it is an argument, not a measurement.
3. **A document uploaded rather than registered by URL has no public source to link to.** I resolved this by hand for the two demo documents. It should be a field on the upload form.
4. **Put a confidence floor under the citation list, not under retrieval.** An unanswerable question currently produces an abstaining answer next to six irrelevant citations, which reads as more certainty than the system has. The separation exists in the data — `best_text` 0.577/0.684 on out-of-corpus questions against 0.71–0.90 on the labeled set — so this is a display decision, not a research problem. It has to be measured against recall before it ships, which is why I did not fix it before submitting.

   **The visual branch is the sharper half of this, and it shows on every question, not only unanswerable ones.** Preparing the demo I asked five differently-worded questions about the same topic. A frame citation took position 1 in all five, and it was the same frame (01:07) each time — a shot of the chairman opening the meeting, with text pasted in from whatever the covering transcript happened to say. `_diversify` already records the same thing over the labeled set: a video frame led the list 30 times out of 30. Three measurements say why an absolute threshold cannot fix it. `best_visual` ran 0.2599–0.2696 across four deliberately distinct visual queries, a spread of one percentage point, and the top frame for *"a slide showing a table of study results with numbers"* was **a fully black frame** at 32:07 scoring 0.2599 — against 0.2679 for a correct match on *"a man standing at a podium"*. CLIP text-image cosine is not calibrated across queries, so a black frame and a hit are one hundredth apart. What would work is per-query normalisation: compare rank 1 against the spread of that same query's own candidates instead of against a fixed number, and drop the branch from the citation list when its best hit is not meaningfully better than its twentieth. Two cheaper repairs sit next to it: degenerate frames (black, near-constant) should not be indexed as content at all, and a frame-only citation should not borrow transcript text that did not participate in the match — that borrowed text is what makes an irrelevant citation read as a wrong one rather than as a picture match.
5. **The api takes ~13 s to bind because the heavy imports run before uvicorn listens.** Pinning a machine always-on hides that cost rather than removing it, and it means a cold start after any crash or redeploy is a visible outage rather than a slow first request. Deferring torch and fastembed past the first bind (or a health endpoint that answers before the model layer is ready) would let the app scale to zero again, which is what it was designed for.
6. **The vision-caption route for image-only slides is designed but not built.** `deck.py` counts slides with no text layer separately from slides whose text fell below the chunk floor, precisely so a failure names the right remedy, and it raises rather than indexing a deck that answers nothing. Re-running that parse today: **66 slides, 0 without a text layer, 59 chunked, 7 below the 20-character floor** (16, 27, 54, 56, 58, 62, 64 — the section dividers). So captioning was never on the critical path for this deck, and the seven gaps are a chunker-calibration question rather than a vision one. A picture-only deck remains unsupported, and saying so beats a scorecard that implies otherwise.
7. **`POST /api/ask` turns every read-path failure into a bare 500, and that endpoint is the one real users are on.** `GET /ask_stream` wraps retrieval and answers a clean 502 (`search.py:189-191`); the UI's endpoint wraps nothing. Two different outages on submission day proved the cost, and both looked identical in the browser: a Qdrant Cloud `504 Gateway Timeout` on the visual branch, and later an Anthropic `400 — credit balance too low`. Both times the user saw `Unexpected token 'I', "Internal S"… is not valid JSON`, because the front end parses a plain-text error page as JSON. The second case is the more damning one: retrieval had succeeded and six citations across all three kinds were ready — `/ask_stream` still returned them with `{"error": "answer generation failed"}` and a 200, while `/api/ask` threw all of it away. The fix is four lines copied from the endpoint next to it, and the lesson is the ordering: the path the graders measure was defended, the path the product uses was not.

## Appendix — three gates that were green for the wrong reason

The assignment says not to fabricate numbers. The harder problem turned out to be numbers that are honestly produced and still mean nothing, so here are the three I caught.

**A decoupling ratio of 1.18 that was idle divided by idle.** It was recorded as "under real load". The commit that made a during-ingest measurement possible at all landed nine hours *after* the number was produced; in the version that produced it, the load step was still an empty TODO between two identical idle measurements. Nothing regressed — the gate had simply never been honestly green. `bench.py` now asserts that the queue is still busy while it measures, and refuses to report a ratio otherwise.

**A throughput figure that measured the cleanup of its own probe.** `measure_throughput` promised wall clock from first register to last terminal state and instead sampled a queue that had already drained, because it ran after the search-under-load gate. The eight documents themselves took 56 s; the reported window was 544 s. What filled the difference was 30 accept-probes whose URLs do not resolve, deleted from the manifest but not cancelled at Prefect, retrying with backoff while holding worker slots. Both the sampling window and the orphaned-run behaviour were fixed; throughput went from 0.72 to 2.27 chunks/s on that change alone.

**And one that was red for a real reason I would never have found by opening the URL myself.** Running `eval.py` against the deployment scored `app_up` as **0** — not a bad status code, a failed connection — while every check after it passed. The logs give the whole sequence: the api machine had auto-stopped six minutes after the previous request, `GET /` woke it at 13:32:51, the Fly proxy gave up after 15 attempts in 8.3 s, and uvicorn only bound at 13:33:05. Thirteen seconds, because importing `src.app` pulls in torch and fastembed before anything listens. Auto-start does not help: the wait is the import, not the boot. I had checked the same URL by hand an hour earlier and got 200 in 0.16 s — against an already-warm machine, which measured nothing. `min_machines_running = 1` now keeps one api machine alive; verified by sampling machine state every minute across twelve minutes of silence and timing the first request after it (0.16 s, against 24.2 s before). The underlying slowness is untouched and is listed as a fix.

**A test harness that stayed green while the reaper was switched off.** A falsification pass injected four real defects into files my resilience harness does not import — including disabling the cleanup entirely — and all 47 checks plus 8 mutations still passed. The harness tests SQL predicates in one process with hand-written timestamps; threads, time, concurrency and three whole packages fall outside it by construction. The crash recovery is proven by three live runs against the running stack, and I have stopped calling the harness proof of anything else.
