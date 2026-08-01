# Assignment 3 — Moment Search at Scale · Eval Report

Student: Robin Bertus  ·  Base URL: http://localhost:8100

| Check | Result | Evidence |
|---|---|---|
| app_up | ✅ pass | GET / -> 200 |
| documents_async | ✅ pass | POST /admin/documents -> 202 in 201ms |
| sources_status | ✅ pass | GET /admin/sources -> 200, kinds=['deck', 'paper', 'video'] |
| paper_indexed | ✅ pass | page-locator citation present: True |
| deck_indexed | ✅ pass | slide-locator citation present: True |
| cross_source | ✅ pass | kinds across answers: ['deck', 'paper', 'video'] |
| grounded | ✅ pass | 6 citations, all with text+locator: True |
| decoupled | ❌ fail | run `python benchmark/bench.py` — search p95 during ingest <= 1.3x idle |
| RED_LINE_canary_clean | ✅ pass | clean |

_Manual criteria (resilience, deploy, video demo) graded from your submission._
Run `python benchmark/bench.py --resilience` for the no-loss proof.