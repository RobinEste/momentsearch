#!/usr/bin/env python3
"""queries.template.jsonl -> queries.jsonl, with the running app's source ids.

    python benchmark/fill_queries.py                       # after the corpus is indexed
    python benchmark/fill_queries.py --status any          # before ingest finishes

The labels in the template are written by hand against the source documents:
which page of the testimony, which slide of the deck, which video. What cannot
be written by hand is the source_id — the app mints it at registration time from
(user, kind, uri), so it differs per machine and per re-registration. Hand-
copying three ids into thirty lines is exactly the kind of transcription this
should not depend on, hence this script.

It reads /admin/sources with no X-User-Id header on purpose: bench.py does the
same, so both see the DEFAULT_USER_ID tenant. A corpus registered under another
user would fill in ids that recall@10 can never match.

Stdlib only, like the rest of benchmark/.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASE = os.getenv("BASE_URL", "http://localhost:8100").rstrip("/")
TEMPLATE = ROOT / "benchmark" / "queries.template.jsonl"
OUT = ROOT / "benchmark" / "queries.jsonl"

# One placeholder per source kind the template labels against.
PLACEHOLDERS = {"video": "{{VIDEO}}", "paper": "{{PAPER}}", "deck": "{{DECK}}"}


def fetch_sources(status: str) -> list[dict]:
    url = f"{BASE}/admin/sources"
    with urllib.request.urlopen(url, timeout=20) as r:
        rows = json.loads(r.read().decode())["sources"]
    return rows if status == "any" else [r for r in rows if r.get("status") == status]


def resolve(rows: list[dict], overrides: dict[str, str]) -> dict[str, str]:
    """One id per kind, or a loud failure naming the candidates.

    Ambiguity here is not a corner case: registering the same PDF twice (say as
    a paper and then again after a rename) leaves two paper rows, and silently
    picking the newest would label every paper query against a source that may
    hold nothing. Two candidates is a question for the operator, not a guess.
    """
    resolved = {}
    for kind in PLACEHOLDERS:
        if overrides.get(kind):
            resolved[kind] = overrides[kind]
            continue
        matches = [r for r in rows if r.get("kind") == kind]
        if not matches:
            sys.exit(f"! no {kind} source found at {BASE} — register and ingest it first")
        if len(matches) > 1:
            listed = "\n  ".join(f"{r['id']}  {r.get('title') or r.get('url')}"
                                 f"  ({r.get('status')}, {r.get('chunk_count') or r.get('frame_count')})"
                                 for r in matches)
            sys.exit(f"! {len(matches)} {kind} sources; pass --{kind}-id to pick one:\n  {listed}")
        resolved[kind] = matches[0]["id"]
    return resolved


def render(ids: dict[str, str]) -> list[str]:
    lines = []
    for raw in TEMPLATE.read_text().splitlines():
        if not raw.strip():
            continue
        for kind, token in PLACEHOLDERS.items():
            raw = raw.replace(token, ids[kind])
        item = json.loads(raw)  # parse AFTER substitution: a bad id must fail here
        if "{{" in raw or not item.get("q") or not item.get("source_id"):
            sys.exit(f"! unfilled or incomplete line: {raw[:80]}")
        lines.append(json.dumps(item))
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", default="indexed",
                    help="only use sources in this status ('any' to skip the check)")
    for kind in PLACEHOLDERS:
        ap.add_argument(f"--{kind}-id", default=None, help=f"pin the {kind} source id")
    args = ap.parse_args()

    rows = fetch_sources(args.status)
    ids = resolve(rows, {k: getattr(args, f"{k}_id") for k in PLACEHOLDERS})
    lines = render(ids)
    OUT.write_text("\n".join(lines) + "\n")

    per_kind = {k: sum(1 for ln in lines if f'"{v}"' in ln) for k, v in ids.items()}
    print(f"wrote {OUT.relative_to(ROOT)}: {len(lines)} labeled queries")
    for kind, count in per_kind.items():
        print(f"  {kind:<6} {ids[kind]}  {count} queries")


if __name__ == "__main__":
    main()
