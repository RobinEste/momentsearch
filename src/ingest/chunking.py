"""Structure-aware recursive chunking for document text.

Splits paragraph-first, then sentence, then hard-cuts — never merging across a
page boundary, because the page number IS the citation locator and a chunk that
spans two pages makes it a lie.

Two sizes are at work here, and conflating them is the trap.

Characters are the TARGET: cheap, deterministic, and good enough to pack chunks
of a sensible shape without asking anybody. Tokens are the GUARANTEE, because
the embedding model (bge-small-en-v1.5) truncates at 512 tokens SILENTLY —
over-long input loses its tail with no error [measured: the tokenizer's own
truncation config says max_length=512, strategy longest_first].

A character cap cannot provide that guarantee, and this is measured, not
assumed. Chars-per-token on real papers, truncation disabled:

    two-column ACL paper   3.99 median   (citations, symbols, numbers cost more)
    single-column survey   5.20 median   (prose is cheaper)

Sizing on that looks safe, and it is not: chunking the survey at 1500 characters
still produced 6 chunks over 512 tokens, the worst at 741 — about 2.0
chars/token inside tables and formula blocks. A per-page median hides the dense
regions that decide the invariant. Hence `verify_token_limit`: pack by
characters, then split whatever the real tokenizer says is too long.

Why recursive and not semantic clustering: two 2026 studies found cluster-based
semantic chunking costs more and does not reliably beat recursive splitting.
See the 29-07-2026 entry in opleiding.md. recall@10 on our own corpus is the
tiebreaker, not the literature.
"""
from __future__ import annotations

import math
import re

CHUNK_TARGET_CHARS = 1400   # ~350 tokens at the measured 4.0 chars/token
CHUNK_MAX_CHARS = 1500      # hard ceiling: <=500 tokens even at 3.0 chars/token
CHUNK_OVERLAP_CHARS = 140   # 10%, so a sentence split across chunks survives once
MIN_CHUNK_CHARS = 80        # below this it is a stray caption, not an answer
_MAX_VERIFY_ROUNDS = 4      # bounded: degrade into smaller chunks, never hang

_PARAGRAPH_RE = re.compile(r"\n\s*\n")
# Sentence end followed by whitespace and a capital/digit. Crude on purpose:
# "et al." and "Fig. 3" would fool anything short of a real segmenter, and a
# slightly early split costs far less here than a dependency does.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _hard_split(text: str, limit: int, *, merge_short_tail: bool = True) -> list[str]:
    """Last resort for a single unit that is already over the ceiling (a table,
    a formula block, a paragraph with no sentence punctuation).

    With `merge_short_tail`, a trailing piece shorter than MIN_CHUNK_CHARS is
    folded back into the one before it. That is for callers that afterwards drop
    fragments that short as parsing noise: the tail of a hard split is not noise,
    it is the last sentences of the document, and dropping it removes them from
    the index without a trace.

    Callers that keep every fragment must pass False, because the merge can undo
    the split entirely — two pieces of 79 characters become one of 158 — and a
    caller splitting to get under a token limit would then never converge.
    """
    parts = [text[i:i + limit] for i in range(0, len(text), limit)]
    if merge_short_tail and len(parts) > 1 and len(parts[-1].strip()) < MIN_CHUNK_CHARS:
        parts[-2] += parts[-1]
        parts.pop()
    return parts


def _units(text: str) -> list[str]:
    """Paragraphs, falling back to sentences, falling back to hard cuts — so no
    unit handed to the packer is ever above the ceiling."""
    units: list[str] = []
    for paragraph in _PARAGRAPH_RE.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= CHUNK_MAX_CHARS:
            units.append(paragraph)
            continue
        for sentence in _SENTENCE_RE.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            units.extend([sentence] if len(sentence) <= CHUNK_MAX_CHARS
                         else _hard_split(sentence, CHUNK_MAX_CHARS))
    return units


def chunk_text(text: str) -> list[str]:
    """One page (or slide) of text -> chunks, greedily packed to the target size.

    Returns [] for text that holds nothing worth retrieving, which is a normal
    outcome for a title page or an image-only slide, not an error.
    """
    text = (text or "").strip()
    if len(text) < MIN_CHUNK_CHARS:
        return []

    chunks: list[str] = []
    current = ""
    for unit in _units(text):
        if not current:
            current = unit
        elif len(current) + 1 + len(unit) <= CHUNK_TARGET_CHARS:
            current = f"{current}\n{unit}"
        else:
            chunks.append(current)
            # Overlap carries the tail of the previous chunk into the next one,
            # so a statement split across the boundary is still retrievable whole
            # from one of the two.
            tail = current[-CHUNK_OVERLAP_CHARS:] if CHUNK_OVERLAP_CHARS else ""
            current = f"{tail}\n{unit}".strip() if tail else unit
    if current:
        chunks.append(current)

    # The overlap can push a chunk past the target; the ceiling is what protects
    # against silent truncation, so enforce it after packing rather than before.
    result: list[str] = []
    for chunk in chunks:
        if len(chunk) <= CHUNK_MAX_CHARS:
            if len(chunk.strip()) >= MIN_CHUNK_CHARS:
                result.append(chunk.strip())
            continue
        # Every part of an over-long chunk is kept, however short the last one
        # is. Merging that tail back would undo the split for anything between
        # the ceiling and ceiling+MIN_CHUNK_CHARS, leaving the "hard ceiling"
        # quietly unenforced; dropping it would lose real text. Neither, then.
        result.extend(part.strip() for part
                      in _hard_split(chunk, CHUNK_MAX_CHARS, merge_short_tail=False)
                      if part.strip())

    if not result and chunks:
        # The gate at the top accepted this text, so returning nothing here would
        # silently drop a whole page. It happens because that gate measures the
        # text with its paragraph breaks while the packer rejoins paragraphs with
        # a single newline — one character shorter per boundary, which is enough
        # to fall under the minimum. Keep the content rather than the arithmetic.
        merged = "\n".join(c.strip() for c in chunks).strip()
        return [merged] if merged else []
    return result


def verify_token_limit(chunks: list[str], count_tokens) -> list[tuple[int, str]]:
    """Split any chunk the real tokenizer says is over the model's limit.

    Returns (source_index, text) pairs so the caller can keep whatever metadata
    it had per chunk — a page number, say — through a split that turns one chunk
    into several. That indirection is what lets a caller verify a whole document
    in ONE call instead of one call per page: `count_tokens` reaches the model
    over HTTP, and that model also serves search queries, so every avoidable
    round trip is contention on the path that must stay decoupled.

    `count_tokens` takes a list of strings and returns (counts, limit), or None
    when token counts are not knowable (a provider whose tokenizer we do not
    have). None means: keep the character-sized chunks — safe only because that
    provider's limit is thousands of tokens away.

    Splitting is by characters proportional to the overshoot, then re-measured,
    because a token count gives no character offset to cut at. Two rounds are
    enough in practice; the loop is bounded so a pathological input degrades
    into smaller chunks instead of hanging.
    """
    items = list(enumerate(chunks))
    for _ in range(_MAX_VERIFY_ROUNDS):
        if not items:
            return []
        measured = count_tokens([text for _, text in items])
        if measured is None:
            return items
        counts, limit = measured
        if not limit or all(n <= limit for n in counts):
            return items
        out: list[tuple[int, str]] = []
        for (source, text), n in zip(items, counts):
            if n <= limit:
                out.append((source, text))
                continue
            # Aim a little under the limit so the next measurement is not a coin
            # flip on the boundary. No MIN_CHUNK_CHARS floor on `size`: with one,
            # a chunk that is still over the limit at 80 characters stops
            # shrinking, every round returns it unchanged, and the loop exits
            # having guaranteed nothing.
            parts = max(2, math.ceil(n / (limit * 0.9)))
            size = max(1, math.ceil(len(text) / parts))
            out.extend((source, part)
                       for part in _hard_split(text, size, merge_short_tail=False))
        # Only empties are dropped here. The MIN_CHUNK_CHARS filter belongs to
        # the initial chunking, where a short fragment is parsing noise; here
        # every fragment is part of a chunk that was already worth keeping.
        items = [(source, part.strip()) for source, part in out if part.strip()]
    measured = count_tokens([text for _, text in items])
    if measured is not None:
        counts, limit = measured
        over = [n for n in counts if limit and n > limit]
        if over:
            # Reached only by text so token-dense that splitting cannot keep up.
            # Saying so beats the silent truncation this function exists to stop.
            print(f"[chunking] {len(over)} chunk(s) still over the {limit}-token "
                  f"limit after {_MAX_VERIFY_ROUNDS} rounds (max {max(over)}); "
                  "these will be truncated by the embedding model")
    return items


def context_line(title: str | None, page_label: str, section: str | None) -> str:
    """A one-line header prepended to a chunk before embedding.

    This is contextual retrieval: a chunk that says "it improves recall by 12%"
    is unsearchable without knowing what "it" is. Generated from structure we
    already parsed, with a template and NOT with an LLM call — an LLM here would
    put the model in the ingest path, which is exactly the coupling channel
    non-negotiable 4 forbids.
    """
    parts = [p for p in (title, section, page_label) if p]
    return f"From {', '.join(parts)}:" if parts else ""
