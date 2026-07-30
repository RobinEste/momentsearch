"""PDF -> pages. The parser seam for the document branch.

Everything downstream (chunking, embedding, the page locator) depends on this
module ONLY through `parse_pdf(data) -> list[Page]`. That is deliberate: the
choice of parser is an open question we intend to settle with a measurement
(pypdfium2 vs Docling on the same corpus and the same chunker), and a seam is
what makes that an A/B instead of a rewrite.

Why pypdfium2 today, measured on two real arXiv papers (2312.10997 single
column, 1908.10084 two-column ACL):
  * 2.0 ms/page text extraction, so an 800-page backfill costs ~1.6s of CPU on
    the very path that non-negotiable 4 wants decoupled from search
  * it also renders pages to images (~6 ms/page), which the deck branch needs
    for slides without a text layer — one dependency covers both source types
  * BSD-3-Clause + Apache-2.0. PyMuPDF is faster still but is AGPL-or-commercial,
    which this Apache-2.0 fork should not inherit

What it does NOT give, both measured rather than assumed:
  * a reliable outline: 49 bookmarks in one of those papers, zero in the other.
    So a section path is a bonus, never a guarantee — see `_section_map`
  * clean prose: text comes out in content-stream order, so figure labels and
    captions land inside the body text ("preliminary artifacts survive"). The
    repeated-line stripping below removes running heads and page numbers; figure
    labels are left in, and recall@10 decides whether they need more work
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

# A line must repeat on at least this share of pages to count as a running
# head/foot rather than as content. Deliberately high: dropping a real sentence
# that happens to recur is worse than keeping a page number.
_REPEAT_SHARE = 0.5
# Below this, "repeats often" means nothing, and the absolute floor of three
# occurrences matters as much as the share: at four pages a 50% share is two
# hits, and a heading that also appears in the table of contents hits exactly
# twice. That would strip the real heading from the page it belongs to.
_MIN_PAGES_FOR_REPEAT = 6
_MIN_REPEAT_COUNT = 3
_MAX_REPEAT_LINE_CHARS = 120   # a long repeated line is more likely boilerplate prose

# Ceiling on pages, so a malicious or broken document cannot occupy a worker
# slot for the whole flow timeout. Generous for real papers and decks: the
# longest thing anyone would reasonably ingest here is a thesis.
MAX_PAGES = 1000


@dataclass(frozen=True)
class Page:
    """One page of a document. `number` is 1-based because it is a citation
    locator shown to a human, not an array index."""
    number: int
    text: str
    section: str | None = None


def _section_map(doc) -> dict[int, str]:
    """page index (0-based) -> heading path, from the PDF outline.

    Empty when the PDF carries no bookmarks, which is common enough that every
    caller must treat a missing section as normal rather than as an error.
    """
    entries: list[tuple[int, int, str]] = []
    try:
        for bookmark in doc.get_toc():
            dest = bookmark.get_dest()
            if dest is None:
                continue
            # A destination can exist and still resolve to no page — an outline
            # item pointing at a named destination the document never defines.
            # get_index() returns None there rather than raising, so this is not
            # caught by the except below; letting it through would put None in a
            # tuple that gets sorted against ints two lines down, and take the
            # whole document's ingest with it.
            index = dest.get_index()
            if index is None:
                continue
            title = (bookmark.get_title() or "").strip()
            if title:
                entries.append((index, bookmark.level, title))
    except Exception:
        return {}  # a malformed outline must never fail the ingest
    if not entries:
        return {}

    entries.sort(key=lambda e: (e[0], e[1]))
    mapping: dict[int, str] = {}
    path: list[str] = []
    cursor = 0
    for page_index in range(max(e[0] for e in entries) + 1):
        while cursor < len(entries) and entries[cursor][0] <= page_index:
            _, level, title = entries[cursor]
            path = path[:level] + [title]
            cursor += 1
        if path:
            mapping[page_index] = " > ".join(path)
    return mapping


def _repeated_lines(pages: list[str]) -> set[str]:
    """Lines that occur on most pages: running heads, footers, page numbers."""
    if len(pages) < _MIN_PAGES_FOR_REPEAT:
        return set()
    counts: Counter[str] = Counter()
    for text in pages:
        # count each line once per page, so a word repeated within one page does
        # not look like a running head
        counts.update({ln.strip() for ln in text.splitlines() if ln.strip()})
    threshold = max(_MIN_REPEAT_COUNT, int(len(pages) * _REPEAT_SHARE))
    return {line for line, n in counts.items()
            if n >= threshold and len(line) <= _MAX_REPEAT_LINE_CHARS}


def _clean(text: str, drop: set[str]) -> str:
    kept = [ln for ln in text.splitlines() if ln.strip() not in drop]
    # Collapse the whitespace the extractor leaves behind, but keep paragraph
    # breaks: the chunker splits on them.
    out = "\n".join(kept)
    out = re.sub(r"[ \t]+", " ", out)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def parse_pdf(data: bytes) -> list[Page]:
    """Bytes of a PDF -> one Page per page, in document order.

    Pages without a text layer come back with an empty `text` rather than being
    dropped: the caller needs to know a page exists but yielded nothing, because
    for a paper that is a warning sign and for a deck it is the signal to render
    the page and caption it instead.
    """
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(data)
    # The byte limit on the download does not bound the work: a PDF's page tree
    # can share content streams, so a few hundred KB can declare millions of
    # pages. Checked before iterating, because the iteration IS the cost.
    if len(doc) > MAX_PAGES:
        raise ValueError(f"Document has {len(doc)} pages, over the {MAX_PAGES} limit.")
    raw = [page.get_textpage().get_text_bounded() for page in doc]
    drop = _repeated_lines(raw)
    sections = _section_map(doc)
    return [Page(number=i + 1, text=_clean(text, drop), section=sections.get(i))
            for i, text in enumerate(raw)]


# Page rendering for the deck branch belongs here too, next to parse_pdf, so the
# deck flow never has to know which PDF library is in use either. It is not
# written yet on purpose: measured at ~6 ms/page for pypdfium2, but shipping an
# uncalled, untested function ahead of the flow that needs it is the same
# mistake as listing 'deck' in INGEST_DEPLOYMENTS before its flow exists.
