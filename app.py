"""
Milestone 3 — Document ingestion and chunking
The Unofficial Guide (off-campus student housing)

Pipeline (matches planning.md):

    1. Load    — fetch the source URLs (or read local files), save the RAW
                 form to documents/raw/ before touching it.
    2. Clean   — strip HTML tags, scripts, nav/footer/cookie boilerplate,
                 unescape HTML entities (&amp; &nbsp; &#39;), collapse
                 whitespace. A second pass removes lines that repeat across
                 many pages (site headers/footers).
    3. Chunk   — paragraph-aware, 500-token chunks with 75-token overlap.
    4. Inspect — print one cleaned document and 5 representative chunks.

Token counts use the all-MiniLM-L6-v2 tokenizer so "tokens" mean the same
thing they will at embedding time (Milestone 4).

Outputs:
    documents/raw/<name>.html     raw downloaded pages (pre-cleaning)
    documents/clean/<name>.txt    cleaned text (human-readable, for inspection)
    chunks.jsonl                  final chunks with source metadata

Usage:
    python app.py
"""

from __future__ import annotations

import html
import json
import re
import sys
import urllib.request
from collections import Counter
from dataclasses import dataclass, asdict
from html.parser import HTMLParser
from pathlib import Path

from transformers import AutoTokenizer

# --- Configuration (matches planning.md) ----------------------------------- #

# 230 (not the originally planned 500) leaves headroom under all-MiniLM-L6-v2's
# 256-token input limit so no chunk is truncated at embedding time. See the
# updated reasoning in planning.md -> Chunking Strategy.
CHUNK_SIZE = 230          # tokens per chunk      (planning.md → Chunking)
CHUNK_OVERLAP = 35        # tokens of overlap     (planning.md → Chunking)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

ROOT = Path(__file__).parent
DOCUMENTS_DIR = ROOT / "documents"
RAW_DIR = DOCUMENTS_DIR / "raw"
CLEAN_DIR = DOCUMENTS_DIR / "clean"
OUTPUT_FILE = ROOT / "chunks.jsonl"

# all-MiniLM-L6-v2 truncates inputs to this many tokens at embedding time;
# text past it in a chunk will NOT be embedded. See the warning at the end.
MODEL_MAX_TOKENS = 256

# The 10 sources from planning.md → Documents.  (name, url)
SOURCES = [
    ("findmyplace-costs",        "https://findmyplace.co/blog/off-campus-student-housing-costs-breakdown/"),
    ("collegerentals-portal",    "https://www.collegerentals.com/off-campus-housing/mn/minneapolis/university-of-minnesota-twincities/?beds[]=1"),
    ("illustrarch-guide",        "https://illustrarch.com/schooling/41550-complete-guide-to-student-housing-in-the-usa.html"),
    ("universityliving-intl",    "https://www.universityliving.com/blog/accommodation/off-campus-accommodation-in-us-for-international-students/"),
    ("blueground-cities",        "https://www.theblueground.com/blog/find-your-home/student-housing-top-usa-cities/"),
    ("eduvouchers-accommodation","https://eduvouchers.com/blogs/students-diary/student-accommodation-in-usa"),
    ("zolve-safety",             "https://zolve.com/blog/how-to-find-safe-and-comfortable-student-housing-in-the-usa/"),
    ("outpost-community",        "https://outpost.me/blog/how-to-survive-in-a-student-community"),
    ("diggz-roommates",          "https://blog.diggz.co/living-with-roommates-in-college-the-unofficial-requirement/"),
    ("reddit-uofmn",             "https://www.reddit.com/r/uofmn/"),
]

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

# --------------------------------------------------------------------------- #


@dataclass
class Chunk:
    chunk_id: str        # e.g. "zolve-safety#3"
    source: str          # source name
    chunk_index: int     # position within its document
    token_count: int     # number of tokens in this chunk
    text: str            # the cleaned chunk text


# --- 1. Load: fetch raw pages and save them before cleaning ---------------- #

def fetch_all() -> None:
    """Download each source URL and save the raw HTML to documents/raw/.
    Cached: a source already on disk is not re-downloaded. Failures are
    logged and skipped so one dead URL doesn't stop the run."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES:
        dest = RAW_DIR / f"{name}.html"
        if dest.exists():
            print(f"  cached  {name}")
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
            dest.write_text(raw, encoding="utf-8")
            print(f"  fetched {name:<28} ({len(raw):,} bytes)")
        except Exception as exc:  # network error, 403, timeout, etc.
            print(f"  FAILED  {name:<28} {type(exc).__name__}: {exc}")


# --- 2. Clean -------------------------------------------------------------- #

class _HTMLTextExtractor(HTMLParser):
    """HTML -> text. Drops non-content tags and inserts blank lines after
    block tags so paragraph-aware chunking has boundaries to work with."""

    _SKIP = {"script", "style", "noscript", "head", "nav", "footer",
             "header", "aside", "form", "button", "svg", "iframe"}
    _BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6",
              "tr", "section", "article"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self._parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self._BLOCK:
            self._parts.append("\n\n")

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


# Lines that are almost always nav/UI boilerplate rather than content.
_JUNK_LINE = re.compile(
    r"^(read more|share|tweet|sign in|log ?in|sign ?up|subscribe|menu|home|"
    r"accept( all)?( cookies)?|cookie|we use cookies|follow us|next|previous|"
    r"back to top|©.*|all rights reserved.*|\d+ comments?|search)$",
    re.IGNORECASE,
)


def clean_html(raw: str, is_html: bool) -> list[str]:
    """Return a list of cleaned paragraph strings."""
    if is_html:
        parser = _HTMLTextExtractor()
        parser.feed(raw)
        text = parser.get_text()
    else:
        text = raw

    text = html.unescape(text)          # &amp; -> &, &#39; -> '
    text = text.replace("\xa0", " ")    # &nbsp; -> normal space

    paragraphs: list[str] = []
    for block in text.split("\n\n"):
        block = re.sub(r"[ \t]+", " ", block)
        block = re.sub(r"\s*\n\s*", " ", block).strip()
        if not block or _JUNK_LINE.match(block):
            continue
        # Keep only prose-like paragraphs. Menu/link fragments that aren't
        # wrapped in <nav> ("Austin", "View all", "Rent Better", "Contact")
        # are short and have no sentence punctuation, so this drops them
        # while keeping real sentences and headings like "...in the U.S.".
        word_count = len(block.split())
        has_sentence = bool(re.search(r"[.!?]", block))
        if word_count < 8 and not has_sentence:
            continue
        paragraphs.append(block)
    return paragraphs


def remove_repeated_boilerplate(docs: dict[str, list[str]],
                                min_docs: int = 3,
                                max_len: int = 80) -> dict[str, list[str]]:
    """Site headers/footers appear verbatim on many pages. Drop any short
    paragraph that shows up in >= min_docs different documents."""
    freq = Counter()
    for paragraphs in docs.values():
        for p in set(paragraphs):
            freq[p] += 1
    boilerplate = {p for p, n in freq.items() if n >= min_docs and len(p) <= max_len}
    if boilerplate:
        print(f"  removing {len(boilerplate)} boilerplate line(s) seen across pages")
    return {
        name: [p for p in paragraphs if p not in boilerplate]
        for name, paragraphs in docs.items()
    }


# --- Loading cleaned docs from disk ---------------------------------------- #

def load_documents() -> dict[str, list[str]]:
    """Clean every raw page plus any local files the user dropped in
    documents/. Returns {name: [paragraph, ...]}."""
    docs: dict[str, list[str]] = {}

    for path in sorted(RAW_DIR.glob("*.html")):
        paragraphs = clean_html(path.read_text(encoding="utf-8", errors="ignore"),
                                is_html=True)
        if paragraphs:
            docs[path.stem] = paragraphs

    # Any extra local files placed directly in documents/ (txt/md/pdf/html).
    for path in sorted(DOCUMENTS_DIR.glob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            docs[path.stem] = clean_html(path.read_text(encoding="utf-8", errors="ignore"),
                                         is_html=False)
        elif suffix in {".html", ".htm"}:
            docs[path.stem] = clean_html(path.read_text(encoding="utf-8", errors="ignore"),
                                         is_html=True)
        elif suffix == ".pdf":
            docs[path.stem] = clean_html(_read_pdf(path), is_html=False)

    return {name: p for name, p in docs.items() if p}


def _read_pdf(path: Path) -> str:
    try:
        import pdfplumber
    except ImportError as exc:
        raise SystemExit("Install pdfplumber to ingest PDFs: pip install pdfplumber") from exc
    with pdfplumber.open(path) as pdf:
        return "\n\n".join(page.extract_text() or "" for page in pdf.pages)


# --- 3. Chunk -------------------------------------------------------------- #

def _token_len(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _split_long_paragraph(para: str, tokenizer, chunk_size: int,
                          overlap: int) -> list[tuple[str, int]]:
    """Split one paragraph that is bigger than a chunk into word-based
    pieces of ~chunk_size tokens, preserving the ORIGINAL text. Word counts
    are scaled by this paragraph's tokens-per-word ratio, which is accurate
    enough for boundary-finding."""
    words = para.split()
    total = _token_len(para, tokenizer)
    if not words:
        return []
    tokens_per_word = total / len(words)
    words_per_chunk = max(1, int(chunk_size / tokens_per_word))
    overlap_words = max(0, int(words_per_chunk * overlap / chunk_size))
    step = max(1, words_per_chunk - overlap_words)

    pieces: list[tuple[str, int]] = []
    for start in range(0, len(words), step):
        piece_words = words[start:start + words_per_chunk]
        piece = " ".join(piece_words)
        pieces.append((piece, int(len(piece_words) * tokens_per_word)))
        if start + words_per_chunk >= len(words):
            break
    return pieces


def chunk_text(paragraphs: list[str], source: str, tokenizer,
               chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """Paragraph-aware chunking that preserves the original text.

    Token counts decide the boundaries (so chunks match the embedding
    model's view), but the stored text is the original prose — never a
    re-detokenized string, which would lowercase text and mangle numbers
    and punctuation. Whole paragraphs are packed until the next would
    overflow chunk_size; the last `overlap` tokens' worth of trailing
    paragraphs seed the next chunk. A lone oversized paragraph is split by
    words first."""
    # Build (text, token_count) units, splitting any oversized paragraph.
    units: list[tuple[str, int]] = []
    for para in paragraphs:
        n = _token_len(para, tokenizer)
        if n > chunk_size:
            units.extend(_split_long_paragraph(para, tokenizer, chunk_size, overlap))
        else:
            units.append((para, n))

    chunks: list[Chunk] = []
    current: list[tuple[str, int]] = []
    current_tokens = 0
    index = 0

    def flush() -> None:
        nonlocal current, current_tokens, index
        if not current:
            return
        text = "\n\n".join(t for t, _ in current).strip()
        chunks.append(Chunk(
            chunk_id=f"{source}#{index}",
            source=source,
            chunk_index=index,
            token_count=current_tokens,
            text=text,
        ))
        index += 1
        # Carry trailing units (up to `overlap` tokens) into the next chunk.
        kept: list[tuple[str, int]] = []
        kept_tokens = 0
        for t, n in reversed(current):
            if kept_tokens + n > overlap:
                break
            kept.insert(0, (t, n))
            kept_tokens += n
        current, current_tokens = kept, kept_tokens

    for text, n in units:
        if current and current_tokens + n > chunk_size:
            flush()
        current.append((text, n))
        current_tokens += n

    flush()
    return chunks


# --- 4. Inspection helpers ------------------------------------------------- #

def preview_document(name: str, paragraphs: list[str], max_chars: int = 1500) -> None:
    print("\n" + "=" * 70)
    print(f"CLEANED DOCUMENT PREVIEW: {name}")
    print("=" * 70)
    text = "\n\n".join(paragraphs)
    print(text[:max_chars] + ("\n... [truncated]" if len(text) > max_chars else ""))


def preview_chunks(chunks: list[Chunk], n: int = 5) -> None:
    if not chunks:
        return
    # Evenly spaced across the corpus so the sample is representative,
    # not just the first few chunks of the first document.
    step = max(1, len(chunks) // n)
    sample = chunks[::step][:n]
    print("\n" + "=" * 70)
    print(f"{n} REPRESENTATIVE CHUNKS (of {len(chunks)} total)")
    print("=" * 70)
    for c in sample:
        print(f"\n--- {c.chunk_id}  ({c.token_count} tokens) ---")
        print(c.text)


# --- 5. Driver ------------------------------------------------------------- #

def main() -> None:
    # Windows consoles default to cp1252 and choke on characters like "→"
    # that appear in page text; force UTF-8 output.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)

    print("\n[1/3] Fetching raw documents ...")
    fetch_all()

    print("\n[2/3] Cleaning ...")
    docs = load_documents()
    if not docs:
        raise SystemExit("No documents loaded. Check the fetch step above.")
    docs = remove_repeated_boilerplate(docs)

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    for name, paragraphs in docs.items():
        (CLEAN_DIR / f"{name}.txt").write_text("\n\n".join(paragraphs), encoding="utf-8")

    # Inspect one cleaned document before chunking.
    first = next(iter(docs))
    preview_document(first, docs[first])

    print("\n[3/3] Chunking "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}) ...")
    all_chunks: list[Chunk] = []
    for name, paragraphs in docs.items():
        doc_chunks = chunk_text(paragraphs, name, tokenizer)
        all_chunks.extend(doc_chunks)
        print(f"  {name:<28} -> {len(doc_chunks)} chunks")

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    preview_chunks(all_chunks, n=5)

    sizes = [c.token_count for c in all_chunks]
    over_limit = sum(1 for s in sizes if s > MODEL_MAX_TOKENS)
    print("\n" + "=" * 70)
    print(f"Done. {len(docs)} documents -> {len(all_chunks)} chunks "
          f"written to {OUTPUT_FILE.name}")
    if sizes:
        print(f"  token counts: min={min(sizes)} max={max(sizes)} "
              f"avg={sum(sizes) // len(sizes)}")
    if over_limit:
        print(
            f"\n  WARNING: {over_limit}/{len(all_chunks)} chunks exceed "
            f"{MODEL_MAX_TOKENS} tokens. all-MiniLM-L6-v2 truncates inputs at "
            f"{MODEL_MAX_TOKENS} tokens, so text past that point will NOT be "
            f"embedded. Consider setting CHUNK_SIZE = 256 (overlap ~40) and "
            f"updating planning.md if retrieval quality looks weak."
        )


if __name__ == "__main__":
    main()
