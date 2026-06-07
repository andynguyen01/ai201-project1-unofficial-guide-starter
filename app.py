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
import os
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

CHROMA_DIR = ROOT / "chroma_db"     # ChromaDB persists the index here
COLLECTION_NAME = "housing"
TOP_K = 4                           # chunks returned per query

# Milestone 5 — grounded generation
GROQ_MODEL = "llama-3.3-70b-versatile"
# If the closest chunk is farther than this cosine distance, the corpus
# almost certainly doesn't cover the question — decline rather than letting
# the LLM improvise from its training knowledge.
RELEVANCE_THRESHOLD = 0.65
NO_ANSWER = "I don't have enough information on that."

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

    text = html.unescape(text)          
    text = text.replace("\xa0", " ")    

    paragraphs: list[str] = []
    for block in text.split("\n\n"):
        block = re.sub(r"[ \t]+", " ", block)
        block = re.sub(r"\s*\n\s*", " ", block).strip()
        if not block or _JUNK_LINE.match(block):
            continue
    
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
    step = max(1, len(chunks) // n)
    sample = chunks[::step][:n]
    print("\n" + "=" * 70)
    print(f"{n} REPRESENTATIVE CHUNKS (of {len(chunks)} total)")
    print("=" * 70)
    for c in sample:
        print(f"\n--- {c.chunk_id}  ({c.token_count} tokens) ---")
        print(c.text)


# --- 5. Driver ------------------------------------------------------------- #

def run_ingest() -> None:
    """Milestone 3 — fetch, clean, chunk, and write chunks.jsonl."""
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


# --- 6. Embedding + vector store (Milestone 4) ----------------------------- #

_model = None  

def get_model():
    """Load all-MiniLM-L6-v2 once (local, no API key, no rate limits)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model {EMBED_MODEL} ...")
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _load_chunks() -> list[dict]:
    if not OUTPUT_FILE.exists():
        raise SystemExit("chunks.jsonl not found — run `python app.py ingest` first.")
    with OUTPUT_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run_index() -> None:
    """Embed every chunk and (re)build the ChromaDB collection.

    Each record stores the chunk text, its embedding, and metadata
    (source document name + position in that document) needed for
    attribution later."""
    import chromadb

    chunks = _load_chunks()
    model = get_model()

    print(f"Embedding {len(chunks)} chunks ...")
    embeddings = model.encode(
        [c["text"] for c in chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embeddings,
        metadatas=[{"source": c["source"], "chunk_index": c["chunk_index"]}
                   for c in chunks],
    )
    print(f"Indexed {collection.count()} chunks into ChromaDB at "
          f"{CHROMA_DIR.name}/ (collection '{COLLECTION_NAME}').")


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """Return the top-k most relevant chunks for a query, each with its
    source name, position, and cosine distance (lower = more relevant)."""
    import chromadb

    model = get_model()
    query_embedding = model.encode([query], normalize_embeddings=True).tolist()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    res = collection.query(query_embeddings=query_embedding, n_results=k)

    results = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0],
                               res["distances"][0]):
        results.append({
            "text": doc,
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "distance": dist,
        })
    return results


TEST_QUERIES = [
    "What is a realistic monthly rent range for off-campus student housing, "
    "and what costs beyond base rent should I budget for?",
    "What are concrete tips for getting along with roommates in shared "
    "student housing?",
    "How can I tell whether an off-campus housing option is safe?",
]


def run_test(k: int = TOP_K) -> None:
    """Run sample queries and print returned chunks + distance scores so we
    can judge whether retrieval is actually on-topic."""
    for query in TEST_QUERIES:
        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)
        for r in retrieve(query, k=k):
            flag = "  <-- weak match (>0.6)" if r["distance"] > 0.6 else ""
            print(f"\n[distance {r['distance']:.3f}] "
                  f"{r['source']}#{r['chunk_index']}{flag}")
            snippet = r["text"][:280].replace("\n", " ")
            print(f"  {snippet}{'...' if len(r['text']) > 280 else ''}")


# --- 7. Grounded generation (Milestone 5) ---------------------------------- #
#
# Pipeline diagram stage 5:
#   question --[retrieve]--> chunks --[Groq llama-3.3-70b]--> grounded answer
#
# Grounding is enforced two ways, neither of which trusts the LLM's goodwill:
#   1. A relevance gate — if retrieval finds nothing close enough, we return
#      the "not enough information" answer WITHOUT calling the LLM at all, so
#      it can't improvise from training knowledge.
#   2. A strict system prompt that forbids outside knowledge and mandates the
#      exact fallback sentence.
# Source attribution is added programmatically from retrieval metadata — it is
# never left to the model to invent.

SYSTEM_PROMPT = (
    "You are an assistant that answers questions about off-campus student "
    "housing. You must answer using ONLY the information in the context "
    "documents provided by the user. Follow these rules strictly:\n"
    "1. Use only facts stated in the context. Never use outside or prior "
    "knowledge, and never guess or fill in gaps.\n"
    f"2. If the context does not contain enough information to answer, reply "
    f"with exactly this sentence and nothing else: \"{NO_ANSWER}\"\n"
    "3. Be concise and factual. Do not invent sources, numbers, or details "
    "that are not in the context."
)


def _build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks as numbered, source-labeled blocks so the
    model sees exactly which document each fact came from."""
    return "\n\n".join(
        f"[Document {i} — source: {c['source']}]\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    )


def _unique_sources(chunks: list[dict]) -> list[str]:
    """Distinct source names in retrieval order (for attribution)."""
    seen: list[str] = []
    for c in chunks:
        if c["source"] not in seen:
            seen.append(c["source"])
    return seen


def generate_answer(question: str, chunks: list[dict]) -> str:
    """Call Groq with the retrieved context and the grounding system prompt."""
    from dotenv import load_dotenv
    from groq import Groq

    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_key_here":
        raise SystemExit("Set GROQ_API_KEY in .env (see .env.example).")

    user_message = (
        f"Context documents:\n\n{_build_context(chunks)}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context documents above."
    )
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0,          # deterministic, no creative drift
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content.strip()


def ask(question: str, k: int = TOP_K) -> dict:
    """End-to-end: retrieve -> (relevance gate) -> generate -> attribute.

    Returns {"answer": str, "sources": list[str], "chunks": list[dict]}.
    """
    chunks = retrieve(question, k=k)

    # Relevance gate: nothing close enough -> decline without calling the LLM.
    if not chunks or chunks[0]["distance"] > RELEVANCE_THRESHOLD:
        return {"answer": NO_ANSWER, "sources": [], "chunks": chunks}

    answer = generate_answer(question, chunks)

    # If the model itself declined, don't attach misleading sources.
    sources = [] if answer.strip() == NO_ANSWER else _unique_sources(chunks)
    return {"answer": answer, "sources": sources, "chunks": chunks}


# --- 8. Interface (Gradio web UI) ------------------------------------------ #

def handle_query(question: str):
    question = (question or "").strip()
    if not question:
        return "Please enter a question.", ""
    result = ask(question)
    sources = "\n".join(f"• {s}" for s in result["sources"]) or "(no sources)"
    return result["answer"], sources


def run_ui() -> None:
    import gradio as gr

    with gr.Blocks(title="The Unofficial Guide — Off-Campus Housing") as demo:
        gr.Markdown(
            "# The Unofficial Guide\n"
            "Ask about off-campus student housing — pricing, roommates, "
            "safety, and choosing a place. Answers come only from the "
            "collected guides; if they don't cover your question, the system "
            "will say so."
        )
        inp = gr.Textbox(label="Your question",
                         placeholder="e.g. How much should I budget per month?")
        btn = gr.Button("Ask", variant="primary")
        answer = gr.Textbox(label="Answer", lines=8)
        sources = gr.Textbox(label="Retrieved from", lines=4)
        btn.click(handle_query, inputs=inp, outputs=[answer, sources])
        inp.submit(handle_query, inputs=inp, outputs=[answer, sources])

    demo.launch()


# --- 9. Command dispatch --------------------------------------------------- #

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    cmd = sys.argv[1] if len(sys.argv) > 1 else "ui"

    if cmd == "ui":              # Milestone 5: launch the Gradio web UI (default)
        run_ui()
    elif cmd == "ingest":        # Milestone 3 only
        run_ingest()
    elif cmd == "index":         # Milestone 4: embed + store
        run_index()
    elif cmd == "test":          # Milestone 4: retrieval sanity check
        run_test()
    elif cmd == "ask":           # Milestone 5: end-to-end answer in the terminal
        if len(sys.argv) < 3:
            raise SystemExit('Usage: python app.py ask "your question here"')
        result = ask(sys.argv[2])
        print("\nANSWER:\n" + result["answer"])
        print("\nSOURCES:")
        for s in result["sources"]:
            print(f"  • {s}")
        if not result["sources"]:
            print("  (none)")
    elif cmd == "query":         # retrieval-only debug view
        if len(sys.argv) < 3:
            raise SystemExit('Usage: python app.py query "your question here"')
        for r in retrieve(sys.argv[2]):
            print(f"[{r['distance']:.3f}] {r['source']}#{r['chunk_index']}")
            print(f"  {r['text'][:280]}\n")
    elif cmd == "all":           # full pipeline: ingest -> index -> test
        run_ingest()
        print("\n" + "#" * 70 + "\n# EMBEDDING + INDEXING\n" + "#" * 70)
        run_index()
        print("\n" + "#" * 70 + "\n# RETRIEVAL TEST\n" + "#" * 70)
        run_test()
    else:
        raise SystemExit(f"Unknown command '{cmd}'. "
                         "Use: ui | ingest | index | test | ask | query | all")


if __name__ == "__main__":
    main()
