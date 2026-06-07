# The Unofficial Guide — Project 1

A retrieval-augmented (RAG) question-answering system for **off-campus student
housing**. It scrapes housing guides, chunks and embeds them, retrieves the most
relevant passages for a question, and uses an LLM to answer **only** from those
passages — declining when the corpus doesn't cover the question.

Run it: `python app.py` (launches the Gradio UI at http://localhost:7860).
Other stages: `python app.py ask "..."`, `python app.py test`, `python app.py all`.

---

## Domain

This system covers **off-campus housing for college students** — monthly rent
ranges and hidden costs, getting along with roommates, choosing a safe
neighborhood, and weighing location/commute tradeoffs.

This knowledge is valuable and hard to find through official channels because
universities publish information about *on-campus* dorms, not the off-campus
rental market. The real guidance — what utilities actually cost on top of rent,
how to vet a landlord, how to write a roommate agreement, how to judge whether a
neighborhood is safe — lives scattered across blogs, relocation services,
international-student resources, and student forums. This project pulls that
scattered, unofficial knowledge into one searchable, grounded assistant.

---

## Document Sources

Ten sources were collected. Eight were scraped and cleaned successfully into the
corpus; two could not be ingested (noted below) — an honest limitation discussed
in the Spec Reflection.

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | Find My Place — cost breakdown | Blog article (HTML) | https://findmyplace.co/blog/off-campus-student-housing-costs-breakdown/ |
| 2 | CollegeRentals — Minneapolis listings | Rental portal (HTML) | https://www.collegerentals.com/off-campus-housing/mn/minneapolis/university-of-minnesota-twincities/ |
| 3 | illustrarch — complete student-housing guide | Blog article (HTML) | https://illustrarch.com/schooling/41550-complete-guide-to-student-housing-in-the-usa.html |
| 4 | University Living — international students | Blog article (HTML) | https://www.universityliving.com/blog/accommodation/off-campus-accommodation-in-us-for-international-students/ |
| 5 | Blueground — top US student cities | Blog article (HTML) | https://www.theblueground.com/blog/find-your-home/student-housing-top-usa-cities/ |
| 6 | Eduvouchers — accommodation in the USA | Blog article (HTML) | https://eduvouchers.com/blogs/students-diary/student-accommodation-in-usa |
| 7 | Zolve — finding safe & comfortable housing | Blog article (HTML) | https://zolve.com/blog/how-to-find-safe-and-comfortable-student-housing-in-the-usa/ |
| 8 | Outpost — surviving a student community | Blog article (HTML) | https://outpost.me/blog/how-to-survive-in-a-student-community |
| 9 | Diggz — living with roommates in college | Blog article (HTML) | https://blog.diggz.co/living-with-roommates-in-college-the-unofficial-requirement/ |
| 10 | r/uofmn | Reddit forum | https://www.reddit.com/r/uofmn/ |

**Ingestion notes (honest):**
- **#3 (illustrarch)** returned **HTTP 403 (bot-blocked)** and produced no text.
- **#10 (r/uofmn)** returned only a JavaScript shell (no server-rendered post
  text), so it produced 0 chunks.
- The remaining **8 sources** produced **87 chunks** total. The pipeline already
  supports local files, so these two can be added later by saving the pages as
  HTML into `documents/` and re-running.

---

## Chunking Strategy

**Chunk size:** 230 tokens *(revised down from a planned 500 — see below)*

**Overlap:** 35 tokens (~15%)

**Preprocessing before chunking:** Each page's raw HTML is saved to
`documents/raw/`, then cleaned: `<script>/<style>/nav/header/footer/aside/form`
tags are dropped, HTML entities are unescaped (`&amp;` → `&`, `&nbsp;` → space),
whitespace is collapsed, and link-soup fragments (short paragraphs with no
sentence punctuation, e.g. "Austin", "View all", "Contact") are removed. A
second pass drops any short paragraph that repeats across ≥3 pages (site
headers/footers). Cleaned text is written to `documents/clean/` for inspection.

**Why these choices fit the documents:** The corpus is mostly long-form guide
articles organized as discrete tips. Chunking is **paragraph-aware** — whole
paragraphs are packed together up to the token limit rather than cut
mid-sentence — so each tip stays intact and on a single topic. The ~15% overlap
carries a sentence or two across boundaries.

**Why 230 instead of 500 (revision during implementation):** The embedding model
`all-MiniLM-L6-v2` truncates any input longer than **256 tokens**. With the
originally planned 500-token chunks, roughly the back half of every chunk would
never be embedded, and retrieval would silently ignore it. Chunk size was
lowered to 230 (overlap 35) to stay safely under 256 with headroom for the
model's special tokens. Token counts are measured with the all-MiniLM-L6-v2
tokenizer itself, so "tokens" mean exactly what they mean at embedding time.

**Final chunk count:** **87 chunks** across 8 documents (min 36, max 229,
avg 197 tokens — no chunk is truncated at embedding time).

---

## Sample Chunks

Five representative chunks, each labeled with its source document. (Curly quotes
and dashes are preserved from the originals; chunks are stored in `chunks.jsonl`.)

**1. `findmyplace-costs#8`** — *source: findmyplace-costs*
> By-the-bed leases. You sign for your bedroom only, not the whole unit. Price is
> similar to a private room ($500–$1,000), but if a roommate stops paying rent,
> the property chases them — not you. That insulation is worth hunting for,
> especially if you're signing with people you don't know well. Professionally
> managed student complexes are where these live.

**2. `diggz-roommates#4`** — *source: diggz-roommates*
> One other student shares a smart way their suite handled it: "Establish a list
> of 'suite rules.' They don't need to feel harsh. For example, one suitemate
> didn't like people wearing shoes in their room, so now we don't. Another had
> food they wanted to share with us, and some food just for them. It's really
> important to just talk about what you like…"

**3. `eduvouchers-accommodation#3`** — *source: eduvouchers-accommodation*
> $400 – $2,000 per month (varies by location and whether shared or single).
> Renting an apartment off-campus is a popular choice among students seeking
> independence and flexibility. This option allows students to select living
> arrangements that suit their personal preferences, whether by living alone or
> sharing the space with roommates to minimize costs.

**4. `universityliving-intl#1`** — *source: universityliving-intl*
> What are Off-Campus Student Accommodations? Off-campus student accommodations
> refer to housing options that are not offered by the universities and are
> situated outside the campus area. This can include private rental apartments,
> homestays, shared rooms, or other rental properties where students can live
> while attending classes.

**5. `zolve-safety#5`** — *source: zolve-safety*
> Research the neighborhood using tools like NeighborhoodScout or City-Data, and
> check reviews from current or former residents. Choosing housing near your
> university is often safer. 4. What should I look for in a rental agreement?
> Check for details on the rental period, security deposit, rent amount, included
> utilities, and maintenance responsibilities…

*(Note: chunk #5 visibly merges a safety tip with the start of an unrelated
rental-agreement FAQ item — this mixed-topic chunk is the root of the failure
case documented below.)*

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers` (384-dimensional
vectors), stored in **ChromaDB** with cosine distance. It runs locally with no
API key and no rate limits, and is well-suited to the short prose chunks here.
Retrieval returns **top-k = 4** chunks per query.

**Production tradeoff reflection:** If cost weren't a constraint, I'd weigh a
larger or API-hosted model. A model like OpenAI `text-embedding-3-large` would
likely improve accuracy on domain-specific phrasing (the failure case below is
partly an embedding-resolution problem). A **multilingual** model would matter
because one of my sources targets international students who may query in another
language — MiniLM is English-centric. The cost of those models is **latency** and
**dependence on a paid API** versus MiniLM's instant, free, offline inference.
MiniLM's biggest real limitation here is its **256-token input cap**, which
directly shaped my chunk size; a model with a longer context window would let me
use larger chunks that carry more semantic context per embedding.

---

## Retrieval Test Results

Three queries run through `python app.py query "..."`, showing the top-4 chunks
and their cosine distances (lower = more relevant).

**Query A — "What is a realistic monthly rent range… and what costs beyond base
rent should I budget for?"**

| distance | chunk |
|---|---|
| 0.181 | `universityliving-intl#1` |
| 0.184 | `eduvouchers-accommodation#8` |
| 0.189 | `eduvouchers-accommodation#3` |
| 0.190 | `eduvouchers-accommodation#2` |

*Why these are relevant:* All four distances are very low (~0.18), and the chunks
are squarely about off-campus accommodation types, costs, and what to evaluate —
`eduvouchers-accommodation#3` literally contains the "$400 – $2,000 per month"
range that answers the rent half of the question. The retrieval is on-topic; the
only gap is that the most *detailed* hidden-cost chunk (`findmyplace-costs`)
ranked just outside the top 4, which is why the generated answer named utilities
and furnishings but not the full deposit/fees/insurance list.

**Query B — "What are concrete tips for getting along with roommates in shared
student housing?"**

| distance | chunk |
|---|---|
| 0.248 | `diggz-roommates#4` |
| 0.298 | `diggz-roommates#11` |
| 0.335 | `outpost-community#9` |
| 0.345 | `diggz-roommates#12` |

*Why these are relevant:* Three of the four chunks come from `diggz-roommates`,
my dedicated roommate-advice document, and the fourth (`outpost-community#9`)
covers vetting potential roommates. `diggz-roommates#4` is the "suite rules"
chunk and `#12` is about resolving disagreements — directly the "getting along"
intent. Low distances (0.25–0.35) and a single dominant source indicate tight,
on-topic retrieval. This query is the system's best-performing case.

**Query C — "How can I tell whether an off-campus housing option is safe?"**

| distance | chunk |
|---|---|
| 0.286 | `universityliving-intl#5` |
| 0.349 | `blueground-cities#2` |
| 0.356 | `blueground-cities#0` |
| 0.362 | `eduvouchers-accommodation#4` |

This query retrieves *generally* housing-related chunks but **misses the
dedicated safety document** (`zolve-safety` ranks only 9th) — analyzed in detail
in the Failure Case Analysis.

---

## Grounded Generation

**LLM:** Groq `llama-3.3-70b-versatile` (free-tier, OpenAI-compatible),
called with `temperature=0` and the `GROQ_API_KEY` loaded from `.env`.

Grounding is enforced with **three mechanisms, none of which rely on the model's
goodwill** (see `ask()` / `generate_answer()` in `app.py`):

1. **Relevance gate (structural).** If the closest retrieved chunk's cosine
   distance exceeds **0.65**, the system returns the refusal **without calling
   the LLM at all** — so it physically cannot answer from training knowledge for
   off-topic questions.
2. **Strict system prompt.** The model is instructed to *"answer using ONLY the
   information in the context documents… never use outside or prior knowledge,
   and never guess"*, and — if the context is insufficient — to reply with the
   exact sentence **"I don't have enough information on that."**
3. **Programmatic source attribution.** Source names are taken from the retrieved
   chunks' metadata and attached in code; they are **never invented by the
   model**. (If the model itself returns the refusal sentence, no sources are
   attached.)

**System prompt grounding instruction (excerpt):**
> "Use only facts stated in the context. Never use outside or prior knowledge,
> and never guess or fill in gaps. If the context does not contain enough
> information to answer, reply with exactly this sentence and nothing else:
> 'I don't have enough information on that.'"

**How source attribution is surfaced:** Each answer is accompanied by a
deduplicated list of the source documents the retrieved chunks came from (shown
in the UI's "Retrieved from" box and printed by `python app.py ask`). Retrieved
context blocks are also labeled `[Document N — source: <name>]` when passed to
the model.

---

## Example Responses

Two grounded answers with source attribution visible in the output, plus one
out-of-scope query showing the refusal. (Captured from `python app.py ask "..."`.)

**Example 1 — in-scope (budgeting):**
```
Q: How much should I budget per month for off-campus housing and what extra costs are there?

ANSWER:
You should budget $300 to $2,500 per person per month for off-campus housing,
with most students paying around $800 for a private bedroom. Extra costs to
consider include $100–$300 in hidden fees, as well as utilities and furnishings.

SOURCES:
  • findmyplace-costs
  • eduvouchers-accommodation
```

**Example 2 — in-scope (roommates):**
```
Q: What are some tips for getting along with roommates?

ANSWER:
1. Establish a list of 'suite rules' / a shared doc of preferences and responsibilities.
2. Talk to someone else first to focus the issue before raising it.
3. Talk to the person one-on-one to avoid awkward confrontations.
4. Invest in time outside your shared space.
5. Establish written agreements outlining rent distribution and rules for shared spaces.

SOURCES:
  • diggz-roommates
  • outpost-community
```

**Example 3 — out-of-scope (refusal):**
```
Q: Who won the 2022 FIFA World Cup final?

ANSWER:
I don't have enough information on that.

SOURCES:
  (none)
```
The refusal is produced by the **relevance gate** — the closest chunk's cosine
distance exceeded 0.65, so the system declined without ever calling the LLM, and
therefore attached no sources.

---

## Query Interface

The interface is a **Gradio web app** (`python app.py` → http://localhost:7860).

**Input field:**
- **"Your question"** — a single-line textbox. Submit by clicking **Ask** or
  pressing **Enter**.

**Output fields:**
- **"Answer"** (8-line textbox) — the grounded answer, or the refusal sentence.
- **"Retrieved from"** (4-line textbox) — a bulleted list of the source documents
  the answer drew from (empty/"(no sources)" on a refusal).

**Sample interaction transcript:**
```
[Your question]   How can I tell whether housing is safe?

[Ask] ▸

[Answer]
To determine whether an off-campus housing option is safe, research the safety of
the neighborhood — looking for areas with low crime rates — and consider secure
buildings with features like gated entries or security cameras.

[Retrieved from]
• universityliving-intl
• blueground-cities
• eduvouchers-accommodation
```
The interface needs no explanation to operate: type a question, press Ask, read
the answer and the sources it came from.

---

## Evaluation Report

All five planning.md questions were run through the live system
(`python app.py ask`). Retrieval distances are cosine distances (lower = closer).

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | Realistic monthly rent range + costs beyond base rent? | Rent varies; budget utilities, internet, deposit, fees, renter's insurance, furnishing | "$400–$2,000/month; budget for utilities and furnishings." (sources: universityliving, eduvouchers) | Relevant (0.18–0.19) | **Partially accurate** — correct range, but only named utilities + furnishings; omitted deposit/fees/insurance the corpus does contain |
| 2 | What should an international student check before signing a lease? | Lease terms/length, guarantor/cosigner or extra deposit, proximity, utilities included, landlord legitimacy | Lease terms, deposit rules, tenant rights, termination clauses, what's included (utilities), guarantor requirement, neighborhood safety | Partially relevant (0.38–0.47) | **Accurate** |
| 3 | Concrete tips for getting along with roommates? | Roommate agreement, split bills in writing, cleaning/quiet hours, communicate directly, respect shared space | Suite rules / written roommate pact, filter for compatibility, one-on-one communication, agreements on rent/utilities/shared spaces | Relevant (0.25–0.35) | **Accurate** |
| 4 | How can I tell whether a housing option is safe? | Crime data, secure entries/locks/lighting, near campus/transit, landlord/building reviews, visit in person | "Research neighborhood crime rates; look for secure buildings (gated entries, cameras)." | Partially relevant (0.29–0.36) | **Partially accurate** — thin; missed in-person visits, resident reviews, crime-data tools (see Failure Case) |
| 5 | Factors when choosing housing location? | Commute/distance, transit, groceries/amenities, safety, rent rises closer to campus | Proximity to campus, transportation/transit, community atmosphere, safety, parking | Relevant (0.24–0.34) | **Accurate** (minor omissions: amenities, the rent-vs-proximity tradeoff) |

**Summary:** 3 accurate, 2 partially accurate, 0 inaccurate. No hallucinations
observed; an out-of-domain control question ("Who won the 2022 FIFA World Cup?")
was correctly **declined** with no sources.

---

## Failure Case Analysis

**Question that failed:** Q4 — *"How can I tell whether an off-campus housing
option is safe?"* (partially accurate / weak retrieval).

**What the system returned:** A thin answer — research neighborhood crime rates
and look for secure buildings — missing the concrete, actionable safety advice
the corpus actually contains (using crime-data tools like NeighborhoodScout /
City-Data, reading reviews from current/former residents, choosing housing near
campus, visiting in person).

**Root cause (tied to a specific pipeline stage — chunking → embedding →
retrieval):** My corpus contains a *dedicated* safety document, `zolve-safety`
("How to Find Safe and Comfortable Student Housing"). Yet for this query it ranks
only **9th** (distance 0.411), below the **top-k = 4** cutoff, so it never
reaches the LLM. Worse, the single most relevant passage in the entire corpus —
`zolve-safety#5`: *"Research the neighborhood using tools like NeighborhoodScout
or City-Data, and check reviews from current or former residents. Choosing
housing near your university is often safer."* — doesn't even make the **top 12**.

The reason is a **chunk-boundary / mixed-topic dilution** problem: that 230-token
chunk merges the *tail* of the safety-tips list with the *start* of an unrelated
FAQ item ("4. What should I look for in a rental agreement? Check for…"). Because
the chunk is half about neighborhood safety and half about lease agreements, its
embedding sits *between* the two topics. Its cosine distance to a focused
"is this housing safe?" query is therefore weaker than four broader chunks from
other documents that happen to phrase "safety/neighborhood/secure" in more
query-like prose. The specific, high-value safety tip is effectively buried.

**What I would change to fix it:** Chunk on **FAQ-item / heading boundaries** so a
numbered tip and an FAQ question never share a chunk, keeping each chunk
single-topic and sharpening its embedding. Prepending the section heading to each
chunk's text (e.g. "Safety: …") would further bias the embedding toward the right
topic. Simply raising top-k wouldn't help here — the key chunk ranks outside the
top 12, so the problem is embedding resolution, not the cutoff.

---

## Spec Reflection

**One way the spec helped me during implementation:** Writing the Chunking
Strategy and Retrieval Approach sections *before* coding gave the AI tool precise,
verifiable targets — exact chunk size, overlap, embedding model, and top-k.
Because the spec named concrete numbers, I could check generated code against
them (e.g., confirm chunks were actually ≤ the token limit and that top-k=4 was
wired through), instead of accepting whatever the model produced. The
architecture diagram also let me ask for one pipeline stage at a time, which kept
the generated code aligned with the overall design.

**One way my implementation diverged from the spec, and why:** Two divergences.
First, I lowered chunk size from the planned **500 tokens to 230** once I learned
during implementation that `all-MiniLM-L6-v2` truncates input at 256 tokens —
500-token chunks would have left half of each chunk unembedded (planning.md's
Chunking Strategy was updated to record this). Second, my original architecture
sketch labeled the generator "Claude API," but the starter is built around
**Groq `llama-3.3-70b-versatile`**, so I aligned the implementation to Groq.
Separately, two of the ten planned sources (illustrarch, r/uofmn) couldn't be
ingested (403 / JavaScript-only), so the live corpus is 8 documents rather than
10 — a reminder that real-world scraping is lossier than a source list suggests.

---

## AI Usage

**Instance 1 — Ingestion & chunking**

- *What I gave the AI:* My planning.md Documents and Chunking Strategy sections
  plus the pipeline diagram, and asked it to implement a script that loads the
  sources, cleans them, and chunks them to my 500-token / 75-overlap spec.
- *What it produced:* A working fetch→clean→chunk script — but its first version
  counted chunks by **decoding token IDs**, which (because the MiniLM tokenizer
  is uncased and splits punctuation) mangled the stored text: `"$1,695"` became
  `"$ 1, 695"` and everything was lowercased. Inspection also showed leftover
  navigation menus ("athens austin … view all").
- *What I changed or overrode:* I directed it to store the **original text** and
  use token counts only to find boundaries, and to add link-soup / cross-page
  boilerplate removal after I read the cleaned output. I also **overrode the
  chunk size from 500 to 230** after it flagged the 256-token truncation limit.

**Instance 2 — Embedding, retrieval & grounded generation**

- *What I gave the AI:* My Retrieval Approach section and diagram, asking it to
  implement embedding with all-MiniLM-L6-v2, storage in ChromaDB with source
  metadata, a `retrieve()` function, and grounded Groq generation.
- *What it produced:* Embedding/ChromaDB/retrieval code and a generation function
  with a strict grounding system prompt. It went beyond the prompt by adding a
  **relevance gate** (decline without calling the LLM if the best distance > 0.65)
  and **programmatic** source attribution rather than trusting the model to cite.
- *What I changed or overrode:* I kept the relevance gate — it's what makes the
  out-of-domain refusal reliable. I also had to resolve an environment issue it
  surfaced: ChromaDB's `grpc` dependency was blocked by Windows **Smart App
  Control**; I chose to disable SAC (a permanent system change) to keep ChromaDB
  as planned rather than swap vector stores.

---

## Demo Video
https://www.loom.com/share/5db9544f62454f74b6f667771e87b650

