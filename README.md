# MIVI — MakeMyEducation's AI College Counsellor

MIVI ("mimi") answers students' college questions in English, Hindi and
Hinglish, grounded in MakeMyEducation's **live production catalogue**:

| | |
|---|---|
| Colleges | **38,700** (35,316 with course offerings) |
| Course offerings | **211,071** |
| Coverage | All Indian states + overseas institutions |
| Source | `makeMyEducation.courses` export + the live `admin.makemyeducation.com` API |

Every factual claim is traceable to a real college page:
a citation id **is** the public URL slug —
`makemyeducation.com/college/<college_id>`.

> This replaces the original 15-college prototype. That work is archived in
> [`docs/PROTOTYPE_REPORT.md`](docs/PROTOTYPE_REPORT.md); the retired data and
> the schema that no longer exists are documented in
> [`data/DEPRECATED.md`](data/DEPRECATED.md).

---

## Quickstart

```bash
pip install -r requirements.txt
```

Point at the courses export (169 MB, lives outside the repo):

```bash
export MME_COURSES_CSV="D:/make my education/colleg data cleaning/colleg data cleaning/makeMyEducation.courses (2).csv"
```

Build the corpus — four commands, all resumable, all rebuildable from source:

```bash
python -m rag_core.sources.crawl_colleges list
```
```bash
python -m rag_core.store.build_db
```
```bash
python -m rag_core.index_build
```
```bash
uvicorn app:app --port 8000
```

The optional enrichment crawl can run at any time; the corpus works without it:

```bash
python -m rag_core.sources.crawl_colleges detail
```

Ask a single question from the CLI:

```bash
python answer.py "Which colleges in Dehradun offer MBA under 2 lakh per year?"
```

---

## The problem this system had to solve

The courses export contains **no college identity at all** — no name, no city,
no state. The only reference is `college`, a Mongo ObjectId:

```
_id, title, courseFullName, tutionFees[0].startPrice, ..., college
69b905a1...,  MEng,  Master of Engineering,  3000000, ..., 69b9016f216756163868f1a2
```

That same ObjectId is what appears in the public URL when a student clicks a
college on the website. So the join is recovered from the site's own public
API (`rag_core/sources/mme_api.py`):

```
GET /api/user/college?limit=500&page=N   -> the whole 38,700-college catalogue
GET /api/user/college/<ObjectId>         -> per-college enrichment
```

**Measured join coverage: 99.88%** — 35,316 of 35,358 referenced colleges
resolve; the 42 misses are colleges deleted since the export.

---

## Architecture

```
courses CSV (211k rows, streamed)  ─┐
                                    ├─►  build_db  ─►  data/mivi.db  ─┐
live API crawl (38.7k colleges)    ─┘     (SQLite)                    │
                                                                      ├─► retrieve
                                            index_build ─► index.npz ─┘      │
                                             (38.7k × 384 dense)             ▼
                                                                         pipeline
                                                                             │
                                                                    app.py (FastAPI)
```

| Layer | Module | Role |
|---|---|---|
| Sources | `rag_core/sources/` | CSV normaliser, API client, resumable crawler |
| Store | `rag_core/store/` | SQLite schema, ETL, the canonical `render_card` |
| Index | `rag_core/index_build.py` | Dense embeddings, incremental via card hashes |
| Retrieval | `rag_core/retrieve.py` | SQL prefilter → BM25 + dense → RRF fusion |
| Reasoning | `rag_core/pipeline.py` | Route → retrieve → generate → **verify** |
| Serving | `app.py`, `rag_core/service.py` | Sessions, rate limiting, caching, threadpool |

### Why these choices

**SQLite, not a hosted vector DB.** The corpus is ~360 MB, read-mostly, and
rebuilt from source. One file means zero infra to provision, FTS5 gives real
BM25 in-process, and WAL mode serves many concurrent readers. The schema maps
1:1 onto Postgres + pgvector when it outgrows one box.

**Exact dense search, not ANN.** 38,700 × 384 float32 is ~57 MB — a single
matmul over it takes milliseconds, and it removes an entire class of
index-drift bugs. ANN becomes worth its complexity somewhere north of ~500k
vectors.

**One college = one chunk.** Students ask about *institutions* ("which colleges
in Dehradun offer MBA under 2 lakh"), and a college-level chunk keeps every
citation resolvable to a real page. Course detail is folded in as a rollup.

**Counting is SQL's job, never the model's.** "How many colleges in Uttarakhand
offer MBA" is answered by `count_matching()` — an exact `COUNT(*)`. The model
then presents the top N. At 15 records you could stuff everything into the
context and let the model count; at 35,300 that is neither possible nor honest.

---

## Correctness rules the data forces

These are measured properties of the live corpus, not preferences. They are
enforced in code because a wrong fee destroys a student's trust permanently.

**1. Overseas fees are not rupees.** 3,565 course rows are overseas listings
quoted in their own currency (confirmed: domestic rows carry `₹`, overseas
carry `$`). Treating a foreign "3,000,000" as ₹30 lakh would be catastrophic,
so rupee-typed columns are populated for domestic rows **only**, and searches
default to domestic.

**2. Fee `0` means "not recorded", not free.** Normalised to `NULL`, so it can
never win a "cheapest college" ranking.

**3. Fields this corpus does not have.** NAAC grade, NIRF ranking, placement
packages, scholarships, hostel details and admission cutoffs are empty across
effectively the whole catalogue. MIVI must answer `answered: false` and offer
an adjacent known fact — never invent a number. The prototype *had* these
columns, which makes this the single highest regression risk in the rewrite.

```
Q: What is the average placement package at IIT Roorkee?
A: We don't have placement package figures for IIT Roorkee — that one's best
   checked on their own placements page. What I can tell you: it's a government
   institution established in 1847, tuition runs Rs 800,000/year, and it takes
   GATE, CAT, JEE Advanced, JEE Main, NATA, AAT, IIT JAM.        [answered: false]
```

**4. Known data-quality caveat.** ~2.2% of domestic course rows carry
implausible tuition (under ₹1,000, some at ₹10) — clearly placeholder entries
in the source. These distort "cheapest" rankings, so a plausibility floor is
applied to superlative *ranking* only; a direct question about one of those
colleges still shows its recorded fee as-is.

---

## Filling the gaps: website harvesting

The catalogue has no scholarships, hostel charges or placements — but the
colleges publish that on their own sites. `rag_core/sources/web_enrich.py`
reads it:

```bash
python -m rag_core.sources.web_enrich --limit 25     # smoke test
```
```bash
python -m rag_core.sources.web_enrich --status       # progress
```

**Two tiers of truth, never merged.** Harvested facts land in
`college_web_facts` with their source URL, fetch date and an extraction
confidence — never in the curated tables. Cards render them under an explicit
`FROM THE COLLEGE'S OWN WEBSITE` heading, and the generator is required to
attribute them out loud ("their own site lists…"). A scholarship an LLM read
off a college PDF is a weaker claim than a verified tuition figure, and a
student deserves to know which is which. Promotion into the curated tables is
a separate, reviewed decision.

**Engineering choices.** Plain HTTP is the engine, not Firecrawl: the account
has ~1,000 credits/month, which covers under 1% of the corpus. Firecrawl stays
as the fallback for JS-rendered or blocking sites where a credit buys
something. robots.txt is honoured per host, requests are paced per-domain, and
concurrency is *across* domains — twelve workers on twelve colleges is
neighbourly; twelve on one small college's shared hosting is an outage.

The sweep is resumable via `web_crawl_state`, and `build_db` explicitly carries
both harvest tables across a rebuild — they represent days of politeness-limited
crawling and are the one thing in the store that cannot be re-derived in
seconds.

**The real bottleneck is LLM extraction, not fetching.** One call per page, and
both configured providers are on free tiers (Gemini: 15 requests/minute). At
~3 pages across the colleges that publish a usable site, a full sweep is on the
order of 70+ hours at free-tier rates. A paid tier on a small model turns that
into hours for a few dollars — that is the decision to make before running it
at full scale.

---

## Grounding and safety

Preserved verbatim from the prototype, because they were right:

- **Deterministic verification.** Cited ids must have been retrieved; a college
  named in the prose must be cited; every number in the answer must trace to a
  cited college's actual field. One corrective retry, then fail closed
  (`answered: false`) rather than ship an ungrounded answer.
- **Distress guard.** Self-harm language is caught in code *before* any model
  call and answered with care plus India's KIRAN helpline (1800-599-0019).
- **Prompt-injection deflection**, in code, before the model sees the message.
- **Zero-LLM fast path** for greetings and thanks.
- **Provider failover.** A rate-limited or hung primary fails over to a second
  provider mid-request; students never see the error.

---

## Website integration

`POST /api/chat` returns everything the site needs to render linked answers:

```jsonc
{
  "answer": "23 colleges in Dehradun offer an MBA under 2 lakh per year — here are a few: ...",
  "citations": ["69c503f521675616386a0380", "..."],
  "answered": true,
  "session_id": "...",
  "profile": { "marks_pct": null, "field_interest": "MBA", "budget": "2 lakh/year" },
  "colleges": [
    { "college_id": "69c503f521675616386a0380",
      "name": "Institute of Co-Operative Management, Dehradun",
      "city": "Dehradun", "state": "Uttarakhand",
      "url": "https://makemyeducation.com/college/69c503f521675616386a0380" }
  ]
}
```

Other endpoints: `GET /api/health` (corpus counts + readiness, cheap enough for
a load balancer), `GET /api/college/{id}`, and `POST /api/answer` for backwards
compatibility.

Serving notes: the pipeline is synchronous and CPU-bound on embedding, so it
runs in a threadpool with a concurrency cap — a traffic spike queues instead of
thrashing the event loop. Sessions, the answer cache and rate limiting are
in-process (single instance); Redis is the drop-in for multi-instance. CORS is
restricted to the makemyeducation.com origins by default.

---

## Repo map

```
rag_core/
  config.py            paths, models, retrieval depth, failover
  sources/
    mme_api.py         API client + AIMD adaptive throttle
    crawl_colleges.py  resumable list + detail crawlers
    courses_csv.py     streaming CSV normaliser
  store/
    schema.sql         colleges, courses, facts, aggregates, FTS5
    db.py              connections + render_card (the retrieval unit)
    build_db.py        ETL, atomic swap
  index_build.py       incremental dense index
  retrieve.py          SQL prefilter -> BM25 + dense -> RRF
  pipeline.py          route -> retrieve -> generate -> verify
  service.py           sessions, rate limiting, answer cache
  llm.py               provider abstraction + failover
app.py                 FastAPI service
answer.py              single-question CLI
evals/                 grounded eval set + runner
docs/                  archived prototype report
data/DEPRECATED.md     what was retired and why
```

## Rebuild cadence

`build_db` is a full rebuild from source in ~25 seconds and swaps atomically,
so a half-built corpus is never served. `index_build` is incremental — it
hashes each card and only re-embeds what changed, which is the difference
between a multi-hour first build and a seconds-long refresh.
