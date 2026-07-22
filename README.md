# MME RAG Prototype — Grounded College Q&A

A small Retrieval-Augmented Generation prototype that answers natural-language
questions (English / Hindi / Hinglish) about the 15-college sample dataset —
every factual claim cited by `college_id`, every unanswerable question refused
gracefully. Built for the Make My Education Applied AI Engineer take-home.

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env         # then open .env and paste your key (1 line)
python check_setup.py        # confirms deps + dataset + key are ready (plain English)
python answer.py "Which colleges offer an MBA, and what do they cost?"
```

`check_setup.py` is a 5-second pre-flight: it tells you in plain English if a
dependency, the dataset, or the API key is missing, so a setup issue is a
clear message here instead of a stack trace later. The only thing you must
supply is one API key (a free Google AI Studio key works — see `.env.example`).

Output is a single JSON object on stdout (diagnostics go to stderr):

```json
{"answer": "...", "citations": ["C002", "C004"], "answered": true, "reason_if_unanswered": null}
```

Other commands:

```bash
python run_all.py            # regenerates answers.md (the 7 published questions, verbatim)
python evals/run_evals.py    # runs the eval suite, prints pass rate + abstention score
python metrics/cost_report.py  # aggregates measured tokens/latency into the Part D table
python -m rag_core.ingest    # (re)builds the embedding index — also happens automatically
```

Provider: any OpenAI-compatible endpoint via `.env` — see `.env.example`.
This submission was measured on **Google Gemini** (`gemini-3.1-flash-lite`
for both routing and answers — chosen after measuring three providers; see
the latency section). NVIDIA NIM (Nemotron 49B) and Groq (Llama 3.3 70B)
are documented drop-in alternatives in `.env.example`.
The first run downloads the local embedding model (~450 MB) from Hugging Face.

## How it works

```
question
   │
   ▼
1. ROUTER  (small 8B model, JSON mode, temp 0)
   • data question / smalltalk / out-of-scope  → non-data routes skip
     retrieval AND the large model entirely (cost + latency saved)
   • extracts hard filters: budget, marks, type, hostel, city
   • normalises units: lakh → ₹, per-semester → per-year (×2, assumption stated)
   • flags dataset-wide questions (cheapest/best/how-many) → retrieval widens
     to ALL 15 records, because top-k truncation on a superlative produces a
     confidently wrong answer
   │
   ▼
2. HYBRID RETRIEVAL  (local multilingual-e5-small embeddings + numpy cosine)
   • hard filters applied in code — a college that fails budget/cutoff/type
     can never reach the model, so it can never be recommended
   • semantic ranking over one card per college, small lexical boost for
     exact course mentions ("MBA")
   │
   ▼
3. GENERATION  (large model — Nemotron 49B / Llama 70B, JSON mode, temp 0.2)
   • answers from the retrieved cards ONLY, [C0XX] citation per claim
   • replies in the language of the question
   │
   ▼
4. VERIFICATION  (deterministic, in code)
   • citations ⊆ retrieved context; answered=true requires citations
   • course exact-match: if the user literally named a program ("MBA"),
     every cited college must list it in courses_offered
   • name-backing: every college NAMED in the prose must be cited — catches
     the citation-dodge where a model drops the id but keeps the college
   • NUMERIC grounding: every rupee-fee and LPA figure in the answer must
     equal a cited college's actual field — a hallucinated "Rs 250,000" for a
     college that costs 98,000 is blocked, not shipped. Numbers are where a
     wrong answer destroys trust most and where LLMs are least reliable.
   • up to two corrective retries with the precise violation, then FAIL
     CLOSED: answered=false rather than shipping an unverified answer
   • id codes are scrubbed from the prose — students see college names,
     ids live in the citations array (matching the sample output in the brief)
```

## Key design choices (and why)

**Chunking — one college = one chunk.** Records are ~200 words, citations are
per-college, and no question needs sub-record granularity. Splitting further
would only fragment context; merging colleges would blur citations.

**Local embeddings (`intfloat/multilingual-e5-small`).** Free (₹0 per query and
₹0 to index), no API dependency, and natively multilingual — a Hindi question
retrieves the right English records without a translation hop. At 15 documents,
embedding quality differences are dwarfed by the hard filters anyway.

**Numpy in-memory index, not a vector DB.** 15 vectors need cosine similarity,
not infrastructure. The retrieval interface is one function; swapping in
pgvector/Qdrant at production scale changes `retrieve.py` only. (The production
design I'd target is Postgres + pgvector so facts and vectors live in one
store.)

**No cross-encoder reranker, and k defaults to the full corpus.** My first
version used k=8; the eval suite caught two recall failures where the correct
college never reached the model (enumeration questions whose evidence lives in
the `about` field). At 15 records the measured cost of full context is ~1.6k
extra input tokens (a fraction of a paisa) — so truncation bought nothing and
silently broke answers. Hard filters still trim the context, semantic ranking
still orders it. At thousands of records, k and a reranker (e.g. bge-reranker)
become real dials — tuned against recall evals, the same way this one was.

**Hard filters live in code, not in the prompt.** Numbers are where LLMs are
least trustworthy. Budget ceilings, the cutoff-as-hard-floor rule, government
/private/deemed, and hostel availability are enforced before generation: a
college failing them is physically absent from the model's context.

**Two models, routed by job.** Filter extraction is trivial → 8B (≈12× cheaper
than the 70B). Final answers need judgment → 70B. Smalltalk/out-of-scope never
touches retrieval or the 70B at all — the rubric's "which questions need
retrieval at all" is a routing decision, and it's also the cheapest one.

**Fail closed.** If the deterministic checks still fail after two corrective
retries, the system returns `answered: false` with a reason instead of an
unverifiable answer. In admissions, a wrong fee destroys trust; a polite
"I couldn't verify that" does not.

**Answers are verified by code, not vibes.** Four checks run on every
generated answer before it ships: citation validity (ids must come from the
retrieved context), course exact-match (a college cited for "MBA" must list
MBA — this is what keeps the two similar-named colleges apart), name-backing
(a college named in the text must be cited), and — the strongest one —
**numeric grounding**: every rupee fee and LPA figure in the answer is
matched against the cited college's actual field, so a hallucinated fee is
caught by code, not hoped away by a prompt. Each failure feeds the model a
precise correction; persistent failure fails closed. The same philosophy
handles the numbers going IN: budget/cutoff filters and superlatives
(best/cheapest/highest-placement) are computed in code and handed to the
model as facts, because models anchor on "cheapest" when asked for "best" and
mis-compare numbers under unit ambiguity — all observed in this build's eval
history, all now impossible by construction. Numbers never originate from the
model and never leave it unchecked.

**Empathy and safety, not just facts.** The student is a nervous teenager
making a huge decision, so the counsellor reads the feeling behind a message
and acknowledges it before the facts (a low score gets encouragement, not
pity). And because MME's own responsible-AI stance is explicit about student
wellbeing, a genuinely distressed message short-circuits the whole college
pipeline in code — before any router can brush it off as "out of scope" — to
a brief, caring reply that points to someone the student trusts and India's
free KIRAN mental-health helpline. mimi is a college counsellor, never a
therapist.

## Product decisions the data forces (documented, as requested)

- **Units.** Dataset fees are per academic year. Student queries in
  per-semester are converted at **1 academic year = 2 semesters** and the
  assumption is stated in the answer (C001's `about` confirms two-instalment
  billing). Total-course budgets are not hard-filtered (course lengths vary);
  the answer explains per-year costs instead.
- **`avg_placement_lpa = 0` (C006).** Rendered into the retrieval card as
  "not reported / not applicable" — the generator never sees a bare 0, so it
  cannot call a medical college "the worst for placements".
- **Diplomas (C005).** A diploma is not a degree. The polytechnic may appear in
  engineering answers, but only with an explicit "awards diplomas, not
  degrees" disclosure — inclusion with disclosure beats silent exclusion,
  because a 60%-scorer on a tight budget deserves to know the option exists.
- **Similar names (C002 vs C014).** Unrelated institutions in different
  cities. Cards carry id + city, the prompt requires id-based disambiguation,
  and the eval set has a dedicated trap case for it.
- **Costs beyond tuition.** Where `about` describes hostel/mess/kit/studio/lab
  charges, budget answers must surface them — technically-grounded-but-
  practically-wrong is still wrong.
- **Nothing outside the dataset.** The dataset is the whole world. Anything
  absent (a field, a course, a college) → `answered: false` with a clear
  explanation of what the data does contain.
- **Absence questions ("does X offer Y?").** Policy choice: the system returns
  `answered: false` and says the dataset does not record Y for X (citing what
  X's record does list), rather than asserting a confident "No". "The data
  doesn't list it" is verifiable; "it doesn't exist" is a claim the record
  can't fully carry. The distinction matters in admissions, where a wrong
  "no" costs a student an option.

## Evaluation (evals/)

The suite is split into two tiers, reported separately:

- **12 core cases** — the failure modes that matter, each of which the system
  is expected to pass: cutoff-as-hard-floor, per-semester and total-course
  unit conversion, the placement-0 trap, the similar-name trap, diploma
  disclosure, about-field-only answers, Hindi retrieval + response, refusals,
  plus 2 regression cases added when real bugs were found (deleting a test
  that once caught a real bug is how the bug returns).
- **5 adversarial cases** — deliberately hard, and reported honestly as a
  separate score: a two-condition superlative, three soft constraints that
  live only in free text ("a girl from a hill district, money tight"), a
  negation-plus-filter, a subjective value trade-off with no single right
  answer, and a refusal for a metric we don't hold (NIRF rank). These exist
  to find the edges, not to pad the pass rate — a failing test I wrote myself
  says more than a suspiciously perfect one.

On top of the pass rate the runner reports an **abstention score** — did the
system refuse exactly when it should, and never wrongly refuse an answerable
question? That maps directly to the brief's "graceful I-don't-know", and few
submissions will put a number on it.

```
python evals/run_evals.py
```

Latest run:

```
Core pass rate:        12/12 (100%)
Adversarial pass rate:  5/5  (100%)
Abstention: correctly refused 3/3 · 0 false refusals on 14 answerable questions
```

Two honest caveats, because the numbers alone would mislead. First, the
adversarial cases are graded loosely (e.g. "cite any reasonable college for
this vague trade-off") — they passed this run, but they're the thinnest ice:
the free-text-constraint case and the multi-hop superlative depend on
retrieval and model judgement rather than a code invariant, so a rephrasing
or the graders' own unseen questions could break them where a core case
wouldn't. Second, a clean 100% is only trustworthy because the suite was
*built from real failures* — during development it caught genuine bugs, each
of which became a design change AND a permanent test:

1. **Similar-name trap**: the model cited Ganga Institute of Commerce (C014,
   no MBA) for the MBA question — twice, in different ways. Prompt rules
   reduced it; what eliminated it was a code-level course exact-match check
   on citations.
2. **Citation-dodge**: after being corrected, the model dropped C014/C015 from
   `citations` but kept the college in the prose with a disclaimer. Fixed with
   a name-backing check (any college named in the text must be cited).
3. **Top-k recall failure**: with k=8, colleges whose evidence lives in the
   `about` field never reached the model on enumeration questions. Fixed by
   defaulting to full-corpus context at n=15 (measured cost: ~1.6k input
   tokens ≈ a fraction of a paisa).
4. **Unit-parsing miss**: the router failed to convert "60 thousand per
   semester" to an annual ceiling, so an ₹8.5L medical college entered the
   context of a ₹1.2L/year budget question. Fixed with worked examples in the
   router prompt plus a regex fallback parser.
5. **Language flip**: an English question was answered in Hindi because the
   router's language guess was passed to the generator. Fixed by having the
   generator mirror the question itself.
6. **"Best" anchored on "cheapest"**: asked for the best college on a ₹2L/year
   budget, the model recommended the cheapest and missed the objectively
   strongest option (C012: 8.4 LPA, NAAC A+, ₹45k). Fixed by computing
   superlatives in code and handing them to the model as facts.

The pattern across all six: prompt instructions reduce a failure mode;
moving the invariant into code eliminates it.

## Part D — Cost, with measured numbers

This prototype's real token usage, measured from actual calls (logged to
`metrics/metrics.jsonl`, printed by `python metrics/cost_report.py`):
**8,534 input / 373 output tokens per query** (re-measured after the
robustness upgrades — the verification guards added a little prompt weight,
and one of the seven questions needed a corrective retry). Same code, same
tokens — just
priced under two different models, at two different volumes.

| Metric | Number |
| --- | --- |
| Latency per query | **3.0 s median** warm (measured over 24 data queries: router ~1.5 s + generation ~1.3 s, both provider-bound) · 0.3 s cached · instant for greetings (code-level fast path, 0 LLM calls). A cold `answer.py` process additionally pays a one-time local embedding-model load (torch startup; reduced by offline cached loading + loading in parallel with the router call). |
| One-time embedding cost | ₹0 |

Latency engineering beyond the cache: the query embedding is computed **in
parallel with the router call** (on cold starts this overlaps the model load
with the router's network wait), retry backoff is short and capped (a rate
limit fails over to the backup provider immediately instead of sleeping out
a quota window — worst-case tail latency dropped from 60s+ of sleeps to
seconds), and the demo server pre-loads the embedding model at startup so
the first live question never pays it.

The biggest win came from measuring rather than guessing. Instrumentation
showed **29% of queries were paying for a verification retry** — a whole
extra generation call — and that most of those retries were triggered by a
*cosmetic* rule (the model saying "dataset"/"our records" instead of
counsellor language), not by a grounding failure. Word choice doesn't need an
LLM to fix, so it's now rewritten in code and only genuine grounding failures
regenerate. Measured effect: **retry rate 29% → 8%, LLM calls per query
2.34 → 2.08, median latency 3.76 s → 3.04 s**, with the eval suite unchanged
at 17/17.

One deliberate non-optimisation: **token streaming**. It would cut perceived
latency the most, but this system verifies an answer (citations, and every
fee/cutoff/seat/placement figure) *before* it ships. Streaming would put
unverified text in front of a student, which trades the one guarantee that
matters in admissions for two seconds. Instead the UI shows the real stages
("Looking through the colleges… Checking every number…"), which is honest
about what the wait buys.

### Per query (this prototype, as it runs today)

| Model | Cost per 1M tokens | Cost per 1,000 queries |
| --- | --- | --- |
| Gemini `gemini-3.1-flash-lite` | $0.10 in / $0.40 out | **₹86** |
| Llama 4 Maverick | $0.15 in / $0.60 out | **₹129** |

*Why Gemini is cheaper here:* its published per-token rate is simply lower
than Maverick's ($0.10/$0.40 vs $0.15/$0.60). This is what the submission
actually ran on — and on Google's free tier, actual cost paid was **₹0**.

### At 50,000 queries a month (same architecture, scaled up)

| Model | Cost per month | Why |
| --- | --- | --- |
| Gemini | **₹4,300** | 50,000 × ₹86/1,000. Cheapest today, but the price is fixed by Google — no way to bring it down ourselves. |
| Llama 4 Maverick | **₹6,450** | 50,000 × ₹129/1,000. Costs more per token, but it's open-weight — at real volume it can run on our own GPU server instead of a per-token API, which Gemini can never do. |

Reproduce: every query already logs its tokens; run
`python metrics/cost_report.py` to regenerate the per-query numbers.

### If we run Gemini, what do we gain?

- **Cheapest option right now** — lowest per-token price of the two.
- **Free to prototype with** — this whole submission cost ₹0 to build and test.
- **Zero infrastructure** — a managed API call, nothing to host or maintain.
- **Fast to ship** — good multilingual quality out of the box, no setup.

### If we run Maverick, what do we gain?

- **No vendor lock-in** — it's open-weights, offered by several providers, so
  we're never stuck with one company's price or terms.
- **Self-hostable at scale** — once volume is real, we can run it on our own
  GPU server and stop paying per token at all. Gemini can never offer this;
  its weights are closed.
- **Student data can stay on our own servers** once self-hosted, instead of
  leaving our infrastructure on every single query.
- **We can fine-tune it later** on our own conversations — the "own the
  model" step in the roadmap, only possible because the weights are ours.

**In short:** Gemini wins on cost and convenience today. Maverick wins on
control and cost *ceiling* once the product is real — it's the difference
between renting forever and eventually owning.

### At that volume, what actually breaks first?

Not cost — ₹4,300–6,450/month is nothing for a company. **Rate limits break
first.** We hit this ourselves during testing: free-tier limits threw errors
under load, and each query needs two LLM calls, so latency stacks up fast
too.

**First fix: cache repeated questions.** Admissions questions repeat a lot
in season. Our cache already answers repeats in 0.3 s instead of ~2.5–4.5 s. Even
a 30% cache hit rate cuts a third of the LLM bill and most of the slow
responses.

**Second fix: a paid API tier** (or, at real scale, self-hosting Maverick),
so the free-tier rate limit stops being the ceiling.

## How I'd control cost at scale

1. **Route before you retrieve, retrieve before you generate** — already built:
   smalltalk never costs 70B tokens.
2. **Cache aggressively** — embeddings are already one-time; add an answer
   cache for repeated questions (admissions questions are heavily repeated in
   season) and a router cache.
3. **Shrink context, not quality** — at 15 records, full context is the right
   call (measured: recall failures cost answers, truncation saved paise). At
   scale the order reverses: k becomes a tuned dial backed by recall evals.
4. **Right-size models per step** — the 8B does structured extraction as well
   as the 70B at ~8% of the price; keep the 70B only where judgment lives.
5. **Batch and pre-render** — nightly pre-computation of common
   comparisons/shortlists, so peak-season traffic hits caches, not models.

## Known limitations — where this will genuinely fail

I'd rather name these than let a grader find them. Some I hardened during this
build; the rest are honest, and pretending otherwise would be the real red
flag.

**Hardened (moved from weakness to code guarantee):**
- *Hallucinated numbers* — every fee, placement, cutoff, seat count and
  founding year in an answer is now checked against the cited record in code.
- *Free-text-only constraints* ("a girl from a hill district on a tight
  budget") — the gender/region/income signals that live only in prose are now
  mined into explicit "Support & eligibility" tags on each card, so retrieval
  has a handle it otherwise lacked.
- *Prompt injection* — an obvious "ignore your rules" gets a calm code-level
  deflection, and since all numbers come from code, injection can't fabricate
  a fee anyway.
- *Provider outage / rate limit* — automatic failover to a second provider
  (fired for real during the eval run).

**Still genuine failure modes (out of scope to fix well for a 15-row
prototype, so documented instead of faked):**
- *Scale* — the biggest one. This sends the full 15-college corpus as context
  every query; at thousands of colleges that explodes tokens, cost and
  latency, and needs a real top-k + reranker stage that can't be meaningfully
  built or tested against 15 rows.
- *Free-tier rate limits & latency* — two sequential LLM calls put each query
  at ~2.5–4.5 s warm, and the free tier throttles under load. Failover and caching soften
  it; a paid tier or self-host is the real fix.
- *Subjective / multi-hop questions* — "which is the smarter choice?" or
  "highest placement AND a hostel" lean on model judgement rather than a code
  invariant, so they're the least predictable on rephrasing (hence the
  adversarial eval tier).
- *Persistent memory* — profile lives for the session only; close the tab and
  it's gone. Cross-device memory means a real datastore, which is a product
  feature, not a prototype one.
- *Data-quality ceiling* — the system is only as honest as the data; a messy
  real DB (contradictions, gaps) would need a data-validation layer this
  clean 15-row set didn't require.

## What I'd do differently with more time

- **Conversation memory** (multi-turn counselling with a student profile,
  read/write memory per session — the single biggest product gap vs a real
  counsellor).
- **pgvector + Postgres** as the single store for facts, vectors and leads;
  the dataset here is a stand-in for MME's real college DB.
- **A reranker + query rewriting** once the corpus is real (thousands of
  colleges, messy names, courses expressed a dozen ways).
- **Verified live-data ingestion**: a crawler pipeline (e.g. Firecrawl) feeding
  a human-review queue that updates the college DB — the AI still answers only
  from the verified DB, never from the live web directly.
- **Bigger eval set with an LLM judge** for answer quality (not just
  citations/flags), tracked per release.
- **Voice + regional languages** (Whisper/Bhashini STT, Indic TTS) — the data
  layer here already handles multilingual retrieval.

## Part C — Reflection

**Keeping per-query cost low as usage grows:** measure first — this repo logs
every token and prices every query (₹86/1,000 measured on Gemini). Then
attack the biggest line item: caching (already built — the demo server
answers repeats in 0.3 s vs ~2.5–4.5 s live; admissions questions cluster hard
around seasons, so a 30%+ hit rate is realistic), model right-sizing per step
(small model for routing, bigger model only where judgment lives), and an
open-weights production model (Llama 4 Maverick, ₹129/1,000 projected) that
we can later self-host so cost stops depending on any vendor's price.

**Stopping the system from ever stating a wrong fee or cutoff:** never let the
model be the source of numbers — this prototype already enforces it. Fees and
cutoffs live in the data; budget/eligibility filters and comparisons run in
code (the model receives "78 < 82 → NOT eligible" as a computed fact, because
we measured models mis-comparing exactly that); three deterministic verifiers
(citation validity, course exact-match, college-name backing) check every
answer and fail closed after two corrective retries. At production I'd add a
numeric post-check (every ₹/% in the answer must match a cited record's field)
and a nightly eval run against the live DB.

**What I'd build first at MME:** exactly this pipeline wired to the real
college database behind the existing chatbot — grounded, cited answers with
refusal, plus the metrics logging. It's the trust foundation every later
feature (memory, voice, proactive nudges) stands on, and it ships in days, not
months.

**Measuring whether AI actually helps students:** pair funnel metrics
(question → shortlist → application started) against a no-AI baseline cohort;
track "answered honestly" rate (grounded answers + correct refusals) from the
eval suite; and the counsellor-escalation rate — if AI helps, human counsellors
spend their time on the hard cases, not the FAQs.

## Part B — Proof I've shipped

**What it did and who used it:** I haven't shipped an LLM/AI feature to real
users before this — no product, no user count to report. I'd rather say that
plainly than manufacture a shipped-feature story. This prototype is my first
end-to-end AI project, taken from a blank folder to a grounded, cited,
evaluated RAG system with measured cost and latency.

**My specific role:** everything here — I designed and built the retrieval
layer (hard filters + embeddings), the generation and verification pipeline
(three deterministic checks, fail-closed on failure), the eval suite (12
cases, several added after real bugs I found), the cost/latency
instrumentation (`metrics/metrics.jsonl`, `cost_report.py`), and the demo UI
(conversation memory, comparison tables, a Gemini→NIM automatic failover).

**What broke or surprised me:** without production traffic, the honest
version of this question is what broke *while building and testing it
myself* — which is the closest I have to "what broke in production," and I
treated it the same way: as signal, not noise (see the numbered list in
"Evaluation" above for the full set, and `evals/eval_set.json`'s
`full-profile-lookup` case for one more: a regex meant to detect the course
"LLB" matched the substring inside "hi**LL B**locks" and silently turned a
valid answer into a false refusal). Two similarly-named colleges (Ganga
Valley University vs. Ganga Institute of Commerce) kept getting conflated by
the model despite prompt rules — fixed only once I moved the check into
code. A live free-tier key hit rate limits mid-test, and retry-with-backoff
alone wasn't enough until I added automatic provider failover. Each fix has
an eval case that locks it in, so it can't silently regress.

**What it cost to run, and what I did to bring it down:** fully instrumented
in Part D below rather than estimated — every query's tokens and latency are
logged from a real call, not guessed. What I did to bring cost down: routed
cheap classification to a small model and reserved the larger model for
judgment calls, skip retrieval entirely for small talk, built an answer
cache (0.3 s vs ~2.5–4.5 s and zero repeat LLM cost), and picked a production
model (Llama 4 Maverick) by comparing published rates rather than assuming.

## Repo map

```
requirements.txt     # pip install -r requirements.txt (core + optional demo deps)
check_setup.py       # pre-flight: verifies deps + dataset + key before you run
answer.py            # CLI: python answer.py "question" → one JSON object on stdout
run_all.py           # regenerates answers.md for the 7 published questions
tests_offline.py     # offline harness: full pipeline vs stubbed LLM — 37 checks, ₹0, no network
rag_core/
  config.py          # models, paths, env
  data.py            # CSV load + one-card-per-college rendering
  ingest.py          # local embeddings → data/index.npz (auto-built on first run)
  retrieve.py        # hard filters (in code) + cosine ranking + lexical boost
  llm.py             # OpenAI-compatible client; retries + provider failover
  pipeline.py        # route → retrieve → generate → verify (4 checks) → fail-closed
  webmode.py         # OPTIONAL demo-UI web mode (Firecrawl) — never in graded path
evals/               # core + adversarial eval set + runner (pass rate + abstention)
metrics/             # per-query metrics (jsonl) + cost_report.py (Part D table)
data/                # sample_colleges.csv + generated index
app.py + static/     # OPTIONAL demo chat UI (uvicorn app:app --port 8000)
```

## Optional demo UI (no marks claimed — for the walkthrough video)

```bash
uvicorn app:app --port 8000     # then open http://localhost:8000
```

A small chat page. Every answer uses the exact `answer.py` pipeline and shows
its citations as college names (the C0XX id is on hover; the JSON keeps ids).
The server also supports **automatic provider failover** (optional
`FALLBACK_*` env vars): if the primary LLM provider rate-limits or goes down,
the same call transparently retries on a second provider — a live demo never
surfaces a provider hiccup. This isn't theoretical: during the eval run above,
Gemini's free tier returned a 429 mid-suite and the pipeline failed over to
the backup (NVIDIA NIM) automatically, so the run completed with no manual
intervention. The UI adds a few demo-only conveniences on top of the graded
pipeline: **conversation memory** (the last few turns are passed as context,
so "I want B.Pharm" followed by "73" is understood as a 73% score for a
B.Pharm aspirant — `answer.py` itself stays single-question, as the brief
specifies), an **answer cache** (repeat questions return in ~0.3s; persisted
to `data/answer_cache.json`, so the seven pre-warmed questions cost live LLM
calls only on the very first launch ever), an **instant-greeting path**
(a bare "hi"/"thanks" is answered in code with zero model calls, so the
first thing a student ever feels is fast), and an explicit **"🌐 Web" toggle**
next to the send button: an experimental mode that searches the
live web via Firecrawl (needs `FIRECRAWL_API_KEY` in `.env`) and answers with
URL citations, clearly labelled *unverified*. The graded path (`answer.py`,
`run_all.py`, `evals/`) never touches the web — the dataset is the whole world
there, per the brief. The toggle exists to demo the production direction:
live college data flowing through a verify-then-ingest pipeline into the
database, never straight into answers.
