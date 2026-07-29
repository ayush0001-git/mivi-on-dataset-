# Where things stand — resume here

Everything below is resumable and idempotent. Postgres is the only store.
Run `python status.py` first — it reports the LIVE state, which is what matters.

## Reading a running job's progress

**Progress lines go to `data/raw/*_run.err`, not `*_run.log`.** The `.log` files
are stdout and are 0 bytes, which reads as "dead job" at a glance. Tail the
`.err`.

**The detail crawler's progress is not in the database.** It appends to
`data/raw/colleges_detail.jsonl` and only reaches Postgres when the ETL runs, so
its DB counter sits still for hours while it works normally. Two separate
sessions called a healthy crawl dead by reading that counter. `status.py` now
reports the JSONL's row count and mtime instead. Low CPU is also expected: the
crawler is AIMD-throttled to ~0.5-1 req/s against MakeMyEducation's 429s and is
idle on network waits by design.

## Jobs that were running when this was written (28 Jul 2026)

All three were started detached and survive a closed terminal. Check them with
`python status.py`; restart any that died.

| job | command | note |
|---|---|---|
| dense index | `python -m rag_core.index_build` | ~24,600 cards to embed, ~6-14/s. Cards were re-rendered after the fee fix, so the old vectors are stale. |
| detail crawl | `python -m rag_core.sources.crawl_colleges detail --concurrency 4` | ~5,000 of 35,358 left. Rate-limited to ~0.5 req/s by MakeMyEducation's own API — this is the upstream constraint on everything. |
| harvest + OCR | `python -m rag_core.sources.web_enrich --no-llm --concurrency 16 --firecrawl-reserve 50` | 13,118 colleges requeued (the `blocked` + `empty` ones) for a Firecrawl + OCR retry. ~0.5/s. |

## State

| | |
|---|---|
| Corpus | 38,700 colleges · 211,071 courses |
| Cards | 38,700 rendered, fee phrasing corrected |
| Web-harvested facts | ~3,650 across ~2,850 colleges |
| Verified CONFIRMED | 1,670 · quarantined 474 |
| NIRF rankings | 236 colleges |
| Maharashtra FRA approved fees | 100 colleges |
| Admission criteria | 148,631 facts · 24,182 colleges (62%) |
| NAAC grades | 2,324 colleges (A++ 117 … C 115) |
| Verified fee in the DB | 100 colleges, all FRA-approved 2024-25 |
| Firecrawl | 3 keys pooled, ~2,200 spendable credits |
| OCR | Tesseract 5.4 installed and wired |
| Embedding device | GPU when usable (`pick_device`), else CPU |

## Every fee now carries a date, or says it has none

The client's question was "how do I know when this fee was updated". The answer
differs by source, and the card says which:

| source | date | rendered as |
|---|---|---|
| FRA approved | order date + academic year | "Rs 67,272 per year, for 2024-25, legally approved" |
| College website | the year the PAGE states | "the page states these are for 2019-20 — OUT OF DATE" |
| Catalogue (CSV) | **none exists** | "undated" |

`stated_year()` in `sources/verify_facts.py` reads the academic year off the page
text. This is NOT the fetch date, and conflating them is how a stale fee looks
fresh — the card used to show only "checked 2026-07-28" for a page headed "Fee
Structure 2019-20". Harvested years already range 2022-23 to 2026-27.

**The catalogue fee genuinely has no date.** Measured: `courses` has no date
column at all; the detail payload's course `createdAt` is 2026-03/04 for every
row (the client's bulk import) and `updatedAt` is empty on all of them. So
`createdAt` is an import timestamp, and showing it as "fee updated on" would make
every fee look three months old when the underlying figure could be any year —
hiding the staleness instead of exposing it. Do not do this.

`sources/fee_promote.py` writes a dated fee into `colleges.verified_fee_*`. It
does NOT touch `courses.tuition_min_inr`: a harvested fee can be wrong, and a
wrong fee is the most damaging error here. Find stale ones with
`WHERE verified_fee_year < '2024'`.

Most harvested `fees` facts are NOT promotable, and this is not a bug. The
payload is a bag of labelled amounts, not a fee — two real examples:
`{"Tuition Fee for NRI students": "US$ 7000.00", "Hostel ... Charges": "Rs. 4000.00"}`
and `{"fees / fees after deduction": "Rs.500/-"}`. The second is a refund clause.
A line item is promoted only when its label says tuition, its currency is rupees,
its magnitude is plausible, and the fact carries a stated year.

## Where the missing data actually came from

Worth knowing before spending another day on the web: the client's OWN detail
crawl held far more than the open web did. `sources/admission_ingest.py`
recovered 148,631 admission criteria covering 24,182 colleges from JSONL already
on disk. The web sweep, after days of crawling, has 3,714 facts across 2,871
colleges. Measured on the hardest slice, the sweep returns ~60 facts per 1,750
colleges: 82% of those sites publish nothing extractable and 15% block.

So: exhaust what is already downloaded before crawling for it.

## Closed questions — do not re-investigate

- **Annual fees cannot be derived.** Course duration is present on 2 of 40,669
  detail course objects and 10 of 211,071 catalogue rows. `feesDescription`
  mentions a per-year figure in 0 of 21,266 values. The whole-programme total is
  the only honest output.
- **Maharashtra FRA is matched out at 100 colleges.** Of its 524 rejections, 503
  had no candidate in the corpus at all and the 6 near-misses have no close
  corpus name. That is a corpus limit, not a matcher bug.
- **NIRF still has ~33 recoverable institutions** in the near-miss band,
  including IIT Delhi, Shiv Nadar (0.913) and IIFT (0.917). These score ABOVE
  the 0.78 floor, so a gate other than the score is rejecting them. Not yet
  diagnosed — this one IS worth finishing.
- **admissionStatus and admissionMode are schema defaults**, not observations:
  "Open" on 265,566 of 265,567 rows and "Merit Based" on 85,814 of 85,815. Never
  render them.

## The finding that matters most

`courses.tuition_*_inr` is the **whole-programme** fee, not annual. Verified
against Maharashtra's Fee Regulating Authority (legally approved ANNUAL fees)
over 564 same-stream pairs — the ratio equals the course length:

    MBA   (2 years)  2.00x
    MCA   (2 years)  2.39x
    BTech (4 years)  3.55x

MIVI had been saying "per year", inflating every fee 2-4x. Sanghavi College was
quoted at Rs 2,39,000/year against an approved Rs 67,272/year — a student who
could afford it was told they could not.

Root cause was in the ETL: `fee_period` defaulted to `"Year"` when the source
field is empty on 100% of rows. Now `"unspecified"`. Course duration exists on
10 of 211,071 rows, so the annual figure **cannot** be derived — the honest
output is the total. Rule 3 of the generator prompt forbids per-year phrasing.
The one exception is a "Fee Regulating Authority approved" line, which IS annual
and carries its year.

## OCR — how it is kept safe

70% of college fee PDFs are scans. `rag_core/sources/ocr.py` renders each page
at **two different DPI (300 and 400)** and keeps only the figures both readings
agree on; disputed numbers are stripped from the text before any extractor sees
them, so no fact can cite them. Preprocessing: grayscale, autocontrast, median
denoise, moment-based deskew, binarise.

`confidence_ceiling()` caps what an OCR page may claim by how well the two
passes agreed — 0.75 for a clean scan, 0.45 for a noisy one. Anything under 0.5
is quarantined and never shown to a student. Measured: 22 OCR runs produced 187
corroborated figures with 20/22 yielding something.

OCR runs only for `fees / scholarship / admission / cutoff / courses` PDFs, and
only when the HTML pass found fewer than 2 facts — it is the most expensive step
in the sweep.

## Next, in priority order

1. Let the three jobs above finish.
2. Re-run `python -m rag_core.sources.reverify` — the gate is wired into new
   harvests, but older facts still need the retro pass.
3. `rag_core/sources/nightly.py` has a `--dry-run` and a Task Scheduler command,
   never yet run for real.
4. Re-render cards + re-index after any large harvest, so new facts reach
   students and semantic search can find them.

## Hard-won rules — do not undo these

- **Postgres only.** SQLite is gone; `rag_core/store/backend.py` owns every
  connection.
- **Never say "per year"** next to a catalogue fee. See above.
- **Nothing scraped overwrites the curated catalogue.** Harvested facts live in
  `college_web_facts` with provenance and render under an explicit
  "FROM THE COLLEGE'S OWN WEBSITE — not verified" heading. Registry data
  (NIRF/FRA) is a separate, higher tier.
- **Every harvested fact is gated** by `verify_facts.py`: every number must
  appear verbatim on the source page, and the page must corroborate the
  college's identity. Do not weaken this to raise coverage.
- **Dry-run anything that writes matched data.** The NIRF matcher was wrong
  three times (stopwords, then string similarity, then acronyms); `--dry-run`
  caught all three before they reached the database.
- **Politeness is not optional** on third-party hosts: robots.txt, per-domain
  pacing, bounded requests per host.
- **No commercial aggregators** (Shiksha, Careers360, CollegeDunia). Raised with
  the client and declined: their terms forbid it, and three aggregators agreeing
  is usually one source copied three times.

## Measured ceilings — so nobody re-litigates them

- Website path tops out near **14%** of colleges: 62% of sites publish nothing
  parseable, and 22,452 colleges have no website URL at all.
- AICTE is **not usable** — its public data carries no institution names, so it
  cannot be joined to the catalogue.
- Firecrawl is a **rescue for blocked sites**, never a bulk engine (~1,000
  credits/month per key).

## Known unfixed defects

From the adversarial review, still open (none produce a wrong number):
superlative citation edge cases, CORS headers absent on 500 responses, one
`superlatives()` query that re-runs its prefilter three times.

Nothing is committed to git yet — ~17,000 lines live only on disk.

## Field-level coverage — the next phase (29 Jul 2026)

Two tables now exist and are the foundation for everything that follows:

`college_priority` — all 38,700 colleges scored and split into three tiers by
likely student demand (NIRF +40, NAAC +25, course count +30, website +10,
popular course +15). Tier 1 = 863 colleges, tier 2 = 12,167, tier 3 = 25,670.
Crawl in that order: only 166 tier-1 colleges lack a fee, against 18,508 in
tier 3, so the effort/return ratio differs by two orders of magnitude.
**This is a PROXY.** The client's website analytics hold real search counts;
when that export arrives, rebuild the tiers from it and delete this note.

`college_field` — one row per (college, field, source) with source_url,
stated_year, fetched_at and confidence. 8,319 rows backfilled from NAAC, NIRF,
FRA and the web harvest. This is what makes targeted refresh possible:

    SELECT college_id, field FROM college_field
    WHERE fetched_at < now() - interval '180 days'

Nightly should read THAT rather than re-crawling the corpus.

### What is actually missing, measured against the client's 11-category spec

EIGHT categories have no columns at all — geo lat/long, campus size, approvals
(AICTE/UGC), seats/intake, placement %, packages, recruiters, facilities,
gallery, brochure URL, social links, cutoffs, scholarships. Extract nothing for
these until the schema exists; there is nowhere to put it.

Columns that exist and are empty: pincode 0%, phone/email 0.1%,
affiliated_university 0%, logo/banner 6%, website_url 57.5%, establish_year 83%.

FREE WIN, do this first: `naac_grade` and `nirf_ranking` read 0% on the
`colleges` table while 2,504 NAAC grades and 264 NIRF ranks already sit in
`registry_fact`. That is a promotion, not a crawl.

### Honest ceiling

8,319 tracked fields against roughly 7.7 lakh possible (38,700 x ~20). Placement
packages, cutoffs and recruiters are simply not published by most of the 25,670
tier-3 colleges — small local institutions with no website. Crawl tier 1 hard,
expect tier 3 to stay largely empty, and leave it empty rather than filling it
with numbers nobody published.
