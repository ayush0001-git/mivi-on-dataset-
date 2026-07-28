# Where things stand — resume here

Everything below is resumable and idempotent. Postgres is the only store.
Run `python status.py` first — it reports the LIVE state, which is what matters.

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
| Firecrawl | 3 keys pooled, ~2,200 spendable credits |
| OCR | Tesseract 5.4 installed and wired |

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
