---
description: Resume MIVI work — reads live state and the handoff plan, then reports where to pick up
---

You are resuming work on MIVI, MakeMyEducation's AI college counsellor. Do this
before anything else, and do it in this order:

1. Run `python status.py` (use `.venv/Scripts/python.exe`). It prints the LIVE
   state: corpus counts, enrichment counts, whether the dense index is stale
   against the cards, and any background job still running.

2. Read `RESUME.md`. It holds the plan, the open decisions for the client, and
   the one finding that matters most (fees in this corpus are whole-programme,
   not annual — read that section before you touch anything fee-related).

3. Read `README.md` only if you need the architecture; `docs/` is the archived
   prototype and is NOT current.

Then tell the user, briefly:
  - what the live state says, especially anything `status.py` flags as stale or
    missing
  - the single next action you recommend, and why
Do not start a long job without saying what it is and roughly how long it takes.

## Things that will bite you if you forget them

- **Postgres is the only store.** SQLite is gone. `rag_core/store/backend.py`
  owns every connection. If Postgres is down, `status.py` says so first — start
  the service before diagnosing anything else.
- **Fees are for the FULL PROGRAMME, not per year.** Verified against
  Maharashtra's Fee Regulating Authority over 564 same-stream pairs (MBA 2yr →
  2.00x, BTech 4yr → 3.55x). Never write "per year" next to a catalogue fee.
  The one exception is a "Fee Regulating Authority approved" line, which IS
  annual and carries its year.
- **Never silently promote scraped data.** Website-harvested facts live in
  `college_web_facts` with provenance and are rendered under an explicit
  "FROM THE COLLEGE'S OWN WEBSITE — not verified" heading. Government registry
  data (NIRF, FRA) is a separate, higher tier. Nothing scraped overwrites the
  curated catalogue.
- **Every harvested fact is gated** by `rag_core/sources/verify_facts.py`: every
  number it asserts must appear verbatim on the source page, and the page must
  corroborate the college's identity. Do not weaken that to raise coverage.
- **Coverage has a measured ceiling of ~14%** on the website path (62% of
  college sites publish nothing parseable; 22,452 colleges have no website URL).
  Raising it needs a source keyed on college identity, not on a website.
- **Do not scrape commercial aggregators** (Shiksha, Careers360, CollegeDunia).
  This was raised with the client and declined: their terms forbid it, and three
  aggregators agreeing is usually one source copied three times.
- **Third-party crawling stays polite**: robots.txt honoured, per-domain pacing,
  bounded requests per host. Firecrawl credits are scarce (~1,000/month) and are
  a fallback for blocked sites only, never a bulk engine.
- **Dry-run first on anything that writes matched data.** The NIRF matcher was
  wrong three times in a row (stopwords, then string similarity, then acronyms)
  and every wrong version was caught by `--dry-run` before it reached the
  database. Keep that habit.

$ARGUMENTS
