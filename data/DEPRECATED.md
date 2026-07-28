# Retired prototype artifacts (`data/deprecated/`)

MIVI was first built against a hand-authored 15-college sample so the pipeline
could be designed and graded before the real data was available. It now runs on
the live MakeMyEducation production corpus, and these files are kept only for
reference — **nothing in the codebase may read them.**

| File | What it was | Why it is retired |
|---|---|---|
| `sample_colleges.csv` | 15 invented colleges in Uttarakhand, with `college_id` codes `C001`–`C015` | Replaced by the real corpus: 211,511 course offerings across ~35,300 colleges, all of India plus overseas |
| `answer_cache.prototype.json` | 10 cached answers from the demo server | **Actively dangerous.** Every entry describes colleges that do not exist. A warm cache keyed on question text alone would have served these as answers about the real corpus |
| `index.prototype.npz` | Embedding index over the 15 cards | Wrong dimensionality of corpus and wrong ids (`C0XX` vs Mongo ObjectIds) |

## Schema that no longer exists

The prototype CSV had columns that the real data simply does not contain, and
the difference is not cosmetic — carrying any of them forward would invite the
model to invent numbers:

`last_year_cutoff_pct`, `total_seats`, `avg_placement_lpa`, `naac_grade`,
`hostel_available`, `courses_offered`, `annual_fees_inr`, `about`

Measured against the live corpus: NAAC grade, NIRF ranking, placements,
scholarships, accommodation and admission cutoffs are **empty across
effectively the entire catalogue**. The correct behaviour when a student asks
for one of these is `answered: false` plus an adjacent fact that *is* known —
never a guess.

## What replaced them

| Now | Built by |
|---|---|
| `data/raw/colleges_list.jsonl` | `rag_core.sources.crawl_colleges list` — full 38,700-college catalogue from the live API |
| `data/raw/colleges_detail.jsonl` | `rag_core.sources.crawl_colleges detail` — optional, resumable per-college enrichment |
| `data/mivi.db` | `rag_core.store.build_db` — joined SQLite corpus (colleges, courses, aggregates, FTS5) |
| `data/index.npz` | `rag_core.index_build` — dense index over college cards |

Rebuild everything from source with those four commands; none of the retired
files participate.
