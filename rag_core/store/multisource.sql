-- MIVI multi-source layer: competing claims, and one auditable winner.
--
-- WHY A SECOND FACT TABLE AT ALL. `college_web_facts` is keyed
-- (college_id, kind) and its upsert overwrites `summary`, `payload`,
-- `source_url` and `confidence` in place. That is correct while there is
-- exactly one source per topic — the college's own website — and it becomes
-- lossy the moment there are two. The AICTE approved-intake sheet and a
-- college's fee page will both report `fees`; under a (college_id, kind) key
-- whichever wrote last simply erases the other, and MIVI can no longer answer
-- the only question that matters when a number looks wrong: "who told you?"
--
-- So the claim table below is keyed (college_id, kind, source_key) and holds
-- COMPETING claims. Nothing in it is a winner. `college_fact_resolved` holds
-- the winner, plus whether the sources agreed and the full text of every claim
-- that lost. A disagreement between two sources about a fee is a fact about
-- our data, and it is recorded rather than resolved away — house rule 6:
-- correctness beats coverage, and rule 4: a scraped value never silently
-- overwrites a catalogue value.
--
-- ADDITIVE ONLY. Nothing here alters colleges, courses, college_facts,
-- college_web_facts or web_crawl_state — those are written by jobs running
-- right now. The resolved table is a projection those jobs feed into via
-- rag_core/sources/reconcile.py, never a replacement for what they write.

-- ---------------------------------------------------------------------------
-- Claims. Append-and-refresh, never merged.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS college_source_fact (
    college_id  TEXT NOT NULL REFERENCES colleges(college_id) ON DELETE CASCADE,
    kind        TEXT NOT NULL,

    -- "<class>:<origin>" — e.g. official:iitr.ac.in, gov:nirf, gov:aishe,
    -- mme:catalogue, other:pdf-prospectus. The class prefix is what
    -- reconcile.py derives a default tier from, and the origin is what makes
    -- two claims INDEPENDENT: two pages on iitr.ac.in are one voice, so they
    -- must collapse to one source_key rather than corroborate each other into
    -- false confidence.
    source_key  TEXT NOT NULL,

    -- 1 official college site, 2 government registry, 3 MME catalogue,
    -- 4 other. Stored rather than re-derived from the prefix on every read:
    -- the tier of a source is a judgement that can be revised (a state
    -- technical board promoted from 4 to 2, say) and the claim should keep the
    -- tier it was recorded under until something deliberately re-tiers it.
    source_tier INT  NOT NULL CHECK (source_tier BETWEEN 1 AND 4),

    -- VERBATIM, both of them. House rule 3: money and marks are copied with
    -- the unit the page stated. Reconciliation normalises numbers to COMPARE
    -- claims and never writes the normalised form back.
    summary     TEXT,
    payload     JSONB,

    source_url  TEXT,
    fetched_at  TIMESTAMPTZ NOT NULL,
    confidence  REAL,

    PRIMARY KEY (college_id, kind, source_key)
);

-- Reconciliation reads every claim for one (college_id, kind) group; the PK
-- prefix already serves that. These two serve the sweep-side questions:
-- "which colleges did source X touch" (re-tiering, retracting a bad source)
-- and "what is old enough to re-fetch".
CREATE INDEX IF NOT EXISTS idx_source_fact_source ON college_source_fact(source_key, kind);
CREATE INDEX IF NOT EXISTS idx_source_fact_fetched ON college_source_fact(fetched_at);

-- ---------------------------------------------------------------------------
-- Winners. One row per (college_id, kind), rebuilt from the claims above.
-- Derived data: safe to TRUNCATE and recompute, never written by hand.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS college_fact_resolved (
    college_id   TEXT NOT NULL REFERENCES colleges(college_id) ON DELETE CASCADE,
    kind         TEXT NOT NULL,

    -- The winning claim, copied verbatim out of college_source_fact.
    summary      TEXT,
    payload      JSONB,
    source_key   TEXT NOT NULL,
    source_tier  INT  NOT NULL,
    source_url   TEXT,
    fetched_at   TIMESTAMPTZ,

    -- NOT the winner's own confidence: the winner's confidence after
    -- corroboration raised it and conflict lowered it. This is the number the
    -- card renderer thresholds on, so the arithmetic behind it lives in one
    -- place (reconcile.py CONFIDENCE section) and is explained there.
    confidence   REAL NOT NULL,

    -- 'single'       one source had anything to say
    -- 'corroborated' >=2 independent sources agreed, after normalisation
    -- 'conflict'     >=2 sources disagreed; `rejected` holds the losers
    agreement    TEXT NOT NULL CHECK (agreement IN ('single', 'corroborated', 'conflict')),
    n_sources    INT  NOT NULL,
    n_agreeing   INT  NOT NULL,

    -- Every claim that did not win, with its source, its value AS PUBLISHED,
    -- and why it lost. This column is the entire point of the table: without
    -- it a conflict is invisible, and MIVI would state a fee with no way to
    -- say "the college's own site says something different". Counsellors get
    -- asked exactly that.
    rejected     JSONB NOT NULL DEFAULT '[]'::jsonb,

    -- Which rule picked the winner (tier / freshness / confidence /
    -- stale-tier-demotion). Makes a surprising answer diagnosable from SQL
    -- alone, without re-running the resolver.
    rule         TEXT,

    -- Digest of the inputs THE DECISION DEPENDED ON, staleness verdict
    -- included. Lets reconcile_all() skip groups whose answer cannot have
    -- changed, and makes a re-run a genuine no-op rather than a rewrite with a
    -- new timestamp (house rule 5: resumable and idempotent).
    fingerprint  TEXT NOT NULL,
    resolved_at  TIMESTAMPTZ NOT NULL,

    PRIMARY KEY (college_id, kind)
);

-- Partial index: conflicts are the rows a human reviews, and they are meant to
-- stay a small minority of the table. Indexing only them keeps the audit query
-- ("show me every college where two sources disagree about fees") an index scan
-- even once the resolved table has a row per college per kind.
CREATE INDEX IF NOT EXISTS idx_resolved_conflict
    ON college_fact_resolved(kind, confidence)
    WHERE agreement = 'conflict';

CREATE INDEX IF NOT EXISTS idx_resolved_kind ON college_fact_resolved(kind);
