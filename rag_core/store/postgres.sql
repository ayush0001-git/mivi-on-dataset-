-- MIVI corpus on PostgreSQL.
--
-- WHAT THIS BUYS, HONESTLY. Measured on the SQLite store: encode_query 40 ms,
-- count_matching 37 ms, full hybrid search 214 ms — against an LLM call of
-- 1-3 s. Retrieval is ~10% of what a student waits for, so moving the store to
-- Postgres does NOT make MIVI feel faster on its own.
--
-- What it does buy, and why it is still the right move for a real product:
--   * CONCURRENT WRITES. SQLite serialises writers. The website-harvest sweep
--     writes continuously while students read; on SQLite that contends with
--     every reader, and the nightly ETL swap needs the file closed entirely.
--   * HORIZONTAL SCALE. A SQLite file cannot be shared by several app servers.
--     "Thousands of users" eventually means more than one process, and that is
--     the wall SQLite hits first — not query speed.
--   * A REAL PLANNER + GIN. Multi-predicate filters (state + course + budget +
--     level) get proper statistics and index intersection instead of SQLite's
--     single-index-per-table plan.
--
-- pgvector is NOT installed on this server, so the 38,700 x 384 dense index
-- stays in the app process as a numpy matrix (59 MB, exact search, ~2 ms). At
-- this size that is genuinely the faster option anyway — an ANN round trip over
-- the network costs more than the matmul. To move vectors into the database
-- later: CREATE EXTENSION vector, add `embedding vector(384)` to colleges, and
-- an HNSW index; nothing else in this schema changes.

CREATE EXTENSION IF NOT EXISTS pg_trgm;    -- fuzzy college-name matching
CREATE EXTENSION IF NOT EXISTS unaccent;   -- diacritic-insensitive search

-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS colleges (
    college_id            TEXT PRIMARY KEY,
    name                  TEXT NOT NULL,
    short_name            TEXT,
    city                  TEXT,
    district              TEXT,
    state                 TEXT,
    country               TEXT,
    address               TEXT,
    pincode               TEXT,
    type                  TEXT,
    establish_year        INTEGER,
    naac_grade            TEXT,
    nirf_ranking          TEXT,
    other_ranking         TEXT,
    affiliated_university TEXT,
    autonomous_status     BOOLEAN DEFAULT FALSE,
    is_abroad             BOOLEAN DEFAULT FALSE,
    is_verified           BOOLEAN DEFAULT FALSE,
    is_featured           BOOLEAN DEFAULT FALSE,
    rating                TEXT,
    about                 TEXT,
    details               TEXT,
    why_choose            TEXT,
    website_url           TEXT,
    email                 TEXT,
    phone                 TEXT,
    logo_url              TEXT,
    banner_url            TEXT,
    source_level          TEXT NOT NULL DEFAULT 'list',
    source_updated_at     TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS courses (
    course_id             TEXT PRIMARY KEY,
    college_id            TEXT NOT NULL REFERENCES colleges(college_id) ON DELETE CASCADE,
    title                 TEXT,
    full_name             TEXT,
    program_level         TEXT,
    department            TEXT,
    duration_value        INTEGER,
    duration_unit         TEXT,
    course_mode           TEXT,
    -- BIGINT, not INTEGER. Postgres's strict typing caught what SQLite's
    -- dynamic typing hid: 20 course rows carry a tuition of 8,900,000,000
    -- (Rs 890 crore/year, all one college — almost certainly Rs 89,000 with
    -- trailing zeros). The source can hold values past int32, so the column
    -- has to as well; the ABSURD ones are handled as a data-quality decision
    -- in the ETL, not by silently failing to store them.
    tuition_min_inr       BIGINT,
    tuition_max_inr       BIGINT,
    tuition_min_raw       BIGINT,
    tuition_max_raw       BIGINT,
    currency              TEXT,
    hostel_min_inr        BIGINT,
    hostel_max_inr        BIGINT,
    one_time_fees         BIGINT,
    fee_period            TEXT,
    entrance_exams        JSONB,
    min_marks             TEXT,
    description           TEXT,
    fees_description      TEXT,
    key_highlights        JSONB,
    is_abroad             BOOLEAN DEFAULT FALSE,
    is_top_course         BOOLEAN DEFAULT FALSE,
    scholarship_available BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS college_agg (
    college_id       TEXT PRIMARY KEY REFERENCES colleges(college_id) ON DELETE CASCADE,
    n_courses        INTEGER NOT NULL DEFAULT 0,
    program_levels   JSONB,
    course_titles    JSONB,
    departments      JSONB,
    entrance_exams   JSONB,
    min_tuition_inr  BIGINT,
    max_tuition_inr  BIGINT,
    min_hostel_inr   BIGINT,
    has_hostel       BOOLEAN DEFAULT FALSE,
    has_scholarship  BOOLEAN DEFAULT FALSE,
    card             TEXT,
    -- Replaces SQLite FTS5. Generated + GIN-indexed, so it can never drift from
    -- the card the way a separately-populated FTS table can.
    search_doc       TSVECTOR
);

CREATE TABLE IF NOT EXISTS college_facts (
    college_id TEXT NOT NULL REFERENCES colleges(college_id) ON DELETE CASCADE,
    kind       TEXT NOT NULL,
    payload    JSONB NOT NULL,
    PRIMARY KEY (college_id, kind)
);

CREATE TABLE IF NOT EXISTS college_web_facts (
    college_id   TEXT NOT NULL REFERENCES colleges(college_id) ON DELETE CASCADE,
    kind         TEXT NOT NULL,
    summary      TEXT,
    payload      JSONB,
    source_url   TEXT NOT NULL,
    fetched_at   TIMESTAMPTZ NOT NULL,
    extractor    TEXT,
    confidence   REAL,
    PRIMARY KEY (college_id, kind)
);

CREATE TABLE IF NOT EXISTS web_crawl_state (
    college_id    TEXT PRIMARY KEY REFERENCES colleges(college_id) ON DELETE CASCADE,
    website_url   TEXT,
    status        TEXT NOT NULL,
    last_attempt  TIMESTAMPTZ,
    attempts      INTEGER DEFAULT 0,
    pages_fetched INTEGER DEFAULT 0,
    note          TEXT
);

-- Durable memory. This is the table that most justifies Postgres: the
-- in-process session store cannot be shared between app servers, so the moment
-- there is more than one process a student's profile follows whichever machine
-- their request happened to land on. Here it follows the student.
CREATE TABLE IF NOT EXISTS student_profile (
    session_id   TEXT PRIMARY KEY,
    profile      JSONB NOT NULL,
    summary      TEXT,
    n_turns      INTEGER NOT NULL DEFAULT 0,
    created_at   TIMESTAMPTZ NOT NULL,
    updated_at   TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS conversation_turn (
    session_id   TEXT NOT NULL,
    turn_no      INTEGER NOT NULL,
    role         TEXT NOT NULL,
    content      TEXT NOT NULL,
    citations    JSONB,
    created_at   TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (session_id, turn_no, role)
);

CREATE INDEX IF NOT EXISTS idx_turn_session   ON conversation_turn(session_id, turn_no);
CREATE INDEX IF NOT EXISTS idx_profile_updated ON student_profile(updated_at);

CREATE TABLE IF NOT EXISTS build_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

-- ---------------------------------------------------------------------------
-- Indexes. Created AFTER the bulk load by migrate_postgres.py — building them
-- first would make a 211k-row COPY several times slower.
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_colleges_state  ON colleges(state);
CREATE INDEX IF NOT EXISTS idx_colleges_city   ON colleges(city);
CREATE INDEX IF NOT EXISTS idx_colleges_type   ON colleges(type);
CREATE INDEX IF NOT EXISTS idx_colleges_abroad ON colleges(is_abroad);

-- Trigram index on the name: students type "SRM Sonipat" and "srm univ
-- sonepat" for the same college, and LIKE '%…%' cannot use a btree.
CREATE INDEX IF NOT EXISTS idx_colleges_name_trgm
    ON colleges USING GIN (name gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_courses_college ON courses(college_id);
CREATE INDEX IF NOT EXISTS idx_courses_title   ON courses(title);
CREATE INDEX IF NOT EXISTS idx_courses_level   ON courses(program_level);

-- The hot path: "does this college offer <course> at/under <budget>". A
-- composite matching the EXISTS subquery means the planner never re-scans a
-- college's whole course list to answer it.
CREATE INDEX IF NOT EXISTS idx_courses_college_title_fee
    ON courses(college_id, title, tuition_min_inr);
CREATE INDEX IF NOT EXISTS idx_courses_fulltext_trgm
    ON courses USING GIN (full_name gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_agg_min_fee ON college_agg(min_tuition_inr);
CREATE INDEX IF NOT EXISTS idx_agg_search  ON college_agg USING GIN (search_doc);

CREATE INDEX IF NOT EXISTS idx_webstate_status ON web_crawl_state(status);
