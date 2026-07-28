-- MIVI production store: the joined MakeMyEducation corpus.
--
-- SQLite is the deliberate choice over a hosted vector DB / Postgres here:
-- the corpus is 35k colleges + 211k courses (~250 MB), it is read-mostly and
-- rebuilt from source, and a single file means the API server has zero infra
-- to provision. FTS5 gives real BM25 in-process, and WAL mode serves many
-- concurrent readers. If the corpus outgrows one box, this schema maps 1:1
-- onto Postgres + pgvector without touching the retrieval code above it.

PRAGMA journal_mode = WAL;      -- concurrent readers never block on the writer
PRAGMA synchronous = NORMAL;    -- durable enough for a rebuildable derived store
PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- colleges — one row per institution, resolved from the live API.
-- The courses CSV carries only `college` (ObjectId); everything human-readable
-- in this table came from admin.makemyeducation.com/api.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS colleges (
    college_id            TEXT PRIMARY KEY,          -- Mongo ObjectId; also the public URL slug
    name                  TEXT NOT NULL,
    short_name            TEXT,
    city                  TEXT,
    district              TEXT,
    state                 TEXT,
    country               TEXT,
    address               TEXT,
    pincode               TEXT,
    type                  TEXT,                      -- Government / Private / Deemed / Public
    establish_year        INTEGER,
    naac_grade            TEXT,
    nirf_ranking          TEXT,
    other_ranking         TEXT,
    affiliated_university TEXT,
    autonomous_status     INTEGER DEFAULT 0,
    is_abroad             INTEGER DEFAULT 0,
    is_verified           INTEGER DEFAULT 0,
    is_featured           INTEGER DEFAULT 0,
    rating                TEXT,
    about                 TEXT,                      -- aboutCollege
    details               TEXT,                      -- long-form overview prose
    why_choose            TEXT,
    website_url           TEXT,
    email                 TEXT,
    phone                 TEXT,
    logo_url              TEXT,
    banner_url            TEXT,
    -- enrichment provenance: 'list' = bulk endpoint only, 'detail' = full crawl
    source_level          TEXT NOT NULL DEFAULT 'list',
    source_updated_at     TEXT
);

CREATE INDEX IF NOT EXISTS idx_colleges_state   ON colleges(state);
CREATE INDEX IF NOT EXISTS idx_colleges_city    ON colleges(city);
CREATE INDEX IF NOT EXISTS idx_colleges_type    ON colleges(type);
CREATE INDEX IF NOT EXISTS idx_colleges_abroad  ON colleges(is_abroad);

-- ---------------------------------------------------------------------------
-- courses — one row per course offering (211k). `tuition_*_inr` is populated
-- ONLY for domestic rows: abroad fees arrive in the institution's own currency
-- and comparing them to a rupee budget would be a serious factual error.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS courses (
    course_id            TEXT PRIMARY KEY,
    college_id           TEXT NOT NULL REFERENCES colleges(college_id),
    title                TEXT,                       -- "BTech", "MBA"
    full_name            TEXT,                       -- "Bachelor of Technology"
    program_level        TEXT,                       -- Undergraduate/Postgraduate/...
    department           TEXT,
    duration_value       INTEGER,
    duration_unit        TEXT,
    course_mode          TEXT,
    tuition_min_inr      INTEGER,                    -- NULL for abroad — see above
    tuition_max_inr      INTEGER,
    tuition_min_raw      INTEGER,                    -- untyped currency, abroad rows
    tuition_max_raw      INTEGER,
    currency             TEXT,                       -- resolved from detail crawl
    hostel_min_inr       INTEGER,
    hostel_max_inr       INTEGER,
    one_time_fees        INTEGER,
    fee_period           TEXT,
    entrance_exams       TEXT,                       -- JSON array
    min_marks            TEXT,
    description          TEXT,
    fees_description     TEXT,
    key_highlights       TEXT,                       -- JSON array
    is_abroad            INTEGER DEFAULT 0,
    is_top_course        INTEGER DEFAULT 0,
    scholarship_available INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_courses_college ON courses(college_id);
CREATE INDEX IF NOT EXISTS idx_courses_title   ON courses(title);
CREATE INDEX IF NOT EXISTS idx_courses_level   ON courses(program_level);
CREATE INDEX IF NOT EXISTS idx_courses_fee     ON courses(tuition_max_inr);

-- ---------------------------------------------------------------------------
-- college_facts — the variable-shape sub-collections from the detail endpoint
-- (placement, accommodation, scholarship, facility, faq, admission). Kept as
-- JSON rather than six sparse tables: coverage is uneven across 35k colleges
-- and the retrieval layer only ever renders them into prose.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS college_facts (
    college_id TEXT NOT NULL REFERENCES colleges(college_id),
    kind       TEXT NOT NULL,                        -- placement|accommodation|...
    payload    TEXT NOT NULL,                        -- JSON array
    PRIMARY KEY (college_id, kind)
);

-- ---------------------------------------------------------------------------
-- college_agg — precomputed per-college rollups. Superlative and filter
-- questions ("cheapest BTech in Dehradun", "how many colleges offer MBA")
-- must be answered by SQL over these columns, never by an LLM doing
-- arithmetic over a truncated context window.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS college_agg (
    college_id       TEXT PRIMARY KEY REFERENCES colleges(college_id),
    n_courses        INTEGER NOT NULL DEFAULT 0,
    program_levels   TEXT,                           -- JSON array
    course_titles    TEXT,                           -- JSON array, deduped
    departments      TEXT,                           -- JSON array
    entrance_exams   TEXT,                           -- JSON array
    min_tuition_inr  INTEGER,
    max_tuition_inr  INTEGER,
    min_hostel_inr   INTEGER,
    has_hostel       INTEGER DEFAULT 0,
    has_scholarship  INTEGER DEFAULT 0,
    card             TEXT                            -- rendered retrieval card
);

CREATE INDEX IF NOT EXISTS idx_agg_min_fee ON college_agg(min_tuition_inr);

-- ---------------------------------------------------------------------------
-- Lexical half of hybrid retrieval. BM25 over the same card text the dense
-- index embeds, so an exact-name or exact-course query ("SRM Sonipat",
-- "GNM nursing") never depends on embedding luck.
-- ---------------------------------------------------------------------------
CREATE VIRTUAL TABLE IF NOT EXISTS colleges_fts USING fts5(
    college_id UNINDEXED,
    name,
    location,                                        -- "city, district, state"
    course_blob,                                     -- deduped course titles + full names
    about,
    tokenize = 'unicode61 remove_diacritics 2'
);

-- ---------------------------------------------------------------------------
-- college_web_facts — facts harvested from each college's OWN website.
--
-- Deliberately a SEPARATE table from college_facts, and never merged into
-- `colleges`. MIVI's whole value is that a number it states is traceable and
-- verified; a fee an LLM read off a third-party PDF is a fundamentally weaker
-- claim than a fee in MakeMyEducation's curated catalogue. Silently blending
-- the two would make every answer as untrustworthy as its weakest source.
--
-- So each row keeps its own provenance and MIVI attributes it out loud
-- ("according to the college's website, checked on <date>"). Promotion into
-- the curated tables is a separate, reviewed decision — never automatic.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS college_web_facts (
    college_id   TEXT NOT NULL REFERENCES colleges(college_id),
    kind         TEXT NOT NULL,          -- scholarship|fees|placement|hostel|admission|courses
    summary      TEXT,                   -- short prose, safe to render on a card
    payload      TEXT,                   -- JSON: the structured extraction
    source_url   TEXT NOT NULL,          -- the exact page this came from
    fetched_at   TEXT NOT NULL,
    extractor    TEXT,                   -- model that did the extraction
    confidence   REAL,                   -- 0-1, from the extractor
    PRIMARY KEY (college_id, kind)
);

CREATE INDEX IF NOT EXISTS idx_webfacts_kind ON college_web_facts(kind);

-- Per-college crawl bookkeeping, so a 38k-college sweep is resumable and can
-- back off from sites that are down, blocked, or have no useful pages.
CREATE TABLE IF NOT EXISTS web_crawl_state (
    college_id    TEXT PRIMARY KEY REFERENCES colleges(college_id),
    website_url   TEXT,
    status        TEXT NOT NULL,         -- pending|ok|no_site|blocked|error|empty
    last_attempt  TEXT,
    attempts      INTEGER DEFAULT 0,
    pages_fetched INTEGER DEFAULT 0,
    note          TEXT
);

CREATE INDEX IF NOT EXISTS idx_webstate_status ON web_crawl_state(status);

-- ---------------------------------------------------------------------------
-- MEMORY. The in-process session store (rag_core/service.py) is a cache with a
-- TTL; this is the durable half. Two different things live here for two
-- different reasons:
--
--   student_profile — the facts a counsellor should never make a student
--     repeat: marks, budget, stream, location, category. In-process memory
--     loses these on restart and cannot be shared by a second app server, and
--     "so what did you score again?" on turn nine is the single most obvious
--     way an assistant reveals it has no memory.
--
--   conversation_turn — the transcript, kept so long chats can be SUMMARISED
--     rather than truncated. The prompt window holds the last few turns; a
--     rolling summary carries the rest, so a fact stated in turn 2 still
--     informs turn 40.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS student_profile (
    session_id   TEXT PRIMARY KEY,
    profile      TEXT NOT NULL,          -- JSON: marks_pct, budget, field_interest…
    summary      TEXT,                   -- rolling natural-language memory
    n_turns      INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversation_turn (
    session_id   TEXT NOT NULL,
    turn_no      INTEGER NOT NULL,
    role         TEXT NOT NULL,          -- user | assistant
    content      TEXT NOT NULL,
    citations    TEXT,                   -- JSON array of college_ids
    created_at   TEXT NOT NULL,
    PRIMARY KEY (session_id, turn_no, role)
);

CREATE INDEX IF NOT EXISTS idx_turn_session ON conversation_turn(session_id, turn_no);
CREATE INDEX IF NOT EXISTS idx_profile_updated ON student_profile(updated_at);

-- ---------------------------------------------------------------------------
-- place_alias — what students call a city vs what the corpus stores.
--
-- Measured on the real corpus, and the reason this table exists:
--     "Vizag"      ->   0 colleges        "Visakhapatnam"      -> 348
--     "Trivandrum" ->   0                 "Thiruvananthapuram" -> 168
--     "Gurgaon"    ->   0                 "Gurugram"           -> 245
-- A student typing the name everyone actually uses gets NOTHING, and the
-- answer is a well-formed, grounded, completely useless "no colleges found".
--
-- This could be repaired with an LLM critic (rag_core/selfrag.py does exactly
-- that), but a lookup is ~1000x cheaper: resolving in SQL costs microseconds
-- against ~1.5 s for a model round trip, and it is deterministic — the same
-- misspelling resolves the same way every time, which a model does not
-- guarantee. The critic stays as the fallback for aliases nobody curated.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS place_alias (
    alias      TEXT PRIMARY KEY,   -- lowercase, what the student types
    canonical  TEXT NOT NULL,      -- exactly as `colleges.city` / `.state` stores it
    kind       TEXT NOT NULL,      -- city | state
    source     TEXT                -- curated | derived
);

CREATE INDEX IF NOT EXISTS idx_alias_canonical ON place_alias(canonical);

-- Build metadata: lets the server refuse to serve an index built from a
-- different corpus than the DB it is querying.
CREATE TABLE IF NOT EXISTS build_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
