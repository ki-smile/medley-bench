-- MEDLEY-BENCH SQLite Schema v3
-- Part 1: Generation & Orchestration

CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY,
    domain TEXT NOT NULL CHECK(domain IN ('medical', 'troubleshooting', 'code_review', 'architecture', 'statistical_reasoning')),
    seed_data TEXT NOT NULL,                -- JSON: original seed for case generation
    vignette TEXT,                           -- Generated vignette text
    difficulty_tier TEXT CHECK(difficulty_tier IN ('easy', 'medium', 'hard')),
    disagreement_score REAL,                -- K=25 jackknife disagreement metric

    -- Probe flags (a case can have multiple)
    is_known_answer INTEGER DEFAULT 0,
    known_answer TEXT,                       -- JSON: {correct_answer, target_wrong_claim}
    is_trap INTEGER DEFAULT 0,
    is_dose_response INTEGER DEFAULT 0,
    is_minimal_instruction INTEGER DEFAULT 0,
    is_error_detection INTEGER DEFAULT 0,   -- Also flagged is_known_answer per v3.1
    is_counterfactual INTEGER DEFAULT 0,

    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS designer_responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL REFERENCES cases(case_id),
    model_id TEXT NOT NULL,
    response TEXT NOT NULL,                 -- JSON: full model response
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(case_id, model_id)
);

CREATE TABLE IF NOT EXISTS analyst_responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL REFERENCES cases(case_id),
    model_id TEXT NOT NULL,
    response TEXT NOT NULL,                 -- JSON: full analyst response
    jackknife_left_out INTEGER DEFAULT 0,   -- 1 if this model was held out (acts as judge)
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(case_id, model_id)
);

CREATE TABLE IF NOT EXISTS claims (
    case_id TEXT NOT NULL REFERENCES cases(case_id),
    claim_id TEXT NOT NULL,
    claim_text TEXT NOT NULL,
    majority_strength INTEGER,              -- Count of analysts supporting this claim
    jsd_score REAL,                         -- Jensen-Shannon divergence across analysts
    PRIMARY KEY (case_id, claim_id)
);

CREATE TABLE IF NOT EXISTS consensus (
    case_id TEXT PRIMARY KEY REFERENCES cases(case_id),
    consensus_data TEXT NOT NULL,           -- JSON: jackknifed consensus result
    method TEXT DEFAULT 'jackknife',
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS prompt_versions (
    prompt_name TEXT NOT NULL,
    version INTEGER NOT NULL,
    content_hash TEXT NOT NULL,             -- SHA-256 of template content
    content TEXT NOT NULL,                  -- Full template text
    created_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (prompt_name, version)
);

CREATE TABLE IF NOT EXISTS collection_failures (
    case_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    role TEXT NOT NULL,              -- 'designer' or 'analyst'
    error TEXT NOT NULL,
    attempt_count INTEGER DEFAULT 1,
    last_attempt TEXT DEFAULT (datetime('now')),
    resolved INTEGER DEFAULT 0,      -- 1 if later succeeded
    PRIMARY KEY (case_id, model_id, role)
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    job_type TEXT NOT NULL,                 -- collect_designer, collect_analyst, benchmark, export
    model_id TEXT,
    domain TEXT,
    status TEXT DEFAULT 'pending',          -- pending, running, completed, failed
    progress TEXT,                          -- JSON: {completed: N, total: M}
    result TEXT,                            -- JSON: final result
    error TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    completed_at TEXT
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_cases_domain ON cases(domain);
CREATE INDEX IF NOT EXISTS idx_cases_known_answer ON cases(is_known_answer) WHERE is_known_answer = 1;
CREATE INDEX IF NOT EXISTS idx_designer_responses_case ON designer_responses(case_id);
CREATE INDEX IF NOT EXISTS idx_analyst_responses_case ON analyst_responses(case_id);
CREATE INDEX IF NOT EXISTS idx_claims_case ON claims(case_id);
