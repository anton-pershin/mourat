-- Business domain hierarchy
CREATE TABLE IF NOT EXISTS business_domains (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS products (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    domain_id TEXT NOT NULL,
    FOREIGN KEY (domain_id) REFERENCES business_domains(id)
);

CREATE TABLE IF NOT EXISTS high_level_technologies (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    product_id TEXT NOT NULL,
    FOREIGN KEY (product_id) REFERENCES products(id)
);

CREATE TABLE IF NOT EXISTS technical_challenges (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS constraints (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS technology_challenges (
    technology_id TEXT NOT NULL,
    challenge_id TEXT NOT NULL,
    PRIMARY KEY (technology_id, challenge_id),
    FOREIGN KEY (technology_id) REFERENCES high_level_technologies(id),
    FOREIGN KEY (challenge_id) REFERENCES technical_challenges(id)
);

CREATE TABLE IF NOT EXISTS technology_constraints (
    technology_id TEXT NOT NULL,
    constraint_id TEXT NOT NULL,
    PRIMARY KEY (technology_id, constraint_id),
    FOREIGN KEY (technology_id) REFERENCES high_level_technologies(id),
    FOREIGN KEY (constraint_id) REFERENCES constraints(id)
);

-- Research domain hierarchy
CREATE TABLE IF NOT EXISTS research_domains (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS research_directions (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    domain_id TEXT NOT NULL,
    FOREIGN KEY (domain_id) REFERENCES research_domains(id)
);

CREATE TABLE IF NOT EXISTS research_objects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    direction_id TEXT NOT NULL,
    FOREIGN KEY (direction_id) REFERENCES research_directions(id)
);

CREATE TABLE IF NOT EXISTS research_questions (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    object_id TEXT NOT NULL,
    FOREIGN KEY (object_id) REFERENCES research_objects(id)
);

CREATE TABLE IF NOT EXISTS research_topics (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS topic_technical_challenges (
    topic_id TEXT NOT NULL,
    challenge_id TEXT NOT NULL,
    PRIMARY KEY (topic_id, challenge_id),
    FOREIGN KEY (topic_id) REFERENCES research_topics(id),
    FOREIGN KEY (challenge_id) REFERENCES technical_challenges(id)
);

CREATE TABLE IF NOT EXISTS topic_research_questions (
    topic_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    PRIMARY KEY (topic_id, question_id),
    FOREIGN KEY (topic_id) REFERENCES research_topics(id),
    FOREIGN KEY (question_id) REFERENCES research_questions(id)
);

-- Content item lookup tables
CREATE TABLE IF NOT EXISTS source_types (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS platforms (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS influence_metrics (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT
);

-- Content items
CREATE TABLE IF NOT EXISTS content_items (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    source_type_id TEXT NOT NULL,
    platform_id TEXT NOT NULL,
    url TEXT,
    published_at DATE,
    authors TEXT,
    influence_score INTEGER CHECK (influence_score BETWEEN 0 AND 100),
    influence_metric_id TEXT NOT NULL,
    FOREIGN KEY (source_type_id) REFERENCES source_types(id),
    FOREIGN KEY (platform_id) REFERENCES platforms(id),
    FOREIGN KEY (influence_metric_id) REFERENCES influence_metrics(id)
);

-- Content item references (self-referencing)
CREATE TABLE IF NOT EXISTS content_item_references (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    PRIMARY KEY (source_id, target_id),
    FOREIGN KEY (source_id) REFERENCES content_items(id),
    FOREIGN KEY (target_id) REFERENCES content_items(id)
);

-- Content item -> technical challenge junction
CREATE TABLE IF NOT EXISTS item_technical_challenges (
    content_id TEXT NOT NULL,
    challenge_id TEXT NOT NULL,
    justification TEXT,
    relevance_score INTEGER CHECK (relevance_score BETWEEN 0 AND 100),
    PRIMARY KEY (content_id, challenge_id),
    FOREIGN KEY (content_id) REFERENCES content_items(id),
    FOREIGN KEY (challenge_id) REFERENCES technical_challenges(id)
);

-- Content item -> research question junction
CREATE TABLE IF NOT EXISTS item_research_questions (
    content_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    justification TEXT,
    relevance_score INTEGER CHECK (relevance_score BETWEEN 0 AND 100),
    PRIMARY KEY (content_id, question_id),
    FOREIGN KEY (content_id) REFERENCES content_items(id),
    FOREIGN KEY (question_id) REFERENCES research_questions(id)
);

-- Content item -> research topic junction
CREATE TABLE IF NOT EXISTS item_research_topics (
    content_id TEXT NOT NULL,
    topic_id TEXT NOT NULL,
    justification TEXT,
    relevance_score INTEGER CHECK (relevance_score BETWEEN 0 AND 100),
    PRIMARY KEY (content_id, topic_id),
    FOREIGN KEY (content_id) REFERENCES content_items(id),
    FOREIGN KEY (topic_id) REFERENCES research_topics(id)
);

-- FTS5 virtual table for full-text search
CREATE VIRTUAL TABLE IF NOT EXISTS content_items_fts USING fts5(
    name,
    description,
    content=content_items,
    content_rowid=rowid
);
