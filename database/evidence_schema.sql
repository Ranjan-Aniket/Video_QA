-- Evidence Items Table
-- Stores individual evidence items extracted from videos with AI predictions
CREATE TABLE IF NOT EXISTS evidence_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id VARCHAR(255) NOT NULL,
    evidence_type VARCHAR(50) NOT NULL,  -- 'ocr', 'emotion', 'object_detection', 'action', 'scene', etc.
    timestamp_start FLOAT NOT NULL,
    timestamp_end FLOAT NOT NULL,

    -- AI Predictions (JSON format)
    gpt4_prediction TEXT,  -- JSON: {"answer": "...", "confidence": 0.9, "evidence": [...]}
    claude_prediction TEXT,  -- JSON: {"answer": "...", "confidence": 0.92, "evidence": [...]}
    open_model_prediction TEXT,  -- JSON: {"yolo": {...}, "ocr": {...}, "whisper": {...}}
    ground_truth TEXT,  -- JSON: Objective facts from 10 deterministic models (YOLO, OCR, Places365, etc.)

    -- Consensus & Confidence
    ai_consensus_reached BOOLEAN DEFAULT FALSE,
    consensus_answer TEXT,  -- The agreed-upon answer if consensus reached
    confidence_score FLOAT,
    disagreement_details TEXT,  -- JSON: why AIs disagreed
    needs_human_review BOOLEAN DEFAULT FALSE,
    priority_level VARCHAR(20) DEFAULT 'medium',  -- 'high', 'medium', 'low'
    flag_reason TEXT,  -- Why flagged for review

    -- Human Review
    human_review_status VARCHAR(20) DEFAULT 'pending',  -- 'pending', 'approved', 'rejected', 'corrected', 'skipped'
    human_reviewer_id VARCHAR(100),
    human_answer TEXT,  -- JSON: corrected answer
    human_confidence VARCHAR(20),  -- 'high', 'medium', 'low'
    human_notes TEXT,
    review_timestamp TIMESTAMP,
    review_duration_seconds FLOAT,

    -- Quality Tracking
    ai_was_correct BOOLEAN,  -- Did AI get it right (compared to human)
    correction_category VARCHAR(50),  -- What type of error: 'hallucination', 'wrong_ocr', 'wrong_emotion', etc.

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (video_id) REFERENCES videos(video_id)
);

CREATE INDEX idx_evidence_video ON evidence_items(video_id);
CREATE INDEX idx_evidence_review_status ON evidence_items(human_review_status);
CREATE INDEX idx_evidence_priority ON evidence_items(priority_level);
CREATE INDEX idx_evidence_needs_review ON evidence_items(needs_human_review);

-- Reviewer Performance Table
CREATE TABLE IF NOT EXISTS reviewer_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reviewer_id VARCHAR(100) NOT NULL,
    date DATE NOT NULL,

    -- Volume Metrics
    items_reviewed INTEGER DEFAULT 0,
    items_approved INTEGER DEFAULT 0,
    items_rejected INTEGER DEFAULT 0,
    items_corrected INTEGER DEFAULT 0,
    items_skipped INTEGER DEFAULT 0,
    avg_review_time_seconds FLOAT,

    -- Quality Metrics
    agreement_with_ai FLOAT,  -- % of time reviewer agrees with AI
    agreement_with_peers FLOAT,  -- Inter-rater reliability
    flagged_items_found INTEGER DEFAULT 0,  -- Found issues AI missed

    -- Efficiency
    items_per_hour FLOAT,
    total_review_time_seconds FLOAT,

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(reviewer_id, date)
);

CREATE INDEX idx_reviewer_perf_date ON reviewer_performance(reviewer_id, date);

-- Review Sessions Table
CREATE TABLE IF NOT EXISTS review_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reviewer_id VARCHAR(100) NOT NULL,
    session_start TIMESTAMP NOT NULL,
    session_end TIMESTAMP,
    items_reviewed INTEGER DEFAULT 0,
    session_type VARCHAR(20),  -- 'priority', 'spot_check', 'batch'
    is_active BOOLEAN DEFAULT TRUE,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_review_sessions_reviewer ON review_sessions(reviewer_id);
CREATE INDEX idx_review_sessions_active ON review_sessions(is_active);

-- Add evidence tracking columns to videos table
-- Run this as an ALTER TABLE if videos table already exists:
-- ALTER TABLE videos ADD COLUMN evidence_extraction_status VARCHAR(50) DEFAULT 'pending';
-- ALTER TABLE videos ADD COLUMN ai_evidence_count INTEGER DEFAULT 0;
-- ALTER TABLE videos ADD COLUMN evidence_needs_review_count INTEGER DEFAULT 0;
-- ALTER TABLE videos ADD COLUMN evidence_approved_count INTEGER DEFAULT 0;
-- ALTER TABLE videos ADD COLUMN evidence_accuracy_estimate FLOAT;
