"""
Database schema definitions using SQLAlchemy ORM - UPDATED WITH REVIEW SYSTEM
All tables for video processing, Q&A, failures, feedback, and HUMAN REVIEW
"""
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text, JSON,
    ForeignKey, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Dict, Any

Base = declarative_base()

# ==================== VIDEO TABLE (UPDATED) ====================

class Video(Base):
    """Video metadata and processing status"""
    __tablename__ = "videos"
    
    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(String(50), unique=True, nullable=False, index=True)
    
    # Video Info
    video_url = Column(Text, nullable=False)
    video_name = Column(String(255))
    duration = Column(Float)  # seconds
    
    # Batch Info
    batch_id = Column(String(50), index=True)
    batch_name = Column(String(100))
    
    # ⭐ NEW: Pipeline State Tracking
    pipeline_stage = Column(
        String(50),
        default='generating',
        index=True
    )  # Values: 'generating', 'awaiting_stage1_review', 'validating', 
       #         'testing_gemini', 'awaiting_stage2_selection', 'completed', 'failed'
    
    stage1_review_progress = Column(Integer, default=0)  # How many questions reviewed
    stage1_total_questions = Column(Integer, default=30)  # Total questions generated
    stage1_approved = Column(Integer, default=0)
    stage1_rejected = Column(Integer, default=0)
    
    stage2_failures_count = Column(Integer, default=0)  # How many Gemini failures
    stage2_selected_count = Column(Integer, default=0)  # How many selected (should be 4)
    
    # Processing Status (original)
    status = Column(
        String(20),
        default="pending",
        nullable=False,
        index=True
    )  # pending, processing, completed, failed
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Processing Details
    processing_time_seconds = Column(Float)
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    
    # Evidence Extraction
    evidence_extracted = Column(Boolean, default=False)
    evidence_confidence = Column(Float)
    transcript_word_count = Column(Integer)
    visual_detections_count = Column(Integer)

    # Evidence Review (HITL)
    evidence_extraction_status = Column(
        String(50),
        default='pending',
        index=True
    )  # Values: 'pending', 'extracting', 'ai_complete', 'awaiting_review', 'review_complete', 'review_failed'
    ai_evidence_count = Column(Integer, default=0)  # Total evidence items created
    evidence_needs_review_count = Column(Integer, default=0)  # Items flagged for review
    evidence_approved_count = Column(Integer, default=0)  # Items approved by human reviewers
    evidence_accuracy_estimate = Column(Float)  # Estimated accuracy based on reviews (0.0-1.0)

    # Q&A Generation
    candidates_generated = Column(Integer)
    candidates_validated = Column(Integer)
    questions_tested = Column(Integer)
    
    # Results
    failures_found = Column(Integer)
    final_selected = Column(Integer)
    avg_failure_score = Column(Float)
    
    # Costs
    cost_evidence_extraction = Column(Float, default=0.0)
    cost_qa_generation = Column(Float, default=0.0)
    cost_gemini_testing = Column(Float, default=0.0)
    cost_total = Column(Float, default=0.0)
    
    # Quality Metrics
    validation_pass_rate = Column(Float)
    final_confidence = Column(Float)
    
    # Relationships
    questions = relationship("Question", back_populates="video", cascade="all, delete-orphan")
    failures = relationship("Failure", back_populates="video", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_video_status_batch', 'status', 'batch_id'),
        Index('idx_video_created', 'created_at'),
        Index('idx_video_pipeline_stage', 'pipeline_stage'),  # ⭐ NEW
    )
    
    def __repr__(self):
        return f"<Video(id={self.id}, video_id={self.video_id}, stage={self.pipeline_stage})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'video_id': self.video_id,
            'video_url': self.video_url,
            'video_name': self.video_name,
            'duration': self.duration,
            'batch_id': self.batch_id,
            'pipeline_stage': self.pipeline_stage,
            'stage1_review_progress': self.stage1_review_progress,
            'stage1_total_questions': self.stage1_total_questions,
            'stage1_approved': self.stage1_approved,
            'stage1_rejected': self.stage1_rejected,
            'stage2_failures_count': self.stage2_failures_count,
            'stage2_selected_count': self.stage2_selected_count,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'processing_time_seconds': self.processing_time_seconds,
            'failures_found': self.failures_found,
            'final_selected': self.final_selected,
            'cost_total': self.cost_total,
            # Evidence Review (HITL)
            'evidence_extraction_status': self.evidence_extraction_status,
            'ai_evidence_count': self.ai_evidence_count,
            'evidence_needs_review_count': self.evidence_needs_review_count,
            'evidence_approved_count': self.evidence_approved_count,
            'evidence_accuracy_estimate': self.evidence_accuracy_estimate,
        }


# ==================== QUESTION TABLE (UPDATED) ====================

class Question(Base):
    """Generated questions (all candidates) with review tracking"""
    __tablename__ = "questions"
    
    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)
    question_id = Column(String(50), unique=True, nullable=False, index=True)
    
    # Foreign Key
    video_id = Column(Integer, ForeignKey('videos.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Generation Info
    generation_tier = Column(String(20))  # template, llama, gpt4mini
    task_type = Column(String(50))  # Temporal Understanding, Sequential, etc.
    template_name = Column(String(100))
    
    # Question & Answer
    question_text = Column(Text, nullable=False)
    golden_answer = Column(Text, nullable=False)
    
    # Timestamps
    start_timestamp = Column(String(20))  # HH:MM:SS
    end_timestamp = Column(String(20))    # HH:MM:SS
    start_seconds = Column(Float)
    end_seconds = Column(Float)
    
    # Cues
    audio_cues = Column(JSON)  # List of audio cue descriptions
    visual_cues = Column(JSON)  # List of visual cue descriptions
    
    # Evidence References
    evidence_refs = Column(JSON)  # References to evidence database
    
    # Validation Results
    validation_passed = Column(Boolean, default=False)
    validation_results = Column(JSON)  # Detailed validation results
    confidence_score = Column(Float)
    
    # Gemini Testing
    tested_with_gemini = Column(Boolean, default=False)
    gemini_answer = Column(Text)
    gemini_failed = Column(Boolean)
    
    # Selection (original)
    pre_filtered = Column(Boolean, default=False)
    pre_filter_score = Column(Float)
    final_selected = Column(Boolean, default=False)
    
    # ⭐ NEW: Stage 1 Review Fields
    stage1_review_status = Column(
        String(20),
        default='pending',
        index=True
    )  # Values: 'pending', 'approved', 'rejected'
    
    stage1_reviewer = Column(String(100))  # Email or username of reviewer
    stage1_feedback = Column(Text)  # Feedback if rejected
    stage1_reviewed_at = Column(DateTime)
    
    # ⭐ NEW: Regeneration Tracking
    generation_attempt = Column(Integer, default=1)  # 1, 2, 3 (max 3)
    parent_question_id = Column(Integer, ForeignKey('questions.id'), nullable=True)
    is_regeneration = Column(Boolean, default=False)
    
    # ⭐ NEW: Stage 2 Selection Fields
    stage2_shown_for_selection = Column(Boolean, default=False)  # Is this a Gemini failure?
    stage2_manually_selected = Column(Boolean, default=False)  # Did human select this?
    stage2_selection_rank = Column(Integer)  # 1, 2, 3, 4 (for final 4)
    stage2_reviewer = Column(String(100))
    stage2_selected_at = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    video = relationship("Video", back_populates="questions")
    failure = relationship("Failure", back_populates="question", uselist=False)
    
    # Self-referential relationship for regeneration
    parent_question = relationship("Question", remote_side=[id], foreign_keys=[parent_question_id])
    
    # Indexes
    __table_args__ = (
        Index('idx_question_video_tier', 'video_id', 'generation_tier'),
        Index('idx_question_task_type', 'task_type'),
        Index('idx_question_selected', 'final_selected'),
        Index('idx_question_stage1_review', 'stage1_review_status'),  # ⭐ NEW
        Index('idx_question_stage2_selection', 'stage2_manually_selected'),  # ⭐ NEW
        Index('idx_question_generation_attempt', 'generation_attempt'),  # ⭐ NEW
    )
    
    def __repr__(self):
        return f"<Question(id={self.id}, task_type={self.task_type}, stage1={self.stage1_review_status})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'question_id': self.question_id,
            'video_id': self.video_id,
            'generation_tier': self.generation_tier,
            'task_type': self.task_type,
            'question_text': self.question_text,
            'golden_answer': self.golden_answer,
            'start_timestamp': self.start_timestamp,
            'end_timestamp': self.end_timestamp,
            'audio_cues': self.audio_cues,
            'visual_cues': self.visual_cues,
            'validation_passed': self.validation_passed,
            'confidence_score': self.confidence_score,
            'tested_with_gemini': self.tested_with_gemini,
            'gemini_answer': self.gemini_answer,
            'gemini_failed': self.gemini_failed,
            # Stage 1 Review
            'stage1_review_status': self.stage1_review_status,
            'stage1_reviewer': self.stage1_reviewer,
            'stage1_feedback': self.stage1_feedback,
            'stage1_reviewed_at': self.stage1_reviewed_at.isoformat() if self.stage1_reviewed_at else None,
            # Regeneration
            'generation_attempt': self.generation_attempt,
            'parent_question_id': self.parent_question_id,
            'is_regeneration': self.is_regeneration,
            # Stage 2 Selection
            'stage2_shown_for_selection': self.stage2_shown_for_selection,
            'stage2_manually_selected': self.stage2_manually_selected,
            'stage2_selection_rank': self.stage2_selection_rank,
            'stage2_selected_at': self.stage2_selected_at.isoformat() if self.stage2_selected_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


# ==================== FAILURE TABLE (UNCHANGED) ====================

class Failure(Base):
    """Gemini failures (where Gemini answered incorrectly)"""
    __tablename__ = "failures"
    
    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)
    failure_id = Column(String(50), unique=True, nullable=False, index=True)
    
    # Foreign Keys
    video_id = Column(Integer, ForeignKey('videos.id', ondelete='CASCADE'), nullable=False, index=True)
    question_id = Column(Integer, ForeignKey('questions.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Failure Classification
    failure_type = Column(String(50), index=True)  # counting_error, context_missing, etc.
    failure_score = Column(Float)  # 0-10 score
    
    # Severity & Clarity
    severity_score = Column(Float)  # How wrong is Gemini?
    clarity_score = Column(Float)   # How obvious is the error?
    educational_score = Column(Float)  # How valuable is this?
    task_type_rarity_score = Column(Float)  # How rare is this task type?
    
    # Gemini Response
    gemini_answer = Column(Text, nullable=False)
    golden_answer = Column(Text, nullable=False)
    
    # Analysis
    difference_summary = Column(Text)  # What's different?
    explanation = Column(Text)  # Why is Gemini wrong?
    
    # Selection
    diversity_bonus = Column(Float, default=0.0)
    adjusted_score = Column(Float)  # Score after diversity bonus
    final_selected = Column(Boolean, default=False)
    selection_rank = Column(Integer)  # 1-4 for selected failures
    
    # Timestamps
    detected_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    video = relationship("Video", back_populates="failures")
    question = relationship("Question", back_populates="failure")
    
    # Indexes
    __table_args__ = (
        Index('idx_failure_type_score', 'failure_type', 'failure_score'),
        Index('idx_failure_selected', 'final_selected'),
        Index('idx_failure_video', 'video_id', 'final_selected'),
    )
    
    def __repr__(self):
        return f"<Failure(id={self.id}, type={self.failure_type}, score={self.failure_score})>"


# ==================== OTHER TABLES (UNCHANGED) ====================

# FeedbackPattern, Batch, SystemMetrics remain the same as original schema.py
# Copy them from your original file

class FeedbackPattern(Base):
    """Learning: successful failure patterns for improvement"""
    __tablename__ = "feedback_patterns"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    pattern_name = Column(String(100), index=True)
    pattern_type = Column(String(50))
    question_template = Column(Text)
    task_type = Column(String(50), index=True)
    failure_type = Column(String(50), index=True)
    times_used = Column(Integer, default=0)
    times_succeeded = Column(Integer, default=0)
    success_rate = Column(Float, default=0.0)
    avg_failure_score = Column(Float, default=0.0)
    current_weight = Column(Float, default=1.0)
    last_weight_update = Column(DateTime)
    discovered_at = Column(DateTime, default=datetime.utcnow)
    last_used = Column(DateTime)
    is_active = Column(Boolean, default=True)
    is_template = Column(Boolean, default=False)
    
    __table_args__ = (
        Index('idx_pattern_active_type', 'is_active', 'pattern_type'),
        Index('idx_pattern_success_rate', 'success_rate'),
    )


class Batch(Base):
    """Batch processing metadata"""
    __tablename__ = "batches"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    batch_id = Column(String(50), unique=True, nullable=False, index=True)
    batch_name = Column(String(100))
    total_videos = Column(Integer, default=0)
    status = Column(String(20), default="pending", index=True)
    videos_pending = Column(Integer, default=0)
    videos_processing = Column(Integer, default=0)
    videos_completed = Column(Integer, default=0)
    videos_failed = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    total_cost = Column(Float, default=0.0)
    avg_cost_per_video = Column(Float)
    avg_failure_score = Column(Float)
    avg_confidence = Column(Float)
    settings_snapshot = Column(JSON)


class SystemMetrics(Base):
    """System-wide metrics for monitoring"""
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    recorded_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    videos_processed_total = Column(Integer, default=0)
    videos_processed_today = Column(Integer, default=0)
    videos_processed_this_week = Column(Integer, default=0)
    total_cost = Column(Float, default=0.0)
    avg_cost_per_video = Column(Float, default=0.0)
    cost_trend = Column(String(20))
    avg_failure_score = Column(Float, default=0.0)
    avg_confidence = Column(Float, default=0.0)
    validation_pass_rate = Column(Float, default=0.0)
    failure_type_distribution = Column(JSON)
    top_templates = Column(JSON)
    processing_time_avg = Column(Float)
    error_rate = Column(Float)


# ==================== DATABASE INITIALIZATION ====================

def init_db(database_url: str = None):
    """Initialize database - create all tables"""
    from sqlalchemy import create_engine
    from config.settings import settings
    from utils.logger import app_logger
    
    url = database_url or settings.database_url
    
    app_logger.info(f"Initializing database at {url[:30]}...")
    
    engine = create_engine(
        url,
        pool_size=settings.database_pool_size,
        max_overflow=10,
        pool_pre_ping=True,
        echo=False
    )
    
    Base.metadata.create_all(engine)
    
    app_logger.info("✅ Database initialized successfully")
    app_logger.info(f"Created tables: {', '.join(Base.metadata.tables.keys())}")
    
    return engine


def drop_all_tables(database_url: str = None):
    """Drop all tables - USE WITH CAUTION!"""
    from sqlalchemy import create_engine
    from config.settings import settings
    from utils.logger import app_logger
    
    url = database_url or settings.database_url
    
    app_logger.warning(f"⚠️  Dropping all tables from {url[:30]}...")
    
    engine = create_engine(url)
    Base.metadata.drop_all(engine)
    
    app_logger.warning("🗑️  All tables dropped")