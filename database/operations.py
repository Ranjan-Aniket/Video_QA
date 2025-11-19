"""
Database operations - UPDATED WITH REVIEW OPERATIONS

New operations for human review workflow:
- ReviewOperations: Stage 1 & Stage 2 review CRUD
- Pipeline state management
- Regeneration tracking
"""
from sqlalchemy import create_engine, func, and_, or_
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

from config.settings import settings
from utils.logger import get_logger
from .schema import (
    Base, Video, Question, Failure, FeedbackPattern, Batch, SystemMetrics
)

logger = get_logger("database")

# ==================== CONNECTION MANAGEMENT (UNCHANGED) ====================

class DatabaseManager:
    """Database connection and session management"""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or settings.database_url
        
        logger.info(f"Initializing DatabaseManager with URL: {self.database_url[:40]}...")
        
        self.engine = create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=settings.database_pool_size,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False
        )
        
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
            expire_on_commit=False
        )
        
        logger.info("✅ DatabaseManager initialized successfully")
    
    @contextmanager
    def get_session(self) -> Session:
        """Context manager for database sessions"""
        session = self.SessionLocal()
        try:
            logger.debug("Database session created")
            yield session
            session.commit()
            logger.debug("Database session committed")
        except Exception as e:
            session.rollback()
            logger.error(f"Database session rolled back due to error: {e}")
            raise
        finally:
            session.close()
            logger.debug("Database session closed")
    
    def create_tables(self):
        """Create all tables"""
        logger.info("Creating database tables...")
        Base.metadata.create_all(self.engine)
        logger.info("✅ All tables created")
    
    def drop_tables(self):
        """Drop all tables - USE WITH CAUTION"""
        logger.warning("⚠️  Dropping all database tables...")
        Base.metadata.drop_all(self.engine)
        logger.warning("🗑️  All tables dropped")


db_manager = DatabaseManager()


# ==================== VIDEO OPERATIONS (UNCHANGED) ====================

class VideoOperations:
    """CRUD operations for Video table"""
    
    @staticmethod
    def create_video(
        video_id: str,
        video_url: str,
        batch_id: str = None,
        batch_name: str = None,
        video_name: str = None,
        duration: float = None
    ) -> Video:
        """Create new video record"""
        with db_manager.get_session() as session:
            video = Video(
                video_id=video_id,
                video_url=video_url,
                batch_id=batch_id,
                batch_name=batch_name,
                video_name=video_name,
                duration=duration,
                status="pending",
                pipeline_stage="generating",  # ⭐ NEW
                created_at=datetime.utcnow()
            )
            
            session.add(video)
            session.flush()
            
            logger.info(f"Created video record: {video_id}")
            
            return video
    
    @staticmethod
    def get_video(video_id: str) -> Optional[Video]:
        """Get video by ID"""
        with db_manager.get_session() as session:
            video = session.query(Video).filter(Video.video_id == video_id).first()
            
            if video:
                logger.debug(f"Retrieved video: {video_id}")
            else:
                logger.warning(f"Video not found: {video_id}")
            
            return video
    
    @staticmethod
    def update_video(video_id: str, **kwargs) -> Optional[Video]:
        """Update video fields"""
        with db_manager.get_session() as session:
            video = session.query(Video).filter(Video.video_id == video_id).first()
            
            if not video:
                logger.warning(f"Video not found for update: {video_id}")
                return None
            
            for key, value in kwargs.items():
                if hasattr(video, key):
                    setattr(video, key, value)
            
            session.flush()
            
            logger.info(f"Updated video {video_id}: {list(kwargs.keys())}")
            
            return video
    
    @staticmethod
    def update_pipeline_stage(video_id: str, stage: str):
        """Update video pipeline stage"""
        VideoOperations.update_video(video_id, pipeline_stage=stage)
        logger.info(f"Video {video_id} pipeline stage: {stage}")
    
    @staticmethod
    def get_videos_by_pipeline_stage(stage: str) -> List[Video]:
        """Get videos at specific pipeline stage"""
        with db_manager.get_session() as session:
            videos = session.query(Video).filter(Video.pipeline_stage == stage).all()

            logger.debug(f"Retrieved {len(videos)} videos in stage '{stage}'")

            return videos

    # ========== Evidence Review (HITL) Methods ==========

    @staticmethod
    def get_video_by_id(video_id: str) -> Optional[Dict[str, Any]]:
        """
        Get video by ID and return as dictionary

        Args:
            video_id: Video ID

        Returns:
            Video dict or None
        """
        video = VideoOperations.get_video(video_id)
        return video.to_dict() if video else None

    @staticmethod
    def get_videos_by_evidence_status(status: str) -> List[Dict[str, Any]]:
        """
        Get videos by evidence extraction status

        Args:
            status: Evidence status ('pending', 'extracting', 'ai_complete',
                   'awaiting_review', 'review_complete', 'review_failed')

        Returns:
            List of video dicts
        """
        with db_manager.get_session() as session:
            videos = session.query(Video).filter(
                Video.evidence_extraction_status == status
            ).all()

            logger.debug(f"Retrieved {len(videos)} videos with evidence status '{status}'")

            return [video.to_dict() for video in videos]

    @staticmethod
    def update_video_evidence_stats(video_id: str) -> bool:
        """
        Update video evidence statistics from evidence_items table

        Recalculates:
        - ai_evidence_count
        - evidence_needs_review_count
        - evidence_approved_count
        - evidence_accuracy_estimate

        Args:
            video_id: Video ID

        Returns:
            True if updated successfully
        """
        from database.evidence_operations import EvidenceOperations

        with db_manager.get_session() as session:
            video = session.query(Video).filter(Video.video_id == video_id).first()

            if not video:
                logger.warning(f"Video not found: {video_id}")
                return False

            # Get evidence statistics
            progress = EvidenceOperations.get_review_progress(video_id)

            video.ai_evidence_count = progress.get('total', 0)
            video.evidence_needs_review_count = progress.get('needs_review', 0)
            video.evidence_approved_count = progress.get('approved', 0)

            # Calculate accuracy estimate
            total = progress.get('total', 0)
            approved = progress.get('approved', 0)
            corrected = progress.get('corrected', 0)

            if (approved + corrected) > 0:
                accuracy = approved / (approved + corrected)
                video.evidence_accuracy_estimate = round(accuracy, 3)
            else:
                video.evidence_accuracy_estimate = None

            session.flush()

            logger.info(f"Updated evidence stats for video {video_id}: "
                       f"total={video.ai_evidence_count}, "
                       f"needs_review={video.evidence_needs_review_count}, "
                       f"approved={video.evidence_approved_count}, "
                       f"accuracy={video.evidence_accuracy_estimate}")

            return True


# ==================== ⭐ NEW: REVIEW OPERATIONS ====================

class ReviewOperations:
    """Operations for human review workflow"""
    
    # ========== STAGE 1: REVIEW OPERATIONS ==========
    
    @staticmethod
    def get_stage1_questions(video_id: int) -> List[Question]:
        """
        Get all questions for Stage 1 review
        
        Args:
            video_id: Video database ID
            
        Returns:
            List of all questions (pending, approved, rejected)
        """
        with db_manager.get_session() as session:
            questions = session.query(Question).filter(
                Question.video_id == video_id
            ).order_by(Question.created_at).all()
            
            logger.info(f"Retrieved {len(questions)} questions for Stage 1 review (video {video_id})")
            
            return questions
    
    @staticmethod
    def get_stage1_pending_questions(video_id: int) -> List[Question]:
        """Get questions pending Stage 1 review"""
        with db_manager.get_session() as session:
            questions = session.query(Question).filter(
                and_(
                    Question.video_id == video_id,
                    Question.stage1_review_status == 'pending'
                )
            ).order_by(Question.created_at).all()
            
            logger.debug(f"Retrieved {len(questions)} pending questions for video {video_id}")
            
            return questions
    
    @staticmethod
    def approve_question(question_id: int, reviewer: str) -> Optional[Question]:
        """
        Approve a question in Stage 1 review
        
        Args:
            question_id: Question database ID
            reviewer: Email/username of reviewer
            
        Returns:
            Updated Question object
        """
        with db_manager.get_session() as session:
            question = session.query(Question).filter(Question.id == question_id).first()
            
            if not question:
                logger.warning(f"Question not found: {question_id}")
                return None
            
            question.stage1_review_status = 'approved'
            question.stage1_reviewer = reviewer
            question.stage1_reviewed_at = datetime.utcnow()
            question.stage1_feedback = None  # Clear any previous feedback
            
            session.flush()
            
            # Update video progress
            video = session.query(Video).filter(Video.id == question.video_id).first()
            if video:
                video.stage1_approved = session.query(func.count(Question.id)).filter(
                    and_(
                        Question.video_id == question.video_id,
                        Question.stage1_review_status == 'approved'
                    )
                ).scalar()
                
                video.stage1_review_progress = video.stage1_approved + video.stage1_rejected
            
            logger.info(f"Approved question {question_id} by {reviewer}")
            
            return question
    
    @staticmethod
    def reject_question(question_id: int, reviewer: str, feedback: str) -> Optional[Question]:
        """
        Reject a question in Stage 1 review
        
        Args:
            question_id: Question database ID
            reviewer: Email/username of reviewer
            feedback: Rejection feedback
            
        Returns:
            Updated Question object
        """
        with db_manager.get_session() as session:
            question = session.query(Question).filter(Question.id == question_id).first()
            
            if not question:
                logger.warning(f"Question not found: {question_id}")
                return None
            
            question.stage1_review_status = 'rejected'
            question.stage1_reviewer = reviewer
            question.stage1_feedback = feedback
            question.stage1_reviewed_at = datetime.utcnow()
            
            session.flush()
            
            # Update video progress
            video = session.query(Video).filter(Video.id == question.video_id).first()
            if video:
                video.stage1_rejected = session.query(func.count(Question.id)).filter(
                    and_(
                        Question.video_id == question.video_id,
                        Question.stage1_review_status == 'rejected'
                    )
                ).scalar()
                
                video.stage1_review_progress = video.stage1_approved + video.stage1_rejected
            
            logger.info(f"Rejected question {question_id} by {reviewer}: {feedback[:50]}")
            
            return question
    
    @staticmethod
    def bulk_approve(question_ids: List[int], reviewer: str) -> int:
        """
        Approve multiple questions at once
        
        Args:
            question_ids: List of question database IDs
            reviewer: Email/username of reviewer
            
        Returns:
            Number of questions approved
        """
        with db_manager.get_session() as session:
            count = session.query(Question).filter(
                Question.id.in_(question_ids)
            ).update({
                'stage1_review_status': 'approved',
                'stage1_reviewer': reviewer,
                'stage1_reviewed_at': datetime.utcnow(),
                'stage1_feedback': None
            }, synchronize_session=False)
            
            session.commit()
            
            logger.info(f"Bulk approved {count} questions by {reviewer}")
            
            return count
    
    @staticmethod
    def bulk_reject(question_ids: List[int], reviewer: str, feedback: str) -> int:
        """Bulk reject multiple questions"""
        with db_manager.get_session() as session:
            count = session.query(Question).filter(
                Question.id.in_(question_ids)
            ).update({
                'stage1_review_status': 'rejected',
                'stage1_reviewer': reviewer,
                'stage1_reviewed_at': datetime.utcnow(),
                'stage1_feedback': feedback
            }, synchronize_session=False)
            
            session.commit()
            
            logger.info(f"Bulk rejected {count} questions by {reviewer}")
            
            return count
    
    @staticmethod
    def get_stage1_progress(video_id: int) -> Dict[str, int]:
        """
        Get Stage 1 review progress
        
        Args:
            video_id: Video database ID
            
        Returns:
            Dictionary with counts: {approved, rejected, pending, total}
        """
        with db_manager.get_session() as session:
            total = session.query(func.count(Question.id)).filter(
                Question.video_id == video_id
            ).scalar()
            
            approved = session.query(func.count(Question.id)).filter(
                and_(
                    Question.video_id == video_id,
                    Question.stage1_review_status == 'approved'
                )
            ).scalar()
            
            rejected = session.query(func.count(Question.id)).filter(
                and_(
                    Question.video_id == video_id,
                    Question.stage1_review_status == 'rejected'
                )
            ).scalar()
            
            pending = total - (approved + rejected)
            
            progress = {
                'approved': approved,
                'rejected': rejected,
                'pending': pending,
                'total': total,
                'progress_percent': ((approved + rejected) / total * 100) if total > 0 else 0
            }
            
            logger.debug(f"Stage 1 progress for video {video_id}: {progress}")
            
            return progress
    
    @staticmethod
    def is_stage1_complete(video_id: int) -> bool:
        """Check if Stage 1 review is complete"""
        progress = ReviewOperations.get_stage1_progress(video_id)
        return progress['pending'] == 0 and progress['total'] > 0
    
    @staticmethod
    def get_approved_questions(video_id: int) -> List[Question]:
        """Get all approved questions"""
        with db_manager.get_session() as session:
            questions = session.query(Question).filter(
                and_(
                    Question.video_id == video_id,
                    Question.stage1_review_status == 'approved'
                )
            ).all()
            
            logger.info(f"Retrieved {len(questions)} approved questions for video {video_id}")
            
            return questions
    
    # ========== REGENERATION OPERATIONS ==========
    
    @staticmethod
    def can_regenerate(question_id: int) -> Tuple[bool, str]:
        """
        Check if question can be regenerated
        
        Args:
            question_id: Question database ID
            
        Returns:
            (can_regenerate, reason)
        """
        with db_manager.get_session() as session:
            question = session.query(Question).filter(Question.id == question_id).first()
            
            if not question:
                return False, "Question not found"
            
            if question.generation_attempt >= 3:
                return False, "Maximum regeneration attempts (3) reached"
            
            if question.stage1_review_status != 'rejected':
                return False, "Only rejected questions can be regenerated"
            
            return True, "OK"
    
    @staticmethod
    def mark_question_for_regeneration(question_id: int):
        """Mark question as needing regeneration"""
        with db_manager.get_session() as session:
            question = session.query(Question).filter(Question.id == question_id).first()
            
            if question:
                question.stage1_review_status = 'pending_regeneration'
                session.flush()
                
                logger.info(f"Marked question {question_id} for regeneration")
    
    # ========== STAGE 2: SELECTION OPERATIONS ==========
    
    @staticmethod
    def get_stage2_failures(video_id: int) -> List[Dict]:
        """
        Get all Gemini failures for Stage 2 selection
        
        Args:
            video_id: Video database ID
            
        Returns:
            List of failures with question data, ranked by score
        """
        with db_manager.get_session() as session:
            # Join Failure with Question
            results = session.query(Failure, Question).join(
                Question, Failure.question_id == Question.id
            ).filter(
                Failure.video_id == video_id
            ).order_by(Failure.failure_score.desc()).all()
            
            failures = []
            for failure, question in results:
                failures.append({
                    'failure_id': failure.failure_id,
                    'failure': failure,
                    'question': question,
                    'combined_score': failure.adjusted_score or failure.failure_score
                })
            
            logger.info(f"Retrieved {len(failures)} Gemini failures for Stage 2 selection (video {video_id})")
            
            return failures
    
    @staticmethod
    def submit_stage2_selection(
        video_id: int,
        selected_question_ids: List[int],
        reviewer: str
    ) -> List[Question]:
        """
        Submit Stage 2 manual selection (final 4)
        
        Args:
            video_id: Video database ID
            selected_question_ids: List of 4 question IDs selected
            reviewer: Email/username of reviewer
            
        Returns:
            List of selected Question objects
        """
        if len(selected_question_ids) != 4:
            raise ValueError(f"Must select exactly 4 questions, got {len(selected_question_ids)}")
        
        with db_manager.get_session() as session:
            selected_questions = []
            
            for rank, question_id in enumerate(selected_question_ids, start=1):
                question = session.query(Question).filter(Question.id == question_id).first()
                
                if not question:
                    logger.warning(f"Question {question_id} not found for selection")
                    continue
                
                question.stage2_manually_selected = True
                question.stage2_selection_rank = rank
                question.stage2_reviewer = reviewer
                question.stage2_selected_at = datetime.utcnow()
                
                selected_questions.append(question)
            
            session.flush()
            
            # Update video
            video = session.query(Video).filter(Video.id == video_id).first()
            if video:
                video.stage2_selected_count = len(selected_questions)
                video.final_selected = len(selected_questions)
            
            logger.info(
                f"Stage 2 selection complete: {len(selected_questions)} questions "
                f"selected by {reviewer} for video {video_id}"
            )
            
            return selected_questions
    
    @staticmethod
    def get_stage2_selected(video_id: int) -> List[Question]:
        """Get Stage 2 selected questions (final 4)"""
        with db_manager.get_session() as session:
            questions = session.query(Question).filter(
                and_(
                    Question.video_id == video_id,
                    Question.stage2_manually_selected == True
                )
            ).order_by(Question.stage2_selection_rank).all()
            
            logger.info(f"Retrieved {len(questions)} Stage 2 selected questions for video {video_id}")
            
            return questions


# ==================== QUESTION OPERATIONS (KEEP EXISTING + ADD NEW) ====================

class QuestionOperations:
    """CRUD operations for Question table"""
    
    @staticmethod
    def create_question(
        question_id: str,
        video_id: int,
        question_text: str,
        golden_answer: str,
        generation_tier: str,
        task_type: str = None,
        template_name: str = None,
        start_seconds: float = None,
        end_seconds: float = None,
        audio_cues: list = None,
        visual_cues: list = None,
        evidence_refs: list = None,
        confidence_score: float = None,
        **kwargs
    ) -> Question:
        """Create new question"""
        with db_manager.get_session() as session:
            question = Question(
                question_id=question_id,
                video_id=video_id,
                question_text=question_text,
                golden_answer=golden_answer,
                generation_tier=generation_tier,
                task_type=task_type,
                template_name=template_name,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                audio_cues=audio_cues,
                visual_cues=visual_cues,
                evidence_refs=evidence_refs,
                confidence_score=confidence_score,
                stage1_review_status='pending',  # ⭐ NEW: Default to pending
                generation_attempt=kwargs.get('generation_attempt', 1),  # ⭐ NEW
                parent_question_id=kwargs.get('parent_question_id'),  # ⭐ NEW
                is_regeneration=kwargs.get('is_regeneration', False),  # ⭐ NEW
                created_at=datetime.utcnow()
            )
            
            session.add(question)
            session.flush()
            
            logger.info(f"Created question: {question_id}")
            
            return question
    
    @staticmethod
    def get_question(question_id: str) -> Optional[Question]:
        """Get question by ID"""
        with db_manager.get_session() as session:
            question = session.query(Question).filter(
                Question.question_id == question_id
            ).first()

            if question:
                logger.debug(f"Retrieved question: {question_id}")
            else:
                logger.warning(f"Question not found: {question_id}")

            return question

    @staticmethod
    def get_questions_by_video(video_id: int) -> List[Question]:
        """Get all questions for a video"""
        with db_manager.get_session() as session:
            questions = session.query(Question).filter(
                Question.video_id == video_id
            ).all()

            logger.debug(f"Retrieved {len(questions)} questions for video {video_id}")

            return questions

    # Keep other existing question operations...


# Keep existing FailureOperations, FeedbackOperations, AnalyticsOperations...