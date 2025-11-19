"""
Evidence Operations - Database operations for evidence items and reviews

Handles CRUD operations for evidence items, reviewer performance tracking,
and human-in-the-loop review workflows.
"""

import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, date
import json
from sqlalchemy import select, update, delete, and_, or_, func, text
from sqlalchemy.orm import Session

from .operations import db_manager

logger = logging.getLogger(__name__)


class EvidenceItem:
    """Evidence Item data class"""

    def __init__(self, **kwargs):
        self.id = kwargs.get('id')
        self.video_id = kwargs.get('video_id')
        self.evidence_type = kwargs.get('evidence_type')
        self.timestamp_start = kwargs.get('timestamp_start')
        self.timestamp_end = kwargs.get('timestamp_end')

        # AI predictions
        self.gpt4_prediction = kwargs.get('gpt4_prediction')
        self.claude_prediction = kwargs.get('claude_prediction')
        self.open_model_prediction = kwargs.get('open_model_prediction')

        # Consensus
        self.ai_consensus_reached = kwargs.get('ai_consensus_reached', False)
        self.consensus_answer = kwargs.get('consensus_answer')
        self.confidence_score = kwargs.get('confidence_score')
        self.disagreement_details = kwargs.get('disagreement_details')
        self.needs_human_review = kwargs.get('needs_human_review', False)
        self.priority_level = kwargs.get('priority_level', 'medium')
        self.flag_reason = kwargs.get('flag_reason')

        # Human review
        self.human_review_status = kwargs.get('human_review_status', 'pending')
        self.human_reviewer_id = kwargs.get('human_reviewer_id')
        self.human_answer = kwargs.get('human_answer')
        self.human_confidence = kwargs.get('human_confidence')
        self.human_notes = kwargs.get('human_notes')
        self.review_timestamp = kwargs.get('review_timestamp')
        self.review_duration_seconds = kwargs.get('review_duration_seconds')

        # Quality tracking
        self.ai_was_correct = kwargs.get('ai_was_correct')
        self.correction_category = kwargs.get('correction_category')

        # Metadata
        self.created_at = kwargs.get('created_at')
        self.updated_at = kwargs.get('updated_at')


class EvidenceOperations:
    """Database operations for evidence items"""

    @staticmethod
    def create_evidence_item(
        video_id: str,
        evidence_type: str,
        timestamp_start: float,
        timestamp_end: float,
        gpt4_prediction: Optional[Dict] = None,
        claude_prediction: Optional[Dict] = None,
        open_model_prediction: Optional[Dict] = None,
        ground_truth: Optional[Dict] = None,
        ai_consensus_reached: bool = False,
        consensus_answer: Optional[str] = None,
        confidence_score: float = 0.0,
        needs_review: bool = False,
        priority: str = 'medium',
        flag_reason: Optional[str] = None
    ) -> int:
        """
        Create a new evidence item

        Returns:
            evidence_id (int)
        """
        with db_manager.get_session() as session:
            # Convert dicts to JSON strings
            gpt4_json = json.dumps(gpt4_prediction) if gpt4_prediction else None
            claude_json = json.dumps(claude_prediction) if claude_prediction else None
            open_json = json.dumps(open_model_prediction) if open_model_prediction else None
            ground_truth_json = json.dumps(ground_truth) if ground_truth else None

            # Insert
            result = session.execute(
                text("""
                INSERT INTO evidence_items (
                    video_id, evidence_type, timestamp_start, timestamp_end,
                    gpt4_prediction, claude_prediction, open_model_prediction,
                    ground_truth, ai_consensus_reached, consensus_answer,
                    confidence_score, needs_human_review, priority_level, flag_reason
                ) VALUES (:video_id, :evidence_type, :timestamp_start, :timestamp_end,
                         :gpt4_prediction, :claude_prediction, :open_model_prediction,
                         :ground_truth, :ai_consensus_reached, :consensus_answer,
                         :confidence_score, :needs_review, :priority, :flag_reason)
                """),
                {
                    'video_id': video_id,
                    'evidence_type': evidence_type,
                    'timestamp_start': timestamp_start,
                    'timestamp_end': timestamp_end,
                    'gpt4_prediction': gpt4_json,
                    'claude_prediction': claude_json,
                    'open_model_prediction': open_json,
                    'ground_truth': ground_truth_json,
                    'ai_consensus_reached': ai_consensus_reached,
                    'consensus_answer': consensus_answer,
                    'confidence_score': confidence_score,
                    'needs_review': needs_review,
                    'priority': priority,
                    'flag_reason': flag_reason
                }
            )
            session.commit()

            evidence_id = result.lastrowid
            logger.info(f"Created evidence item {evidence_id} for video {video_id}")
            return evidence_id

    @staticmethod
    def get_evidence_by_id(evidence_id: int) -> Optional[Dict]:
        """Get evidence item by ID"""
        with db_manager.get_session() as session:
            result = session.execute(
                text("SELECT * FROM evidence_items WHERE id = :id"),
                {'id': evidence_id}
            ).fetchone()

            if result:
                # Convert to dict and parse JSON fields
                evidence = dict(result._mapping)
                if evidence.get('gpt4_prediction'):
                    evidence['gpt4_prediction'] = json.loads(evidence['gpt4_prediction'])
                if evidence.get('claude_prediction'):
                    evidence['claude_prediction'] = json.loads(evidence['claude_prediction'])
                if evidence.get('open_model_prediction'):
                    evidence['open_model_prediction'] = json.loads(evidence['open_model_prediction'])
                if evidence.get('ground_truth'):
                    evidence['ground_truth'] = json.loads(evidence['ground_truth'])
                if evidence.get('disagreement_details'):
                    evidence['disagreement_details'] = json.loads(evidence['disagreement_details'])
                if evidence.get('human_answer'):
                    evidence['human_answer'] = json.loads(evidence['human_answer'])

                return evidence
            return None

    @staticmethod
    def get_review_queue(
        priority: Optional[str] = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[Dict]:
        """
        Get evidence items needing review

        Args:
            priority: Filter by priority ('high', 'medium', 'low')
            limit: Max items to return
            offset: Pagination offset

        Returns:
            List of evidence items
        """
        with db_manager.get_session() as session:
            query = """
                SELECT * FROM evidence_items
                WHERE needs_human_review = 1
                  AND human_review_status = 'pending'
            """
            params = {}

            if priority:
                query += " AND priority_level = :priority"
                params['priority'] = priority

            # Order by priority (high first), then by confidence (low first)
            query += """
                ORDER BY
                    CASE priority_level
                        WHEN 'high' THEN 1
                        WHEN 'medium' THEN 2
                        WHEN 'low' THEN 3
                    END,
                    confidence_score ASC
                LIMIT :limit OFFSET :offset
            """
            params['limit'] = limit
            params['offset'] = offset

            results = session.execute(text(query), params).fetchall()

            items = []
            for row in results:
                evidence = dict(row._mapping)
                # Parse JSON fields
                if evidence.get('gpt4_prediction'):
                    evidence['gpt4_prediction'] = json.loads(evidence['gpt4_prediction'])
                if evidence.get('claude_prediction'):
                    evidence['claude_prediction'] = json.loads(evidence['claude_prediction'])
                if evidence.get('open_model_prediction'):
                    evidence['open_model_prediction'] = json.loads(evidence['open_model_prediction'])

                items.append(evidence)

            return items

    @staticmethod
    def submit_review(
        evidence_id: int,
        reviewer_id: str,
        decision: str,  # 'approved', 'rejected', 'corrected', 'skipped'
        corrected_answer: Optional[Dict] = None,
        confidence: Optional[str] = None,
        notes: Optional[str] = None,
        duration_seconds: float = 0.0
    ) -> bool:
        """
        Submit human review for an evidence item

        Args:
            evidence_id: ID of evidence item
            reviewer_id: ID of reviewer
            decision: Review decision
            corrected_answer: If corrected, the new answer
            confidence: Reviewer confidence
            notes: Review notes
            duration_seconds: Time spent reviewing

        Returns:
            Success boolean
        """
        with db_manager.get_session() as session:
            corrected_json = json.dumps(corrected_answer) if corrected_answer else None

            session.execute(
                text("""
                UPDATE evidence_items
                SET human_review_status = :decision,
                    human_reviewer_id = :reviewer_id,
                    human_answer = :answer,
                    human_confidence = :confidence,
                    human_notes = :notes,
                    review_timestamp = :timestamp,
                    review_duration_seconds = :duration,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = :id
                """),
                {
                    'decision': decision,
                    'reviewer_id': reviewer_id,
                    'answer': corrected_json,
                    'confidence': confidence,
                    'notes': notes,
                    'timestamp': datetime.now(),
                    'duration': duration_seconds,
                    'id': evidence_id
                }
            )
            session.commit()

            logger.info(f"Review submitted for evidence {evidence_id} by {reviewer_id}: {decision}")
            return True

    @staticmethod
    def get_review_progress(video_id: str) -> Dict[str, int]:
        """Get review progress for a video"""
        with db_manager.get_session() as session:
            # First get the video's integer ID from the string video_id
            video_result = session.execute(
                text("SELECT id FROM videos WHERE video_id = :video_id"),
                {'video_id': video_id}
            ).fetchone()

            if not video_result:
                logger.warning(f"Video not found: {video_id}")
                return {}

            video_pk = video_result[0]

            # Now query evidence_items using the integer video ID
            result = session.execute(
                text("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN needs_human_review = 1 THEN 1 ELSE 0 END) as needs_review,
                    SUM(CASE WHEN human_review_status = 'pending' THEN 1 ELSE 0 END) as pending,
                    SUM(CASE WHEN human_review_status = 'approved' THEN 1 ELSE 0 END) as approved,
                    SUM(CASE WHEN human_review_status = 'corrected' THEN 1 ELSE 0 END) as corrected,
                    SUM(CASE WHEN human_review_status = 'rejected' THEN 1 ELSE 0 END) as rejected,
                    SUM(CASE WHEN human_review_status = 'skipped' THEN 1 ELSE 0 END) as skipped
                FROM evidence_items
                WHERE video_id = :video_pk
                """),
                {'video_pk': video_pk}
            ).fetchone()

            return dict(result._mapping) if result else {}

    @staticmethod
    def get_reviewer_stats(reviewer_id: str, date_filter: Optional[date] = None) -> Dict:
        """Get reviewer performance stats"""
        with db_manager.get_session() as session:
            query = """
                SELECT
                    COUNT(*) as items_reviewed,
                    AVG(review_duration_seconds) as avg_review_time,
                    SUM(CASE WHEN human_review_status = 'approved' THEN 1 ELSE 0 END) as approved,
                    SUM(CASE WHEN human_review_status = 'corrected' THEN 1 ELSE 0 END) as corrected,
                    SUM(CASE WHEN human_review_status = 'rejected' THEN 1 ELSE 0 END) as rejected
                FROM evidence_items
                WHERE human_reviewer_id = :reviewer_id
            """
            params = {'reviewer_id': reviewer_id}

            if date_filter:
                query += " AND DATE(review_timestamp) = :date_filter"
                params['date_filter'] = date_filter

            result = session.execute(text(query), params).fetchone()

            return dict(result._mapping) if result else {}

    @staticmethod
    def update_video_evidence_stats(video_id: str):
        """Update evidence statistics for a video"""
        with db_manager.get_session() as session:
            # First get the video's integer ID from the string video_id
            video_result = session.execute(
                text("SELECT id FROM videos WHERE video_id = :video_id"),
                {'video_id': video_id}
            ).fetchone()

            if not video_result:
                logger.warning(f"Video not found: {video_id}")
                return

            video_pk = video_result[0]

            # Get counts from evidence_items using integer video ID
            stats = session.execute(
                text("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN needs_human_review = 1 THEN 1 ELSE 0 END) as needs_review,
                    SUM(CASE WHEN human_review_status != 'pending' THEN 1 ELSE 0 END) as approved,
                    AVG(confidence_score) as avg_confidence
                FROM evidence_items
                WHERE video_id = :video_pk
                """),
                {'video_pk': video_pk}
            ).fetchone()

            if stats:
                stats_dict = dict(stats._mapping)
                # Update videos table using string video_id
                session.execute(
                    text("""
                    UPDATE videos
                    SET ai_evidence_count = :total,
                        evidence_needs_review_count = :needs_review,
                        evidence_approved_count = :approved,
                        evidence_accuracy_estimate = :accuracy
                    WHERE video_id = :video_id
                    """),
                    {
                        'total': stats_dict['total'],
                        'needs_review': stats_dict['needs_review'],
                        'approved': stats_dict['approved'],
                        'accuracy': stats_dict['avg_confidence'],
                        'video_id': video_id
                    }
                )
                session.commit()
                logger.info(f"Updated evidence stats for video {video_id}")


class ReviewSessionOperations:
    """Operations for review sessions"""

    @staticmethod
    def start_session(reviewer_id: str, session_type: str = 'priority') -> int:
        """Start a new review session"""
        with db_manager.get_session() as session:
            result = session.execute(
                text("""
                INSERT INTO review_sessions (reviewer_id, session_start, session_type, is_active)
                VALUES (:reviewer_id, :session_start, :session_type, 1)
                """),
                {
                    'reviewer_id': reviewer_id,
                    'session_start': datetime.now(),
                    'session_type': session_type
                }
            )
            session.commit()

            session_id = result.lastrowid
            logger.info(f"Started review session {session_id} for {reviewer_id}")
            return session_id

    @staticmethod
    def end_session(session_id: int, items_reviewed: int):
        """End a review session"""
        with db_manager.get_session() as session:
            session.execute(
                text("""
                UPDATE review_sessions
                SET session_end = :session_end,
                    items_reviewed = :items_reviewed,
                    is_active = 0
                WHERE id = :id
                """),
                {
                    'session_end': datetime.now(),
                    'items_reviewed': items_reviewed,
                    'id': session_id
                }
            )
            session.commit()
            logger.info(f"Ended review session {session_id}")
