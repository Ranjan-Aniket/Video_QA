"""
Evidence Review API Endpoints

Handles human-in-the-loop review workflow for evidence items.
Provides endpoints for reviewers to view, approve, correct, and skip evidence.
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

from database.evidence_operations import EvidenceOperations, ReviewSessionOperations

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ReviewSubmission(BaseModel):
    """Model for submitting a review"""
    decision: str  # 'approved', 'rejected', 'corrected', 'skipped'
    corrected_answer: Optional[Dict[str, Any]] = None
    confidence: Optional[str] = None  # 'high', 'medium', 'low'
    notes: Optional[str] = None
    review_duration_seconds: float = 0.0


class BatchReviewItem(BaseModel):
    """Single item in a batch review"""
    evidence_id: int
    decision: str
    notes: Optional[str] = None


class BatchReviewSubmission(BaseModel):
    """Model for batch review submission"""
    items: List[BatchReviewItem]
    total_duration_seconds: float = 0.0


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("/queue")
async def get_review_queue(
    priority: Optional[str] = Query(None, regex="^(high|medium|low)$"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """
    Get evidence items needing review

    Args:
        priority: Filter by priority level ('high', 'medium', 'low')
        limit: Max items to return (1-100)
        offset: Pagination offset

    Returns:
        List of evidence items in review queue
    """
    try:
        items = EvidenceOperations.get_review_queue(
            priority=priority,
            limit=limit,
            offset=offset
        )

        # Get total count for pagination
        # (You can optimize this with a separate count query)
        total_pending = len(EvidenceOperations.get_review_queue(
            priority=priority,
            limit=10000  # Get all to count
        ))

        return {
            "success": True,
            "items": items,
            "count": len(items),
            "total_pending": total_pending,
            "has_more": (offset + len(items)) < total_pending
        }

    except Exception as e:
        logger.error(f"Failed to get review queue: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{evidence_id}")
async def get_evidence_item(evidence_id: int):
    """
    Get detailed evidence item for review

    Args:
        evidence_id: ID of evidence item

    Returns:
        Full evidence item with AI predictions, ground truth, etc.
    """
    try:
        evidence = EvidenceOperations.get_evidence_by_id(evidence_id)

        if not evidence:
            raise HTTPException(status_code=404, detail="Evidence item not found")

        return {
            "success": True,
            "evidence": evidence
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get evidence item {evidence_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{evidence_id}/review")
async def submit_review(
    evidence_id: int,
    submission: ReviewSubmission,
    reviewer_id: str = Query(..., description="ID of the reviewer")
):
    """
    Submit review for an evidence item

    Args:
        evidence_id: ID of evidence item
        submission: Review submission data
        reviewer_id: ID of reviewer (from auth)

    Returns:
        Success confirmation
    """
    try:
        # Validate decision
        valid_decisions = ['approved', 'rejected', 'corrected', 'skipped']
        if submission.decision not in valid_decisions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid decision. Must be one of: {valid_decisions}"
            )

        # If corrected, must provide corrected_answer
        if submission.decision == 'corrected' and not submission.corrected_answer:
            raise HTTPException(
                status_code=400,
                detail="Must provide corrected_answer when decision is 'corrected'"
            )

        # Submit review
        success = EvidenceOperations.submit_review(
            evidence_id=evidence_id,
            reviewer_id=reviewer_id,
            decision=submission.decision,
            corrected_answer=submission.corrected_answer,
            confidence=submission.confidence,
            notes=submission.notes,
            duration_seconds=submission.review_duration_seconds
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to submit review")

        return {
            "success": True,
            "message": "Review submitted successfully",
            "evidence_id": evidence_id,
            "decision": submission.decision
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit review for {evidence_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch")
async def submit_batch_review(
    submission: BatchReviewSubmission,
    reviewer_id: str = Query(..., description="ID of the reviewer")
):
    """
    Submit multiple reviews in batch (for spot-checking)

    Args:
        submission: Batch review data
        reviewer_id: ID of reviewer

    Returns:
        Batch submission results
    """
    try:
        results = []
        avg_duration = submission.total_duration_seconds / len(submission.items) if submission.items else 0

        for item in submission.items:
            try:
                success = EvidenceOperations.submit_review(
                    evidence_id=item.evidence_id,
                    reviewer_id=reviewer_id,
                    decision=item.decision,
                    corrected_answer=None,
                    confidence=None,
                    notes=item.notes,
                    duration_seconds=avg_duration
                )

                results.append({
                    "evidence_id": item.evidence_id,
                    "success": success,
                    "error": None
                })

            except Exception as e:
                logger.error(f"Failed to submit review for {item.evidence_id}: {e}")
                results.append({
                    "evidence_id": item.evidence_id,
                    "success": False,
                    "error": str(e)
                })

        successful_count = sum(1 for r in results if r["success"])

        return {
            "success": True,
            "results": results,
            "total_submitted": len(submission.items),
            "successful": successful_count,
            "failed": len(submission.items) - successful_count
        }

    except Exception as e:
        logger.error(f"Failed to submit batch review: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{evidence_id}/skip")
async def skip_evidence_item(
    evidence_id: int,
    reason: str = Query(..., description="Reason for skipping"),
    reviewer_id: str = Query(..., description="ID of the reviewer")
):
    """
    Skip/flag evidence item as too ambiguous to review

    Args:
        evidence_id: ID of evidence item
        reason: Reason for skipping
        reviewer_id: ID of reviewer

    Returns:
        Success confirmation
    """
    try:
        success = EvidenceOperations.submit_review(
            evidence_id=evidence_id,
            reviewer_id=reviewer_id,
            decision='skipped',
            corrected_answer=None,
            confidence=None,
            notes=f"Skipped: {reason}",
            duration_seconds=0.0
        )

        return {
            "success": True,
            "message": "Evidence item skipped",
            "evidence_id": evidence_id
        }

    except Exception as e:
        logger.error(f"Failed to skip evidence {evidence_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/progress/{video_id}")
async def get_review_progress(video_id: str):
    """
    Get review progress for a video

    Args:
        video_id: ID of the video

    Returns:
        Review progress statistics
    """
    try:
        progress = EvidenceOperations.get_review_progress(video_id)

        # Calculate percentages
        total = progress.get('total', 0)
        if total > 0:
            progress['percent_complete'] = round(
                (total - progress.get('pending', 0)) / total * 100,
                1
            )
            progress['percent_approved'] = round(
                progress.get('approved', 0) / total * 100,
                1
            )
        else:
            progress['percent_complete'] = 0
            progress['percent_approved'] = 0

        return {
            "success": True,
            "video_id": video_id,
            "progress": progress
        }

    except Exception as e:
        logger.error(f"Failed to get progress for {video_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/{reviewer_id}")
async def get_reviewer_stats(
    reviewer_id: str,
    date: Optional[str] = Query(None, description="Date filter (YYYY-MM-DD)")
):
    """
    Get reviewer performance statistics

    Args:
        reviewer_id: ID of the reviewer
        date: Optional date filter

    Returns:
        Reviewer performance metrics
    """
    try:
        from datetime import date as date_type

        date_filter = None
        if date:
            try:
                date_filter = date_type.fromisoformat(date)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

        stats = EvidenceOperations.get_reviewer_stats(
            reviewer_id=reviewer_id,
            date_filter=date_filter
        )

        # Calculate additional metrics
        items_reviewed = stats.get('items_reviewed', 0)
        if items_reviewed > 0:
            stats['approval_rate'] = round(
                stats.get('approved', 0) / items_reviewed * 100,
                1
            )
            stats['correction_rate'] = round(
                stats.get('corrected', 0) / items_reviewed * 100,
                1
            )
        else:
            stats['approval_rate'] = 0
            stats['correction_rate'] = 0

        return {
            "success": True,
            "reviewer_id": reviewer_id,
            "date": date,
            "stats": stats
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get stats for {reviewer_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/start")
async def start_review_session(
    reviewer_id: str = Query(..., description="ID of the reviewer"),
    session_type: str = Query('priority', regex="^(priority|spot_check|batch)$")
):
    """
    Start a new review session

    Args:
        reviewer_id: ID of reviewer
        session_type: Type of review session

    Returns:
        Session ID
    """
    try:
        session_id = ReviewSessionOperations.start_session(
            reviewer_id=reviewer_id,
            session_type=session_type
        )

        return {
            "success": True,
            "session_id": session_id,
            "reviewer_id": reviewer_id,
            "session_type": session_type,
            "started_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to start review session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/{session_id}/end")
async def end_review_session(
    session_id: int,
    items_reviewed: int = Query(..., description="Number of items reviewed")
):
    """
    End a review session

    Args:
        session_id: ID of the session
        items_reviewed: Number of items reviewed in session

    Returns:
        Success confirmation
    """
    try:
        ReviewSessionOperations.end_session(
            session_id=session_id,
            items_reviewed=items_reviewed
        )

        return {
            "success": True,
            "session_id": session_id,
            "items_reviewed": items_reviewed,
            "ended_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to end review session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
