"""
Reviews API Router - Human Review Workflow Endpoints

Stage 1: Review all 30 generated questions
Stage 2: Manually select final 4 from Gemini failures

Includes WebSocket support for real-time push notifications
"""
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from database.operations import ReviewOperations, VideoOperations, QuestionOperations

# WebSocket manager - with fallback for import issues
try:
    from backend.api.websockets.manager import manager as ws_manager
except (ImportError, ModuleNotFoundError):
    # Create mock manager if import fails
    class MockWebSocketManager:
        async def notify_stage1_ready(self, *args, **kwargs):
            pass
        async def notify_stage2_ready(self, *args, **kwargs):
            pass
        async def notify_pipeline_complete(self, *args, **kwargs):
            pass
        async def notify_error(self, *args, **kwargs):
            pass
    ws_manager = MockWebSocketManager()

from utils.logger import get_logger

logger = get_logger("reviews_api")

router = APIRouter()


# ==================== REQUEST/RESPONSE MODELS ====================

class ApproveQuestionRequest(BaseModel):
    reviewer: str = Field(..., description="Email or username of reviewer")


class RejectQuestionRequest(BaseModel):
    reviewer: str = Field(..., description="Email or username of reviewer")
    feedback: str = Field(..., description="Rejection feedback explaining the issue")


class BulkReviewRequest(BaseModel):
    question_ids: List[int] = Field(..., description="List of question IDs to review")
    reviewer: str = Field(..., description="Reviewer identifier")
    feedback: Optional[str] = Field(None, description="Feedback for rejected questions")


class RegenerateQuestionRequest(BaseModel):
    reviewer: str = Field(..., description="Reviewer requesting regeneration")


class Stage2SelectionRequest(BaseModel):
    selected_question_ids: List[int] = Field(
        ...,
        description="Exactly 4 question IDs to select as final output",
        min_items=4,
        max_items=4
    )
    reviewer: str = Field(..., description="Reviewer making selection")


class QuestionResponse(BaseModel):
    id: int
    question_id: str
    question_text: str
    golden_answer: str
    generation_tier: str
    task_type: Optional[str]
    start_seconds: Optional[float]
    end_seconds: Optional[float]
    audio_cues: Optional[List]
    visual_cues: Optional[List]
    confidence_score: Optional[float]
    stage1_review_status: str
    stage1_reviewer: Optional[str]
    stage1_feedback: Optional[str]
    stage1_reviewed_at: Optional[str]
    generation_attempt: int
    is_regeneration: bool
    parent_question_id: Optional[int]
    
    class Config:
        from_attributes = True


class ProgressResponse(BaseModel):
    approved: int
    rejected: int
    pending: int
    total: int
    progress_percent: float


class FailureResponse(BaseModel):
    failure_id: str
    question_id: str
    question_text: str
    golden_answer: str
    gemini_answer: str
    failure_type: str
    failure_score: float
    severity_score: float
    clarity_score: float
    combined_score: float
    rank: int


# ==================== STAGE 1: REVIEW ENDPOINTS ====================

@router.get("/stage1/{video_id}", response_model=List[QuestionResponse])
async def get_stage1_questions(video_id: str):
    """
    Get all 30 questions for Stage 1 review
    
    Returns questions in order with their review status.
    Frontend can display all 30 at once for bulk actions.
    """
    try:
        # Get video
        video = VideoOperations.get_video(video_id)
        if not video:
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
        
        # Get all questions
        questions = ReviewOperations.get_stage1_questions(video.id)
        
        if not questions:
            raise HTTPException(
                status_code=404,
                detail=f"No questions found for video {video_id}"
            )
        
        logger.info(f"Retrieved {len(questions)} questions for Stage 1 review: {video_id}")
        
        return [QuestionResponse.from_orm(q) for q in questions]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving Stage 1 questions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stage1/approve/{question_id}")
async def approve_question(question_id: int, request: ApproveQuestionRequest):
    """
    Approve a single question
    
    Updates question status to 'approved' and sends WebSocket notification.
    """
    try:
        question = QuestionOperations.get_question_by_id(question_id)
        if not question:
            raise HTTPException(status_code=404, detail=f"Question {question_id} not found")
        
        # Approve question
        updated = ReviewOperations.approve_question(question_id, request.reviewer)
        
        # Get updated progress
        progress = ReviewOperations.get_stage1_progress(question.video_id)
        
        # Send WebSocket notification
        video = VideoOperations.get_video_by_id(question.video_id)
        await ws_manager.notify_stage1_progress(
            video.video_id,
            approved=progress['approved'],
            rejected=progress['rejected'],
            pending=progress['pending'],
            total=progress['total']
        )
        
        # Check if Stage 1 is complete
        if ReviewOperations.is_stage1_complete(question.video_id):
            await ws_manager.notify_stage1_complete(video.video_id)
        
        logger.info(f"Question {question_id} approved by {request.reviewer}")
        
        return {
            "success": True,
            "question_id": question_id,
            "status": "approved",
            "progress": progress
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error approving question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stage1/reject/{question_id}")
async def reject_question(question_id: int, request: RejectQuestionRequest):
    """
    Reject a single question with feedback
    
    Marks question as rejected. Can be regenerated up to 3 times.
    """
    try:
        question = QuestionOperations.get_question_by_id(question_id)
        if not question:
            raise HTTPException(status_code=404, detail=f"Question {question_id} not found")
        
        # Reject question
        updated = ReviewOperations.reject_question(
            question_id,
            request.reviewer,
            request.feedback
        )
        
        # Get updated progress
        progress = ReviewOperations.get_stage1_progress(question.video_id)
        
        # Send WebSocket notification
        video = VideoOperations.get_video_by_id(question.video_id)
        await ws_manager.notify_stage1_progress(
            video.video_id,
            approved=progress['approved'],
            rejected=progress['rejected'],
            pending=progress['pending'],
            total=progress['total']
        )
        
        logger.info(f"Question {question_id} rejected by {request.reviewer}")
        
        # Check if can regenerate
        can_regen, reason = ReviewOperations.can_regenerate(question_id)
        
        return {
            "success": True,
            "question_id": question_id,
            "status": "rejected",
            "feedback": request.feedback,
            "can_regenerate": can_regen,
            "regeneration_info": reason,
            "generation_attempt": updated.generation_attempt,
            "progress": progress
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rejecting question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stage1/bulk-approve")
async def bulk_approve_questions(request: BulkReviewRequest):
    """
    Approve multiple questions at once
    
    Efficient for reviewing all 30 questions displayed in UI.
    """
    try:
        count = ReviewOperations.bulk_approve(request.question_ids, request.reviewer)
        
        # Get first question to find video
        if request.question_ids:
            first_q = QuestionOperations.get_question_by_id(request.question_ids[0])
            if first_q:
                progress = ReviewOperations.get_stage1_progress(first_q.video_id)
                
                # Send WebSocket notification
                video = VideoOperations.get_video_by_id(first_q.video_id)
                await ws_manager.notify_stage1_progress(
                    video.video_id,
                    approved=progress['approved'],
                    rejected=progress['rejected'],
                    pending=progress['pending'],
                    total=progress['total']
                )
        
        logger.info(f"Bulk approved {count} questions by {request.reviewer}")
        
        return {
            "success": True,
            "approved_count": count,
            "question_ids": request.question_ids
        }
        
    except Exception as e:
        logger.error(f"Error bulk approving: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stage1/bulk-reject")
async def bulk_reject_questions(request: BulkReviewRequest):
    """Reject multiple questions at once"""
    try:
        if not request.feedback:
            raise HTTPException(status_code=400, detail="Feedback required for rejection")
        
        count = ReviewOperations.bulk_reject(
            request.question_ids,
            request.reviewer,
            request.feedback
        )
        
        logger.info(f"Bulk rejected {count} questions by {request.reviewer}")
        
        return {
            "success": True,
            "rejected_count": count,
            "question_ids": request.question_ids
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error bulk rejecting: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stage1/regenerate/{question_id}")
async def regenerate_question(question_id: int, request: RegenerateQuestionRequest):
    """
    Trigger regeneration of rejected question
    
    Creates new question attempt using same tier generator with feedback.
    Maximum 3 attempts per question.
    """
    try:
        question = QuestionOperations.get_question_by_id(question_id)
        if not question:
            raise HTTPException(status_code=404, detail=f"Question {question_id} not found")
        
        # Check if can regenerate
        can_regen, reason = ReviewOperations.can_regenerate(question_id)
        if not can_regen:
            raise HTTPException(status_code=400, detail=reason)
        
        # Import pipeline for regeneration
        from pipeline.regenerator import QuestionRegenerator
        
        regenerator = QuestionRegenerator()
        new_question = await regenerator.regenerate_question(
            question_id=question_id,
            feedback=question.stage1_feedback
        )
        
        # Send WebSocket notification
        video = VideoOperations.get_video_by_id(question.video_id)
        await ws_manager.notify_regeneration_complete(
            video.video_id,
            question.question_id,
            new_question.question_id
        )
        
        logger.info(
            f"Question {question_id} regenerated (attempt {new_question.generation_attempt}) "
            f"-> new question {new_question.id}"
        )
        
        return {
            "success": True,
            "original_question_id": question_id,
            "new_question_id": new_question.id,
            "new_question": QuestionResponse.from_orm(new_question),
            "generation_attempt": new_question.generation_attempt
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error regenerating question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stage1/progress/{video_id}", response_model=ProgressResponse)
async def get_stage1_progress(video_id: str):
    """
    Get Stage 1 review progress
    
    Returns counts of approved, rejected, pending questions.
    """
    try:
        video = VideoOperations.get_video(video_id)
        if not video:
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
        
        progress = ReviewOperations.get_stage1_progress(video.id)
        
        return ProgressResponse(**progress)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting Stage 1 progress: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==================== STAGE 2: SELECTION ENDPOINTS ====================

@router.get("/stage2/{video_id}", response_model=List[FailureResponse])
async def get_stage2_failures(video_id: str):
    """
    Get all Gemini failures ranked by score for Stage 2 selection
    
    Returns failures sorted by combined score (difficulty + diversity).
    Frontend displays as ranked list with scores visible.
    """
    try:
        video = VideoOperations.get_video(video_id)
        if not video:
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
        
        # Check pipeline stage
        if video.pipeline_stage != 'awaiting_stage2_selection':
            raise HTTPException(
                status_code=400,
                detail=f"Video not ready for Stage 2 selection (current stage: {video.pipeline_stage})"
            )
        
        # Get failures
        failures = ReviewOperations.get_stage2_failures(video.id)
        
        if not failures:
            raise HTTPException(
                status_code=404,
                detail=f"No Gemini failures found for video {video_id}"
            )
        
        # Convert to response format
        responses = []
        for idx, item in enumerate(failures, start=1):
            failure = item['failure']
            question = item['question']
            
            responses.append(FailureResponse(
                failure_id=failure.failure_id,
                question_id=question.question_id,
                question_text=question.question_text,
                golden_answer=question.golden_answer,
                gemini_answer=failure.gemini_answer,
                failure_type=failure.failure_type,
                failure_score=failure.failure_score,
                severity_score=failure.severity_score,
                clarity_score=failure.clarity_score,
                combined_score=item['combined_score'],
                rank=idx
            ))
        
        logger.info(f"Retrieved {len(responses)} Gemini failures for Stage 2 selection: {video_id}")
        
        return responses
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving Stage 2 failures: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stage2/select")
async def submit_stage2_selection(request: Stage2SelectionRequest):
    """
    Submit final 4 question selection
    
    Accepts exactly 4 question IDs selected from Gemini failures.
    Triggers Excel export and marks pipeline as complete.
    """
    try:
        # Get first question to find video
        first_q = QuestionOperations.get_question_by_id(request.selected_question_ids[0])
        if not first_q:
            raise HTTPException(status_code=404, detail="Selected questions not found")
        
        video = VideoOperations.get_video_by_id(first_q.video_id)
        
        # Validate pipeline stage
        if video.pipeline_stage != 'awaiting_stage2_selection':
            raise HTTPException(
                status_code=400,
                detail=f"Cannot submit selection at stage: {video.pipeline_stage}"
            )
        
        # Submit selection
        selected = ReviewOperations.submit_stage2_selection(
            video.id,
            request.selected_question_ids,
            request.reviewer
        )
        
        # Trigger Excel export
        from pipeline.exporter import ExcelExporter
        
        exporter = ExcelExporter()
        excel_path = await exporter.export_selected_questions(video.video_id, selected)
        
        # Update video pipeline stage
        VideoOperations.update_pipeline_stage(video.video_id, 'completed')
        VideoOperations.update_video(video.video_id, completed_at=datetime.utcnow())
        
        # Send WebSocket notification
        await ws_manager.notify_pipeline_complete(video.video_id, str(excel_path))
        
        logger.info(
            f"Stage 2 selection complete for video {video.video_id}: "
            f"4 questions selected by {request.reviewer}"
        )
        
        return {
            "success": True,
            "video_id": video.video_id,
            "selected_count": len(selected),
            "selected_question_ids": request.selected_question_ids,
            "excel_path": str(excel_path),
            "pipeline_stage": "completed"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting Stage 2 selection: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==================== STATUS & METRICS ====================

@router.get("/status/{video_id}")
async def get_review_status(video_id: str):
    """
    Get complete review status for video
    
    Returns current pipeline stage, progress, and next actions.
    """
    try:
        video = VideoOperations.get_video(video_id)
        if not video:
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
        
        status = {
            "video_id": video.video_id,
            "pipeline_stage": video.pipeline_stage,
            "status": video.status,
        }
        
        # Add stage-specific progress
        if video.pipeline_stage in ['generating', 'awaiting_stage1_review']:
            progress = ReviewOperations.get_stage1_progress(video.id)
            status['stage1_progress'] = progress
        
        elif video.pipeline_stage == 'awaiting_stage2_selection':
            failures = ReviewOperations.get_stage2_failures(video.id)
            status['stage2_failures_count'] = len(failures)
            status['stage2_selected_count'] = video.stage2_selected_count
        
        elif video.pipeline_stage == 'completed':
            selected = ReviewOperations.get_stage2_selected(video.id)
            status['final_selected'] = len(selected)
            status['completed_at'] = video.completed_at.isoformat() if video.completed_at else None
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting review status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==================== WEBSOCKET ENDPOINT ====================

@router.websocket("/ws/{video_id}")
async def websocket_endpoint(websocket: WebSocket, video_id: str):
    """
    WebSocket endpoint for real-time notifications
    
    Client connects with: ws://localhost:8000/api/reviews/ws/{video_id}
    Receives push notifications for all pipeline events.
    """
    await ws_manager.connect(websocket, video_id)
    
    try:
        while True:
            # Keep connection alive, wait for messages
            data = await websocket.receive_text()
            
            # Client can send ping to keep alive
            if data == "ping":
                await websocket.send_json({"type": "pong"})
            
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
        logger.info(f"WebSocket disconnected: {video_id}")