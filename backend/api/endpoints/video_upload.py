"""
Video Upload API Endpoint

Handles direct video file uploads and triggers the pipeline.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import logging
from datetime import datetime
from typing import Optional
import asyncio

# Database operations
from database.operations import VideoOperations, db_manager

logger = logging.getLogger(__name__)

router = APIRouter()

# Upload directory
UPLOAD_DIR = Path(__file__).parent.parent.parent.parent / "uploads"
try:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Upload directory ensured at: {UPLOAD_DIR}")
except Exception as e:
    logger.error(f"Failed to create upload directory: {e}")
    raise

# Allowed video formats
ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB


def validate_video_file(filename: str) -> bool:
    """Validate video file extension"""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


async def run_pipeline_task(video_id: str, video_path: str):
    """
    Background task to run the pipeline
    DISABLED: Use Smart Pipeline instead (/api/smart-pipeline/run)
    """
    logger.info(f"Auto-start pipeline disabled. Use Smart Pipeline endpoint instead.")
    logger.info(f"Video uploaded: {video_id} at {video_path}")
    # Production pipeline disabled - use AdversarialSmartPipeline via /api/smart-pipeline/run


@router.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    auto_start: bool = Form(True)
):
    """
    Upload a video file and optionally start processing

    Args:
        file: Video file to upload
        title: Optional video title (defaults to filename)
        description: Optional video description
        auto_start: Whether to start processing immediately (default: True)

    Returns:
        Video metadata and upload status
    """

    # Validate file
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    if not validate_video_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    try:
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_name = Path(file.filename).stem
        extension = Path(file.filename).suffix
        unique_filename = f"{original_name}_{timestamp}{extension}"
        file_path = UPLOAD_DIR / unique_filename

        # Generate video_id
        video_id = f"video_{timestamp}_{original_name[:20]}"

        # Save file
        logger.info(f"Saving uploaded file to: {file_path}")
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size = file_path.stat().st_size

        # Check file size
        if file_size > MAX_FILE_SIZE:
            file_path.unlink()  # Delete the file
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.0f}MB"
            )

        # Create video record in database
        video_title = title or original_name
        
        # Only use valid parameters for create_video()
        video_record = VideoOperations.create_video(
            video_id=video_id,
            video_url=str(file_path),
            video_name=video_title,
            duration=None  # Will be extracted during pipeline
        )
        
        logger.info(f"Created video record: video_id={video_id}")

        # Start pipeline if auto_start is enabled
        if auto_start:
            logger.info(f"Auto-starting pipeline for video_id={video_id}")
            background_tasks.add_task(
                run_pipeline_task,
                video_id=video_id,
                video_path=str(file_path)
            )
            status = "processing"
        else:
            status = "uploaded"

        return {
            "success": True,
            "video_id": video_id,
            "filename": unique_filename,
            "file_size": file_size,
            "file_size_mb": f"{file_size / (1024*1024):.2f}",
            "title": video_title,
            "status": status,
            "message": "Video uploaded successfully" + (" and processing started" if auto_start else "")
        }

    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)

        # Clean up file if it was saved
        if 'file_path' in locals() and file_path.exists():
            file_path.unlink()

        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/start/{video_id}")
async def start_processing(video_id: str, background_tasks: BackgroundTasks):
    """
    Start processing a previously uploaded video

    Args:
        video_id: ID of the video to process

    Returns:
        Processing status
    """

    # Get video from database
    video = VideoOperations.get_video(video_id)

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    if not Path(video.video_url).exists():
        raise HTTPException(status_code=404, detail="Video file not found on disk")

    if video.pipeline_stage not in ['uploaded', 'failed']:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot start processing. Current stage: {video.pipeline_stage}"
        )

    # Start pipeline in background
    background_tasks.add_task(
        run_pipeline_task,
        video_id=video_id,
        video_path=video.video_url
    )

    return {
        "success": True,
        "video_id": video_id,
        "status": "processing",
        "message": "Processing started"
    }


@router.get("/list")
async def list_uploaded_videos(
    skip: int = 0,
    limit: int = 100,
    stage: Optional[str] = None
):
    """
    List all uploaded videos

    Args:
        skip: Number of records to skip (pagination)
        limit: Maximum number of records to return
        stage: Filter by pipeline stage (optional)

    Returns:
        List of videos
    """

    try:
        if stage:
            videos = VideoOperations.get_videos_by_pipeline_stage(stage)
        else:
            # Get all videos
            from database.schema import Video
            with db_manager.get_session() as session:
                videos = session.query(Video).offset(skip).limit(limit).all()

        return {
            "success": True,
            "count": len(videos),
            "videos": [
                {
                    "id": v.id,
                    "video_id": v.video_id,
                    "title": v.video_name,
                    "filename": Path(v.video_url).name if v.video_url else None,
                    "pipeline_stage": v.pipeline_stage,
                    "created_at": v.created_at.isoformat() if v.created_at else None,
                    "duration": v.duration,
                    # Evidence Review (HITL) fields
                    "evidence_extraction_status": v.evidence_extraction_status,
                    "ai_evidence_count": v.ai_evidence_count,
                    "evidence_needs_review_count": v.evidence_needs_review_count,
                    "evidence_approved_count": v.evidence_approved_count,
                    "evidence_accuracy_estimate": v.evidence_accuracy_estimate
                }
                for v in videos
            ]
        }

    except Exception as e:
        logger.error(f"Failed to list videos: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{video_id}")
async def delete_video(video_id: int):
    """
    Delete an uploaded video and its file

    Args:
        video_id: Database ID of the video to delete (integer)

    Returns:
        Deletion status
    """
    
    # Get video from database by ID (not video_id string)
    from database.schema import Video
    
    with db_manager.get_session() as session:
        video = session.query(Video).filter(Video.id == video_id).first()
    
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # Delete file from disk
    if video.video_url and Path(video.video_url).exists():
        try:
            Path(video.video_url).unlink()
            logger.info(f"Deleted video file: {video.video_url}")
        except Exception as e:
            logger.error(f"Failed to delete file: {e}")

    # Delete from database
    try:
        with db_manager.get_session() as session:
            session.query(Video).filter(Video.id == video_id).delete()
            session.commit()
            logger.info(f"Deleted video record: id={video_id}")
    except Exception as e:
        logger.error(f"Failed to delete from database: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete video record")

    return {
        "success": True,
        "message": "Video deleted successfully"
    }


@router.get("/status/{video_id}")
async def get_video_status(video_id: str):
    """
    Get current processing status of a video

    Args:
        video_id: ID of the video

    Returns:
        Current status and progress
    """

    video = VideoOperations.get_video(video_id)

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # Get questions count for progress
    from database.operations import QuestionOperations
    questions = QuestionOperations.get_questions_by_video(video.id)

    return {
        "success": True,
        "video_id": video_id,
        "title": video.video_name,
        "pipeline_stage": video.pipeline_stage,
        "questions_generated": len(questions) if questions else 0,
        "created_at": video.created_at.isoformat() if video.created_at else None,
        # Evidence Review (HITL) fields
        "evidence_extraction_status": video.evidence_extraction_status,
        "ai_evidence_count": video.ai_evidence_count,
        "evidence_needs_review_count": video.evidence_needs_review_count,
        "evidence_approved_count": video.evidence_approved_count,
        "evidence_accuracy_estimate": video.evidence_accuracy_estimate
    }