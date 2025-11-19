"""
Adversarial Smart Pipeline Router - API endpoints for adversarial video Q&A pipeline

NEW ENDPOINTS:
- Running the complete adversarial pipeline
- Getting audio analysis (with word timestamps)
- Getting adversarial opportunities (replaces genre)
- Getting extracted frames (premium + template + bulk)
- Getting evidence items
- Getting generated questions (20 template + 7 AI + 3 cross-validated)
- Getting Gemini test results
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
import logging
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from processing.smart_pipeline import AdversarialSmartPipeline
from database.operations import db_manager
from config.settings import settings
import os

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/smart-pipeline",
    tags=["Smart Pipeline"]
)


# ============================================================================
# Request/Response Models
# ============================================================================

class SmartPipelineRequest(BaseModel):
    video_id: str
    enable_gpt4: bool = False
    enable_claude: bool = False
    enable_gemini: bool = False


class SmartPipelineStatus(BaseModel):
    video_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: float  # 0.0 to 1.0
    current_phase: str
    error: Optional[str] = None


class AudioAnalysisResponse(BaseModel):
    transcript: str
    duration: float
    segments_count: int
    speaker_count: int
    language: str


class OpportunitiesResponse(BaseModel):
    temporal_markers_count: int
    ambiguous_references_count: int
    counting_opportunities_count: int
    sequential_events_count: int
    context_rich_frames_count: int
    premium_keyframes_count: int
    premium_keyframes: List[float]
    detection_cost: float


class FrameExtractionResponse(BaseModel):
    total_frames: int
    premium_frames: int
    template_frames: int
    bulk_frames: int


class EvidenceResponse(BaseModel):
    evidence_count: int
    bulk_evidence: int
    key_moment_evidence: int


class QuestionsResponse(BaseModel):
    total_questions: int
    template_count: int
    ai_count: int
    cross_validated_count: int


class ValidationResponse(BaseModel):
    validated_questions: int
    consensus_reached: int
    needs_review: int


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/run")
async def run_smart_pipeline(
    request: SmartPipelineRequest,
    background_tasks: BackgroundTasks
):
    """
    Run the complete adversarial smart pipeline on a video

    This triggers all phases:
    1. Audio Analysis (with word timestamps)
    2. Adversarial Opportunity Mining (GPT-4)
    3. Smart Frame Extraction (premium + template + bulk)
    4. Hybrid Evidence Extraction
    5. Adversarial Question Generation (30 questions)
    6. Gemini Testing (optional)
    """
    try:
        # Get video path from database
        with db_manager.get_session() as session:
            from sqlalchemy import text
            result = session.execute(
                text("SELECT video_url FROM videos WHERE video_id = :video_id"),
                {"video_id": request.video_id}
            ).fetchone()

            if not result:
                raise HTTPException(status_code=404, detail=f"Video {request.video_id} not found")

            video_path = result[0]

        # Check if video file exists
        if not Path(video_path).exists():
            raise HTTPException(status_code=404, detail=f"Video file not found: {video_path}")

        # Run pipeline in background
        background_tasks.add_task(
            _run_pipeline_background,
            video_path=video_path,
            video_id=request.video_id,
            enable_gpt4=request.enable_gpt4,
            enable_claude=request.enable_claude,
            enable_gemini=request.enable_gemini
        )

        return {
            "status": "started",
            "video_id": request.video_id,
            "message": "Smart pipeline started in background"
        }

    except Exception as e:
        logger.error(f"Error starting smart pipeline: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _run_pipeline_background(
    video_path: str,
    video_id: str,
    enable_gpt4: bool,
    enable_claude: bool,
    enable_gemini: bool = False
):
    """Background task to run the adversarial pipeline"""
    try:
        logger.info(f"Starting adversarial pipeline for video {video_id}")

        # Get API keys from settings (which loads from .env)
        openai_api_key = settings.openai_api_key
        claude_api_key = settings.anthropic_api_key  # Note: settings uses anthropic_api_key
        gemini_api_key = settings.gemini_api_key

        logger.info(f"API Keys loaded - OpenAI: {'✓' if openai_api_key else '✗'}, Claude: {'✓' if claude_api_key else '✗'}, Gemini: {'✓' if gemini_api_key else '✗'}")

        # Create pipeline
        pipeline = AdversarialSmartPipeline(
            video_path=video_path,
            output_dir=f"outputs/{video_id}",
            openai_api_key=openai_api_key,
            claude_api_key=claude_api_key,
            gemini_api_key=gemini_api_key if enable_gemini else None
        )

        # Run pipeline
        results = pipeline.run_full_pipeline()

        logger.info(f"Adversarial pipeline completed for video {video_id}")
        logger.info(f"Total cost: ${results['total_cost']:.4f}")

    except Exception as e:
        logger.error(f"Adversarial pipeline failed for video {video_id}: {e}", exc_info=True)


@router.get("/status/{video_id}")
async def get_pipeline_status(video_id: str):
    """Get the status of smart pipeline processing for a video"""
    try:
        # NEW: Check outputs directory directly without database lookup
        # This allows status checks for videos not in the database
        output_dir = Path("outputs") / video_id

        if not output_dir.exists():
            return {
                "video_id": video_id,
                "status": "pending",
                "progress": 0.0,
                "current_phase": "Not started",
                "phases_complete": []
            }

        # Find any JSON files to determine video_stem
        json_files = list(output_dir.glob("*_audio_analysis.json"))
        if json_files:
            video_stem = json_files[0].stem.replace("_audio_analysis", "")
        else:
            # No files yet, but directory exists
            video_stem = video_id

        # Check which phases are complete (NEW ADVERSARIAL PIPELINE)
        phases_complete = []

        if (output_dir / f"{video_stem}_audio_analysis.json").exists():
            phases_complete.append("audio_analysis")

        if (output_dir / f"{video_stem}_opportunities.json").exists():
            phases_complete.append("opportunity_mining")

        if (output_dir / "frames" / video_stem / "frames_metadata.json").exists():
            phases_complete.append("frame_extraction")

        if (output_dir / f"{video_stem}_evidence.json").exists():
            phases_complete.append("evidence_extraction")

        if (output_dir / f"{video_stem}_questions.json").exists():
            phases_complete.append("question_generation")

        # Calculate progress
        total_phases = 5  # Audio, Opportunities, Frames, Evidence, Questions
        progress = len(phases_complete) / total_phases

        # Determine status
        if len(phases_complete) == 0:
            status = "pending"
            current_phase = "Not started"
        elif len(phases_complete) == total_phases:
            status = "completed"
            current_phase = "All phases complete"
        else:
            status = "processing"
            phase_names = {
                1: "Audio Analysis",
                2: "Opportunity Mining",
                3: "Frame Extraction",
                4: "Evidence Extraction",
                5: "Question Generation"
            }
            current_phase = phase_names.get(len(phases_complete) + 1, f"Phase {len(phases_complete) + 1}")

        return {
            "video_id": video_id,
            "status": status,
            "progress": progress,
            "current_phase": current_phase,
            "phases_complete": phases_complete
        }

    except Exception as e:
        logger.error(f"Error getting pipeline status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/audio/{video_id}", response_model=AudioAnalysisResponse)
async def get_audio_analysis(video_id: str):
    """Get audio analysis results for a video (with word timestamps)"""
    try:
        # NEW: Check outputs directory directly without database lookup
        output_dir = Path("outputs") / video_id

        # Find audio analysis file
        json_files = list(output_dir.glob("*_audio_analysis.json"))
        if not json_files:
            raise HTTPException(status_code=404, detail="Audio analysis not found")

        analysis_file = json_files[0]

        # Load analysis
        with open(analysis_file, 'r') as f:
            analysis = json.load(f)

        return AudioAnalysisResponse(
            transcript=analysis["transcript"][:500] + "...",  # Truncate for response
            duration=analysis["duration"],
            segments_count=len(analysis["segments"]),
            speaker_count=analysis["speaker_count"],
            language=analysis["language"]
        )

    except HTTPException:
        raise  # Re-raise HTTPException as-is (404s, etc.)
    except Exception as e:
        logger.error(f"Error getting audio analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/opportunities/{video_id}", response_model=OpportunitiesResponse)
async def get_opportunities(video_id: str):
    """Get adversarial opportunities detected for a video"""
    try:
        # NEW: Check outputs directory directly without database lookup
        output_dir = Path("outputs") / video_id

        # Find opportunities file
        json_files = list(output_dir.glob("*_opportunities.json"))
        if not json_files:
            raise HTTPException(status_code=404, detail="Adversarial opportunities not found")

        opportunities_file = json_files[0]

        # Load opportunities
        with open(opportunities_file, 'r') as f:
            opportunities = json.load(f)

        # Extract premium keyframe timestamps (already a list of floats)
        premium_keyframes = opportunities.get("premium_analysis_keyframes", [])

        return OpportunitiesResponse(
            temporal_markers_count=len(opportunities.get("temporal_markers", [])),
            ambiguous_references_count=len(opportunities.get("ambiguous_references", [])),
            counting_opportunities_count=len(opportunities.get("counting_opportunities", [])),
            sequential_events_count=len(opportunities.get("sequential_events", [])),
            context_rich_frames_count=len(opportunities.get("context_rich_frames", [])),
            premium_keyframes_count=len(premium_keyframes),
            premium_keyframes=premium_keyframes,
            detection_cost=opportunities.get("detection_cost", 0.0)
        )

    except HTTPException:
        raise  # Re-raise HTTPException as-is (404s, etc.)
    except Exception as e:
        logger.error(f"Error getting opportunities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/frames/{video_id}", response_model=FrameExtractionResponse)
async def get_frame_extraction(video_id: str):
    """Get frame extraction results for a video (premium + template + bulk)"""
    try:
        # NEW: Check outputs directory directly without database lookup
        output_dir = Path("outputs") / video_id
        frames_base = output_dir / "frames"

        # Find frames metadata file (search all subdirectories)
        metadata_files = list(frames_base.glob("*/frames_metadata.json"))
        if not metadata_files:
            raise HTTPException(status_code=404, detail="Frame extraction results not found")

        metadata_file = metadata_files[0]

        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # Count frames by type
        frames = metadata.get("frames", [])
        premium_count = sum(1 for f in frames if f.get("frame_type") == "premium")
        template_count = sum(1 for f in frames if f.get("frame_type") == "template")
        bulk_count = sum(1 for f in frames if f.get("frame_type") == "bulk")

        return FrameExtractionResponse(
            total_frames=len(frames),
            premium_frames=premium_count,
            template_frames=template_count,
            bulk_frames=bulk_count
        )

    except HTTPException:
        raise  # Re-raise HTTPException as-is (404s, etc.)
    except Exception as e:
        logger.error(f"Error getting frame extraction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/questions/{video_id}")
async def get_questions(video_id: str):
    """Get generated adversarial questions for a video (30 total: 20 template + 7 AI + 3 cross-validated)"""
    try:
        # NEW: Check outputs directory directly without database lookup
        output_dir = Path("outputs") / video_id

        # Find questions file
        json_files = list(output_dir.glob("*_questions.json"))
        if not json_files:
            raise HTTPException(status_code=404, detail="Questions not found")

        questions_file = json_files[0]

        # Load questions
        with open(questions_file, 'r') as f:
            questions_data = json.load(f)

        # Handle both old (generation_info) and new (metadata) structures
        metadata = questions_data.get("metadata") or questions_data.get("generation_info", {})

        # Return with new structure
        return {
            "total_questions": questions_data.get("total_questions", len(questions_data.get("questions", []))),
            "template_count": metadata.get("template_count", 0),
            "ai_count": metadata.get("ai_count", 0),
            "cross_validated_count": metadata.get("cross_validated_count", 0),
            "generation_cost": metadata.get("total_cost", 0.0),
            "questions": questions_data.get("questions", [])
        }

    except HTTPException:
        raise  # Re-raise HTTPException as-is (404s, etc.)
    except Exception as e:
        logger.error(f"Error getting questions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/gemini-results/{video_id}")
async def get_gemini_results(video_id: str):
    """Get Gemini testing results for a video (if available)"""
    try:
        # NEW: Check outputs directory directly without database lookup
        output_dir = Path("outputs") / video_id

        # Find Gemini results file (optional - may not exist)
        json_files = list(output_dir.glob("*_gemini_results.json"))
        if not json_files:
            return {
                "tested": False,
                "message": "Gemini testing not yet run for this video"
            }

        gemini_file = json_files[0]

        # Load Gemini results
        with open(gemini_file, 'r') as f:
            gemini_data = json.load(f)

        return gemini_data

    except Exception as e:
        logger.error(f"Error getting Gemini results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/transcript/{video_id}")
async def get_full_transcript(video_id: str):
    """Get the full transcript with word-level timestamps"""
    try:
        # NEW: Check outputs directory directly without database lookup
        output_dir = Path("outputs") / video_id

        # Find audio analysis file
        json_files = list(output_dir.glob("*_audio_analysis.json"))
        if not json_files:
            raise HTTPException(status_code=404, detail="Transcript not found")

        analysis_file = json_files[0]

        # Load analysis
        with open(analysis_file, 'r') as f:
            analysis = json.load(f)

        return {
            "transcript": analysis["transcript"],
            "segments": analysis["segments"],
            "duration": analysis["duration"],
            "speaker_count": analysis["speaker_count"],
            "has_word_timestamps": any("words" in seg for seg in analysis.get("segments", []))
        }

    except HTTPException:
        raise  # Re-raise HTTPException as-is (404s, etc.)
    except Exception as e:
        logger.error(f"Error getting transcript: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{video_id}")
async def delete_pipeline_results(video_id: str):
    """Delete all adversarial pipeline results for a video"""
    try:
        import shutil

        # NEW: Delete entire outputs/{video_id}/ directory
        output_dir = Path("outputs") / video_id
        deleted_files = []

        if output_dir.exists():
            # List all files before deletion
            for item in output_dir.rglob("*"):
                if item.is_file():
                    deleted_files.append(str(item))

            # Delete entire directory
            shutil.rmtree(output_dir)
            logger.info(f"Deleted output directory: {output_dir}")

        return {
            "status": "deleted",
            "files_deleted": len(deleted_files),
            "directory_deleted": str(output_dir),
            "files": deleted_files[:10]  # Return first 10 files for preview
        }

    except Exception as e:
        logger.error(f"Error deleting pipeline results: {e}")
        raise HTTPException(status_code=500, detail=str(e))
