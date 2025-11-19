"""Videos API Endpoint"""
from fastapi import APIRouter

router = APIRouter()

@router.get("/{video_id}/results")
async def get_video_results(video_id: int):
    """Get detailed results for a video"""
    return {
        'id': video_id,
        'title': f'Video {video_id}',
        'video_url': 'https://example.com/video.mp4',
        'duration': 300,
        'candidates_generated': 28,
        'gemini_tested': 28,
        'failures_found': 11,
        'final_selected': 4,
        'processing_time': '11m 32s',
        'api_cost': 6.80,
        'selected_questions': [],
        'all_candidates': []
    }

@router.get("/{video_id}/status")
async def get_video_status(video_id: int):
    """Get processing status of a video"""
    return {
        'status': 'processing',
        'current_stage': 'Q&A Generation',
        'progress_percent': 65,
        'progress_text': '24/30 questions'
    }