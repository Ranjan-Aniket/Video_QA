"""
Type definitions for Gemini QA pipeline.

Provides TypedDict definitions for commonly used data structures
to improve type safety and IDE autocomplete.
"""

from typing import TypedDict, List, Optional, Any


class FrameMetadata(TypedDict, total=False):
    """Metadata for a single video frame"""
    frame_id: str
    timestamp: float
    frame_path: str
    scene_type: str
    text_detected: str
    ocr_text: str
    objects: List[str]
    object_count: int
    pose_detected: bool
    pose_keypoints: List[Any]
    scene_score: float
    visual_complexity: float
    clip_score: float
    priority: float
    image_base64: str  # For API calls


class AudioSegment(TypedDict, total=False):
    """Audio transcript segment with timing"""
    start: float
    end: float
    text: str
    speaker: str
    confidence: float
    words: List[dict]


class AudioAnalysis(TypedDict, total=False):
    """Complete audio analysis results"""
    segments: List[AudioSegment]
    duration: float
    language: str
    has_speech: bool
    quality_score: float


class ProtectedWindow(TypedDict):
    """Temporal window protection for moments"""
    start: float
    end: float
    radius: float


class FrameExtraction(TypedDict, total=False):
    """Frame extraction metadata"""
    method: str  # "single", "burst", "window"
    frames: List[float]  # Timestamps
    anchor: float  # Primary timestamp


class Moment(TypedDict, total=False):
    """Detected moment for question generation"""
    frame_ids: List[str]
    timestamps: List[float]
    mode: str  # "precise", "micro_temporal", "inference_window", "cluster"
    duration: float
    visual_cues: List[str]
    audio_cues: List[str]
    correspondence: str
    primary_ontology: str
    secondary_ontologies: List[str]
    adversarial_features: List[str]
    priority: float
    protected_window: ProtectedWindow
    frame_extraction: FrameExtraction
    confidence: float
    validation_errors: List[str]  # For rejected moments


class Question(TypedDict, total=False):
    """Generated question with metadata"""
    question_id: str
    question: str
    golden_answer: str
    question_type: str
    audio_cue: str
    visual_cue: str
    start_timestamp: str  # MM:SS format
    end_timestamp: str  # MM:SS format
    mode: str
    frame_ids: List[str]
    confidence: float
    difficulty: str
    sub_task: str


class ValidationResult(TypedDict):
    """Result of validation check"""
    is_valid: bool
    error_message: str


class CoverageStats(TypedDict, total=False):
    """Coverage statistics for ontology types"""
    total: int
    by_type: dict[str, int]
    meets_requirements: bool
    missing_types: List[str]
