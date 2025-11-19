"""
Evidence Item - Data structure for individual evidence items

Represents a single piece of evidence extracted from video with:
- AI predictions from multiple models
- Ground truth from deterministic models
- Consensus analysis
- Review status
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from datetime import datetime
import json


@dataclass
class AIPrediction:
    """Prediction from a single AI model"""
    model_name: str  # 'gpt4', 'claude', 'yolo', 'ocr', etc.
    answer: Any
    confidence: float
    evidence: Optional[List[str]] = None
    reasoning: Optional[str] = None
    raw_response: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class GroundTruth:
    """
    Objective facts from 10 deterministic models

    This class accepts any field names to support all model outputs.
    """
    # YOLOv8x - Object Detection
    yolov8x_objects: Optional[List[Dict]] = None
    object_count: Optional[int] = None
    person_count: Optional[int] = None

    # CLIP ViT-L/14 - Clothing/Attributes
    clip_clothing: Optional[List[Dict]] = None

    # Places365-R152 - Scene Classification
    places365_scene: Optional[Dict] = None
    is_indoor: Optional[bool] = None
    is_sports_venue: Optional[bool] = None

    # PaddleOCR - Text Extraction
    paddleocr_text: Optional[List[str]] = None
    ocr_blocks: Optional[List[Dict]] = None
    text_orientation: Optional[Dict] = None

    # VideoMAE - Action Recognition
    action_recognition_note: Optional[str] = None

    # BLIP-2 - Contextual Understanding
    blip2_description: Optional[Dict] = None
    image_caption: Optional[str] = None

    # Whisper - Audio Transcription
    whisper_transcript: Optional[str] = None

    # DeepSport - Jersey Numbers
    deepsport_jerseys: Optional[List[Dict]] = None
    player_numbers: Optional[List] = None

    # FER+ - Emotions
    fer_emotions: Optional[List[Dict]] = None
    dominant_emotion: Optional[str] = None

    # Legacy / Compatibility
    legacy_scene: Optional[str] = None
    yolo_objects: Optional[List[Dict]] = None  # Legacy field name
    ocr_text: Optional[List[str]] = None  # Legacy field name
    scene_classification: Optional[str] = None  # Legacy field name

    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class ConsensusAnalysis:
    """Result of AI consensus analysis"""
    consensus_reached: bool
    consensus_level: str  # 'full', 'majority', 'none'
    consensus_answer: Optional[Any] = None
    confidence_score: float = 0.0
    needs_human_review: bool = False
    priority_level: str = 'medium'  # 'high', 'medium', 'low'
    flag_reason: Optional[str] = None
    disagreement_details: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class HumanReview:
    """Human review decision"""
    status: str = 'pending'  # 'pending', 'approved', 'rejected', 'corrected', 'skipped'
    reviewer_id: Optional[str] = None
    decision: Optional[str] = None
    corrected_answer: Optional[Any] = None
    confidence: Optional[str] = None  # 'high', 'medium', 'low'
    notes: Optional[str] = None
    review_timestamp: Optional[datetime] = None
    review_duration_seconds: float = 0.0
    ai_was_correct: Optional[bool] = None
    correction_category: Optional[str] = None

    def to_dict(self) -> Dict:
        data = asdict(self)
        if self.review_timestamp:
            data['review_timestamp'] = self.review_timestamp.isoformat()
        return data


@dataclass
class EvidenceItem:
    """
    Complete evidence item with AI predictions, ground truth, and review status

    This is the core data structure passed through the HITL pipeline.
    """
    # Identifiers
    video_id: str
    evidence_type: str  # 'ocr', 'emotion', 'object_detection', 'action', 'scene', etc.
    timestamp_start: float
    timestamp_end: float

    # AI Predictions
    gpt4_prediction: Optional[AIPrediction] = None
    claude_prediction: Optional[AIPrediction] = None
    open_model_prediction: Optional[AIPrediction] = None

    # Ground Truth
    ground_truth: Optional[GroundTruth] = None

    # Consensus Analysis
    consensus: Optional[ConsensusAnalysis] = None

    # Human Review
    human_review: HumanReview = field(default_factory=HumanReview)

    # Metadata
    evidence_id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        """Ensure defaults are set"""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

    def has_ai_predictions(self) -> bool:
        """Check if at least one AI prediction exists"""
        return any([
            self.gpt4_prediction is not None,
            self.claude_prediction is not None,
            self.open_model_prediction is not None
        ])

    def get_prediction_count(self) -> int:
        """Count how many AI predictions exist"""
        return sum([
            self.gpt4_prediction is not None,
            self.claude_prediction is not None,
            self.open_model_prediction is not None
        ])

    def needs_review(self) -> bool:
        """Check if this evidence needs human review"""
        if self.consensus:
            return self.consensus.needs_human_review
        return False

    def get_priority(self) -> str:
        """Get priority level for review queue"""
        if self.consensus:
            return self.consensus.priority_level
        return 'medium'

    def is_reviewed(self) -> bool:
        """Check if human review is complete"""
        return self.human_review.status in ['approved', 'rejected', 'corrected', 'skipped']

    def get_final_answer(self) -> Any:
        """Get the final answer (human if available, otherwise consensus)"""
        if self.is_reviewed() and self.human_review.corrected_answer:
            return self.human_review.corrected_answer
        elif self.is_reviewed() and self.human_review.status == 'approved':
            return self.consensus.consensus_answer if self.consensus else None
        elif self.consensus and self.consensus.consensus_answer:
            return self.consensus.consensus_answer
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'evidence_id': self.evidence_id,
            'video_id': self.video_id,
            'evidence_type': self.evidence_type,
            'timestamp_start': self.timestamp_start,
            'timestamp_end': self.timestamp_end,
            'gpt4_prediction': self.gpt4_prediction.to_dict() if self.gpt4_prediction else None,
            'claude_prediction': self.claude_prediction.to_dict() if self.claude_prediction else None,
            'open_model_prediction': self.open_model_prediction.to_dict() if self.open_model_prediction else None,
            'ground_truth': self.ground_truth.to_dict() if self.ground_truth else None,
            'consensus': self.consensus.to_dict() if self.consensus else None,
            'human_review': self.human_review.to_dict(),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvidenceItem':
        """Create EvidenceItem from dictionary"""
        # Parse nested objects
        gpt4_pred = AIPrediction(**data['gpt4_prediction']) if data.get('gpt4_prediction') else None
        claude_pred = AIPrediction(**data['claude_prediction']) if data.get('claude_prediction') else None
        open_pred = AIPrediction(**data['open_model_prediction']) if data.get('open_model_prediction') else None

        ground_truth = GroundTruth(**data['ground_truth']) if data.get('ground_truth') else None
        consensus = ConsensusAnalysis(**data['consensus']) if data.get('consensus') else None
        human_review = HumanReview(**data['human_review']) if data.get('human_review') else HumanReview()

        # Parse timestamps
        created_at = datetime.fromisoformat(data['created_at']) if data.get('created_at') else None
        updated_at = datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else None

        return cls(
            evidence_id=data.get('evidence_id'),
            video_id=data['video_id'],
            evidence_type=data['evidence_type'],
            timestamp_start=data['timestamp_start'],
            timestamp_end=data['timestamp_end'],
            gpt4_prediction=gpt4_pred,
            claude_prediction=claude_pred,
            open_model_prediction=open_pred,
            ground_truth=ground_truth,
            consensus=consensus,
            human_review=human_review,
            created_at=created_at,
            updated_at=updated_at
        )


# Helper functions for creating evidence items

def create_ocr_evidence(
    video_id: str,
    timestamp_start: float,
    timestamp_end: float,
    ocr_text: str,
    gpt4_text: Optional[str] = None,
    claude_text: Optional[str] = None
) -> EvidenceItem:
    """Create OCR evidence item"""
    ground_truth = GroundTruth(ocr_text=[ocr_text])

    gpt4_pred = AIPrediction(
        model_name='gpt4',
        answer=gpt4_text,
        confidence=0.9
    ) if gpt4_text else None

    claude_pred = AIPrediction(
        model_name='claude',
        answer=claude_text,
        confidence=0.9
    ) if claude_text else None

    open_pred = AIPrediction(
        model_name='paddleocr',
        answer=ocr_text,
        confidence=1.0
    )

    return EvidenceItem(
        video_id=video_id,
        evidence_type='ocr',
        timestamp_start=timestamp_start,
        timestamp_end=timestamp_end,
        gpt4_prediction=gpt4_pred,
        claude_prediction=claude_pred,
        open_model_prediction=open_pred,
        ground_truth=ground_truth
    )


def create_emotion_evidence(
    video_id: str,
    timestamp_start: float,
    timestamp_end: float,
    gpt4_emotion: str,
    claude_emotion: str,
    transcript_context: Optional[str] = None
) -> EvidenceItem:
    """Create emotion detection evidence item"""
    ground_truth = GroundTruth(
        whisper_transcript=transcript_context
    ) if transcript_context else None

    gpt4_pred = AIPrediction(
        model_name='gpt4',
        answer=gpt4_emotion,
        confidence=0.85
    )

    claude_pred = AIPrediction(
        model_name='claude',
        answer=claude_emotion,
        confidence=0.87
    )

    return EvidenceItem(
        video_id=video_id,
        evidence_type='emotion',
        timestamp_start=timestamp_start,
        timestamp_end=timestamp_end,
        gpt4_prediction=gpt4_pred,
        claude_prediction=claude_pred,
        ground_truth=ground_truth
    )


def create_object_detection_evidence(
    video_id: str,
    timestamp_start: float,
    timestamp_end: float,
    yolo_objects: List[Dict],
    gpt4_objects: Optional[List[str]] = None,
    claude_objects: Optional[List[str]] = None
) -> EvidenceItem:
    """Create object detection evidence item"""
    ground_truth = GroundTruth(yolo_objects=yolo_objects)

    gpt4_pred = AIPrediction(
        model_name='gpt4',
        answer=gpt4_objects,
        confidence=0.88
    ) if gpt4_objects else None

    claude_pred = AIPrediction(
        model_name='claude',
        answer=claude_objects,
        confidence=0.90
    ) if claude_objects else None

    open_pred = AIPrediction(
        model_name='yolo',
        answer=[obj['class'] for obj in yolo_objects],
        confidence=0.92
    )

    return EvidenceItem(
        video_id=video_id,
        evidence_type='object_detection',
        timestamp_start=timestamp_start,
        timestamp_end=timestamp_end,
        gpt4_prediction=gpt4_pred,
        claude_prediction=claude_pred,
        open_model_prediction=open_pred,
        ground_truth=ground_truth
    )
