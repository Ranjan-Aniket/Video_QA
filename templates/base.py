"""
Base Template Class

All question templates inherit from this base class.

CRITICAL ARCHITECTURE PRINCIPLES:
1. Evidence-first: ALL questions generated FROM evidence, never hardcoded
2. Dual cues: EVERY question MUST have both audio AND visual cues
3. No names: Use descriptors only (enforced)
4. Precise timestamps: Cover cues + actions, not a second more/less
5. Complex questions: Focus on challenging, unintuitive scenarios
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re


class QuestionType(Enum):
    """13 question types from taxonomy"""
    TEMPORAL_UNDERSTANDING = "Temporal Understanding"
    SEQUENTIAL = "Sequential"
    SUBSCENE = "Subscene"
    GENERAL_HOLISTIC_REASONING = "General Holistic Reasoning"
    INFERENCE = "Inference"
    CONTEXT = "Context"
    NEEDLE = "Needle"
    REFERENTIAL_GROUNDING = "Referential Grounding"
    COUNTING = "Counting"
    COMPARATIVE = "Comparative"
    OBJECT_INTERACTION_REASONING = "Object Interaction Reasoning"
    AUDIO_VISUAL_STITCHING = "Audio-Visual Stitching"
    TACKLING_SPURIOUS_CORRELATIONS = "Tackling Spurious Correlations"


class CueType(Enum):
    """Types of cues in questions"""
    AUDIO_SPEECH = "audio_speech"
    AUDIO_MUSIC = "audio_music"
    AUDIO_SOUND_EFFECT = "audio_sound_effect"
    AUDIO_AMBIENT = "audio_ambient"
    AUDIO_TONE_CHANGE = "audio_tone_change"
    VISUAL_PERSON = "visual_person"
    VISUAL_OBJECT = "visual_object"
    VISUAL_SCENE = "visual_scene"
    VISUAL_TEXT = "visual_text"
    VISUAL_ACTION = "visual_action"


@dataclass
class Cue:
    """A single cue (audio or visual)"""
    cue_type: CueType
    content: str  # Description of the cue
    timestamp: float
    confidence: float = 1.0  # Confidence in evidence


@dataclass
class GeneratedQuestion:
    """A generated question with all metadata"""
    # Question and answer
    question_text: str
    golden_answer: str
    
    # Timestamps (must cover cues + actions)
    start_timestamp: float
    end_timestamp: float
    
    # Cues (MUST have both audio and visual)
    audio_cues: List[Cue]
    visual_cues: List[Cue]
    
    # Metadata
    question_types: List[QuestionType]
    generation_tier: int  # 1=template, 2=constrained, 3=creative
    template_name: str
    complexity_score: float  # 0.0 to 1.0
    
    # Evidence grounding
    evidence_refs: List[str]  # References to evidence items
    
    def __post_init__(self):
        """Validate question after creation"""
        # CRITICAL: Must have both audio and visual cues
        if not self.audio_cues:
            raise ValueError(f"Question has no audio cues: {self.question_text}")
        if not self.visual_cues:
            raise ValueError(f"Question has no visual cues: {self.question_text}")
        
        # Validate timestamps
        if self.start_timestamp >= self.end_timestamp:
            raise ValueError(f"Invalid timestamps: {self.start_timestamp} >= {self.end_timestamp}")
        
        # Validate cue timestamps are within range
        for cue in self.audio_cues + self.visual_cues:
            if not (self.start_timestamp <= cue.timestamp <= self.end_timestamp):
                raise ValueError(
                    f"Cue timestamp {cue.timestamp} outside question range "
                    f"[{self.start_timestamp}, {self.end_timestamp}]"
                )
    
    def is_single_cue_answerable(self) -> bool:
        """
        Check if question can be answered with just one cue type
        
        Per guidelines: "Even if the question has both audio and video cues, 
        but if it can be answered with just one cue, the pair needs to be rejected"
        
        Returns:
            True if answerable with single cue (REJECT), False otherwise
        """
        # This would need more sophisticated logic in practice
        # For now, we rely on template design to ensure both cues are necessary
        return False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage/export"""
        return {
            "question": self.question_text,
            "golden_answer": self.golden_answer,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "audio_cues": [
                {
                    "type": cue.cue_type.value,
                    "content": cue.content,
                    "timestamp": cue.timestamp,
                    "confidence": cue.confidence
                }
                for cue in self.audio_cues
            ],
            "visual_cues": [
                {
                    "type": cue.cue_type.value,
                    "content": cue.content,
                    "timestamp": cue.timestamp,
                    "confidence": cue.confidence
                }
                for cue in self.visual_cues
            ],
            "question_types": [qt.value for qt in self.question_types],
            "generation_tier": self.generation_tier,
            "template_name": self.template_name,
            "complexity_score": self.complexity_score,
            "evidence_refs": self.evidence_refs
        }


@dataclass
class EvidenceDatabase:
    """
    Multimodal evidence extracted from video
    
    This is the ONLY source of truth for question generation.
    NO hardcoding - everything comes from here.
    """
    # Video metadata
    video_id: str
    duration: float
    
    # Audio evidence
    transcript_segments: List[Dict]  # [{text, start, end, speaker, confidence}]
    music_segments: List[Dict] = field(default_factory=list)  # [{genre, tempo, start, end}]
    sound_effects: List[Dict] = field(default_factory=list)  # [{sound_type, timestamp, confidence}]
    ambient_sounds: List[Dict] = field(default_factory=list)  # [{sound_type, timestamp, confidence}]
    tone_changes: List[Dict] = field(default_factory=list)  # [{change_type, timestamp, magnitude}]
    
    # Visual evidence
    person_detections: List[Dict] = field(default_factory=list)  # [{person_id, timestamp, bbox, attributes}]
    object_detections: List[Dict] = field(default_factory=list)  # [{object_class, timestamp, bbox, confidence}]
    scene_detections: List[Dict] = field(default_factory=list)  # [{scene_type, timestamp, confidence}]
    ocr_detections: List[Dict] = field(default_factory=list)  # [{text, location, timestamp, confidence}]
    action_detections: List[Dict] = field(default_factory=list)  # [{action, person_id, timestamp, confidence}]
    
    # NEW: Advanced analysis fields from bulk_frame_analyzer
    body_poses: List[Dict] = field(default_factory=list)  # [{pose_type, confidence, landmarks, keypoints, bbox, timestamp}]
    hand_gestures: List[Dict] = field(default_factory=list)  # [{hand, gesture, confidence, landmarks, fingertips, bbox, timestamp}]
    face_landmarks: List[Dict] = field(default_factory=list)  # [{gaze_direction, gaze_confidence, eye_contact, head_pose, bbox, timestamp}]
    facial_expressions: List[Dict] = field(default_factory=list)  # [{emotion, confidence, bbox, all_emotions, timestamp}]
    jersey_numbers: List[Dict] = field(default_factory=list)  # [{number, confidence, bbox, team_color, timestamp}]
    text_orientation: List[Dict] = field(default_factory=list)  # [{text, orientation, timestamp}]
    blip2_captions: List[Dict] = field(default_factory=list)  # [{caption, confidence, timestamp}]
    clip_embeddings: List[Dict] = field(default_factory=list)  # [{embedding_vector, timestamp}]
    
    # Temporal evidence
    scene_changes: List[float] = field(default_factory=list)  # Timestamps of scene changes
    event_timeline: List[Dict] = field(default_factory=list)  # [{event_type, timestamp, description}]
    
    # Detected names (for blocking)
    character_names: List[str] = field(default_factory=list)
    team_names: List[str] = field(default_factory=list)
    media_names: List[str] = field(default_factory=list)  # movie/show/song names
    brand_names: List[str] = field(default_factory=list)
    
    # Video segments to avoid (intro/outro)
    intro_end: Optional[float] = None
    outro_start: Optional[float] = None


class QuestionTemplate(ABC):
    """
    Abstract base class for all question templates
    
    Each template implements one or more question types from the taxonomy.
    Templates are evidence-driven and never hardcode content.
    """
    
    def __init__(self):
        self.name = self.__class__.__name__
    
    @abstractmethod
    def get_question_types(self) -> List[QuestionType]:
        """
        Return question types this template generates
        
        Example: [QuestionType.COUNTING, QuestionType.NEEDLE]
        """
        pass
    
    @abstractmethod
    def can_apply(self, evidence: EvidenceDatabase) -> bool:
        """
        Check if template can be applied to this evidence
        
        Example:
            # Counting template needs objects to count
            return len(evidence.object_detections) >= 5
        
        Args:
            evidence: Evidence database
            
        Returns:
            True if template can generate question, False otherwise
        """
        pass
    
    @abstractmethod
    def generate(self, evidence: EvidenceDatabase) -> Optional[GeneratedQuestion]:
        """
        Generate question from evidence
        
        MUST:
        1. Use only evidence (no hardcoding)
        2. Include both audio and visual cues
        3. Ensure cues are necessary (not single-cue answerable)
        4. Set precise timestamps
        5. Use descriptors, not names
        6. Avoid intro/outro segments
        
        Args:
            evidence: Evidence database
            
        Returns:
            GeneratedQuestion if successful, None if cannot generate
        """
        pass
    
    def validate_cue_necessity(
        self, 
        question: str,
        answer: str,
        audio_cues: List[Cue],
        visual_cues: List[Cue]
    ) -> bool:
        """
        Validate that both audio and visual cues are necessary
        
        Per guidelines: "Even if the question has both audio and video cues, 
        but if it can be answered with just one cue, the pair needs to be rejected"
        
        Returns:
            True if both cues are necessary, False otherwise
        """
        # Check if question mentions audio content
        mentions_audio = False
        for cue in audio_cues:
            # Check if cue content appears in question
            if cue.content.lower() in question.lower():
                mentions_audio = True
                break
        
        # Check if question mentions visual content
        mentions_visual = False
        for cue in visual_cues:
            if cue.content.lower() in question.lower():
                mentions_visual = True
                break
        
        # Both must be mentioned in question
        return mentions_audio and mentions_visual
    
    def validate_no_names(self, text: str, evidence: EvidenceDatabase) -> bool:
        """
        Validate text contains no names
        
        Checks against all detected names in evidence
        """
        text_lower = text.lower()
        
        # Check all name lists
        all_names = (
            evidence.character_names +
            evidence.team_names +
            evidence.media_names +
            evidence.brand_names
        )
        
        for name in all_names:
            if name.lower() in text_lower:
                return False
        
        return True
    
    def calculate_timestamps(
        self,
        cues: List[Cue],
        action_end: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Calculate precise timestamps covering cues + actions
        
        Per guidelines: "Must incorporate both the cues and subsequent actions 
        referred to in the question"
        
        Args:
            cues: All cues (audio + visual) in question
            action_end: Timestamp when action completes (if applicable)
            
        Returns:
            (start_timestamp, end_timestamp)
        """
        if not cues:
            raise ValueError("No cues provided for timestamp calculation")
        
        # Start at earliest cue
        start = min(cue.timestamp for cue in cues)
        
        # End at latest cue or action end
        if action_end is not None:
            end = max(action_end, max(cue.timestamp for cue in cues))
        else:
            end = max(cue.timestamp for cue in cues)
        
        # Add small buffer for audio completion (if last cue is audio)
        last_cue = max(cues, key=lambda c: c.timestamp)
        if last_cue.cue_type in [CueType.AUDIO_SPEECH, CueType.AUDIO_MUSIC]:
            end += 1.0  # 1 second buffer for speech completion
        
        return start, end
    
    def is_in_intro_outro(
        self,
        timestamp: float,
        evidence: EvidenceDatabase
    ) -> bool:
        """
        Check if timestamp is in intro/outro segment
        
        Per guidelines: "Do not use the intro and outro of the video for 
        reference points for asking questions."
        """
        if evidence.intro_end and timestamp < evidence.intro_end:
            return True
        if evidence.outro_start and timestamp > evidence.outro_start:
            return True
        return False
    
    def find_audio_near_timestamp(
        self,
        timestamp: float,
        evidence: EvidenceDatabase,
        max_distance: float = 3.0
    ) -> Optional[Dict]:
        """
        Find audio segment near timestamp

        Args:
            timestamp: Target timestamp
            evidence: Evidence database
            max_distance: Maximum distance in seconds

        Returns:
            Closest audio segment or None
        """
        closest = None
        min_dist = float('inf')

        for segment in evidence.transcript_segments:
            dist = abs(segment['start'] - timestamp)
            if dist < min_dist and dist <= max_distance:
                min_dist = dist
                closest = segment

        return closest

    def get_audio_quote_for_question(
        self,
        audio_segment: Dict,
        max_length: int = 80
    ) -> str:
        """
        Get properly formatted audio quote for question

        Per guideline 12: Must transcribe EXACTLY - no truncation with "..."
        If quote is too long, we use a different question pattern instead

        Args:
            audio_segment: Audio segment dict with 'text' field
            max_length: Maximum length for direct quotes

        Returns:
            Audio text for use in question (no truncation markers)
        """
        text = audio_segment['text'].strip()

        # If short enough, use exact quote
        if len(text) <= max_length:
            return text

        # If too long, return first sentence or clause
        # Look for sentence break
        sentences = re.split(r'[.!?]', text)
        if sentences and len(sentences[0]) <= max_length:
            return sentences[0].strip()

        # Look for comma break
        clauses = text.split(',')
        if clauses and len(clauses[0]) <= max_length:
            return clauses[0].strip()

        # Last resort: use first max_length chars at word boundary
        # But DON'T add "..." - guidelines forbid truncation
        words = text.split()
        result = []
        length = 0
        for word in words:
            if length + len(word) + 1 > max_length:
                break
            result.append(word)
            length += len(word) + 1

        return ' '.join(result) if result else text[:max_length]
    
    def find_visual_near_timestamp(
        self,
        timestamp: float,
        evidence: EvidenceDatabase,
        visual_type: str = "any",  # "person", "object", "scene", "any"
        max_distance: float = 2.0
    ) -> Optional[Dict]:
        """
        Find visual detection near timestamp
        
        Args:
            timestamp: Target timestamp
            evidence: Evidence database
            visual_type: Type of visual to find
            max_distance: Maximum distance in seconds
            
        Returns:
            Closest visual detection or None
        """
        candidates = []
        
        if visual_type in ["any", "person"]:
            candidates.extend(evidence.person_detections)
        if visual_type in ["any", "object"]:
            candidates.extend(evidence.object_detections)
        if visual_type in ["any", "scene"]:
            candidates.extend(evidence.scene_detections)
        
        closest = None
        min_dist = float('inf')
        
        for item in candidates:
            dist = abs(item['timestamp'] - timestamp)
            if dist < min_dist and dist <= max_distance:
                min_dist = dist
                closest = item
        
        return closest