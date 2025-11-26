"""
Phase 8: Direct Vision Question Generation (Optimized)

Skips Phase 7 evidence extraction. Instead, calls GPT-4o Vision API directly
on selected frames to generate questions.

Architecture:
- Filter frames (priority >= 0.75, question_types >= 2)
- Single model (GPT-4o) for all frames
- Batched specialist prompts (multiple types per call)
- Token tracking per frame and per question
- Generate 45 questions → Keep best 30
- ✅ NEW: Trust Phase 5 cluster validation
- ✅ NEW: Enforce duplicate question removal

Cost: ~$0.12-0.15 per video (vs $0.80 with Phase 7, 30% cheaper than Claude)
"""

import logging
import json
import base64
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import random
from difflib import SequenceMatcher

from openai import OpenAI
import openai

# ✅ Visual coherence validation
from PIL import Image
import imagehash

# ✅ Import temporal window calculation from centralized location
from .ontology_types import get_min_temporal_window

# ✅ Import centralized validation patterns
from .validation_patterns import HEDGING_PATTERNS, PRONOUN_PATTERNS

from .token_tracker import (
    CostTracker,
    FrameTokenStats,
    QuestionTokenStats,
    TokenUsage
)
from .question_specialists import (
    build_specialist_prompt,
    get_question_type_priority,
    SPECIALIST_PROMPTS,
    MULTI_FRAME_PROMPTS,
    detect_sub_task_types
)
try:
    from .hedging_fixer import HedgingFixer
    HEDGING_FIXER_AVAILABLE = True
except ImportError:
    HEDGING_FIXER_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("HedgingFixer not available (module not found)")

# ✅ GAP #6 FIX: Hallucination validator is now REQUIRED (not optional)
# Critical validation - fail loudly if missing instead of silently degrading quality
from .hallucination_validator import HallucinationValidator
HALLUCINATION_VALIDATOR_AVAILABLE = True  # Always True now, import will fail if missing

# ✅ Audio validators for timestamp and duplicate detection
from .audio_validators import (
    AudioTimestampValidator,
    AudioPerceptualHash,
    validate_question_audio,
    detect_duplicate_audio_segments,
    validate_audio_cue_content,  # Critical Gap #4
    check_audio_cue_quality,      # Critical Gap #4
    validate_audio_modality_diversity  # P1 Fix #3: Audio diversity validation
)

logger = logging.getLogger(__name__)


# ============================================================================
# Retry Logic for Rate Limiting
# ============================================================================

def retry_with_exponential_backoff(
    func,
    max_retries: int = 5,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0
):
    """
    Retry a function with exponential backoff for rate limiting.

    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each retry
        max_delay: Maximum delay between retries

    Returns:
        Function result

    Raises:
        Last exception if all retries fail
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries):
        try:
            return func()
        except openai.RateLimitError as e:
            last_exception = e
            if attempt == max_retries - 1:
                logger.error(f"Rate limit exceeded after {max_retries} retries")
                raise

            # Calculate delay with jitter
            jitter = random.uniform(0, delay * 0.1)
            sleep_time = min(delay + jitter, max_delay)

            logger.warning(f"Rate limited. Retrying in {sleep_time:.1f}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(sleep_time)

            delay = min(delay * backoff_factor, max_delay)
        except Exception as e:
            # Don't retry non-rate-limit errors
            raise

    # Should never reach here, but just in case
    raise last_exception


# ============================================================================
# Timestamp Conversion Helpers
# ============================================================================

def seconds_to_mmss(seconds: float) -> str:
    """Convert float seconds to MM:SS string format."""
    # ✅ FIX #7: Clamp negative timestamps to 00:00
    if seconds < 0:
        return "00:00"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def mmss_to_seconds(mmss: str) -> float:
    """Convert MM:SS string format to float seconds."""
    try:
        # ✅ FIX #7 (Part 3): Handle negative MM:SS format (e.g., "-1:59")
        is_negative = mmss.startswith('-')
        clean_mmss = mmss.lstrip('-')

        parts = clean_mmss.split(':')
        if len(parts) == 2:
            minutes, secs = parts
            seconds = int(minutes) * 60 + int(secs)
            return -seconds if is_negative else seconds
        else:
            logger.warning(f"Invalid MM:SS format: {mmss}, returning 0.0")
            return 0.0
    except Exception as e:
        logger.warning(f"Error parsing timestamp {mmss}: {e}")
        return 0.0


def enforce_temporal_window(
    start_seconds: float,
    end_seconds: float,
    question_type: str,
    video_duration: float,
    frame_timestamp: float
) -> tuple[float, float]:
    """
    Enforce minimum temporal window for question type.

    If the window is too short, expand it symmetrically around the frame timestamp
    while staying within video bounds.

    Args:
        start_seconds: Original start timestamp
        end_seconds: Original end timestamp
        question_type: The question type
        video_duration: Total video duration in seconds
        frame_timestamp: The frame's timestamp (center point)

    Returns:
        (adjusted_start, adjusted_end) in seconds
    """
    current_window = end_seconds - start_seconds
    min_window = get_min_temporal_window(question_type)

    # If window meets minimum, return as-is
    if current_window >= min_window:
        return start_seconds, end_seconds

    # Calculate needed expansion (split evenly on both sides)
    expansion_needed = min_window - current_window
    half_expansion = expansion_needed / 2.0

    # Expand symmetrically around frame timestamp
    new_start = frame_timestamp - (min_window / 2.0)
    new_end = frame_timestamp + (min_window / 2.0)

    # Clamp to video bounds
    new_start = max(0.0, new_start)
    new_end = min(video_duration, new_end)

    # If clamping made window too short, expand the other side
    actual_window = new_end - new_start
    if actual_window < min_window:
        if new_start == 0.0:
            # Hit start boundary, expand end
            new_end = min(min_window, video_duration)
        elif new_end == video_duration:
            # Hit end boundary, expand start
            new_start = max(0.0, video_duration - min_window)

    logger.debug(
        f"Temporal window adjusted for {question_type}: "
        f"{current_window:.1f}s → {new_end - new_start:.1f}s "
        f"(min: {min_window:.1f}s)"
    )

    return new_start, new_end


# ============================================================================
# Structured Output JSON Schema
# ============================================================================

# All valid question types (13 single-frame + 3 multi-frame)
VALID_QUESTION_TYPES = [
    # Single-frame types (from SPECIALIST_PROMPTS)
    "Needle",
    "Audio-Visual Stitching",
    "Temporal Understanding",
    "Sequential",
    "Subscene",
    "General Holistic Reasoning",
    "Inference",
    "Context",
    "Referential Grounding",
    "Counting",
    "Comparative",
    "Object Interaction Reasoning",
    "Tackling Spurious Correlations",
    # Multi-frame types (from MULTI_FRAME_PROMPTS)
    "Temporal Progression",
    "Sequential Action",
    "State Transformation"
]

# Sub-task types for additional categorization (from PDF guidelines)
VALID_SUB_TASK_TYPES = [
    "Human Behavior Understanding",
    "Scene Recognition",
    "OCR Recognition",
    "Causal Reasoning",
    "Intent Understanding",
    "Hallucination",
    "Multi-Detail Understanding"
]

# Strict JSON Schema for OpenAI Structured Outputs
QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "minLength": 20,
                        "maxLength": 300,
                        "description": "The question text asking about video content"
                    },
                    "question_type": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": VALID_QUESTION_TYPES
                        },
                        "minItems": 1,
                        "maxItems": 3,
                        "description": "Question type(s) from the provided list. Can be single ['Type'] or multiple ['Type1', 'Type2', 'Type3']"
                    },
                    "sub_task_type": {
                        "type": ["string", "null"],
                        "enum": VALID_SUB_TASK_TYPES + [None],
                        "description": "Optional fine-grained sub-task type. Choose from: Human Behavior Understanding, Scene Recognition, OCR Recognition, Causal Reasoning, Intent Understanding, Hallucination, Multi-Detail Understanding. Set to null if not applicable."
                    },
                    "golden_answer": {
                        "type": "string",
                        "minLength": 250,
                        "maxLength": 400,
                        "description": "Rich answer (50-80 words) with 3-part structure: (1) direct answer (10-15 words), (2) audio-visual connection (15-20 words), (3) supporting detail (15-25 words)"
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.5,
                        "maximum": 0.99,
                        "description": "Confidence score between 0.5 and 0.99"
                    },
                    "audio_cue": {
                        "type": "string",
                        "minLength": 5,
                        "description": "Audio transcript or description (no placeholders like {audio})"
                    },
                    "visual_cue": {
                        "type": "string",
                        "minLength": 20,
                        "description": "Objective visual description without interpretation"
                    },
                    "start_timestamp": {
                        "type": "string",
                        "pattern": "^\\d{2}:\\d{2}$",
                        "description": "Start time in MM:SS format"
                    },
                    "end_timestamp": {
                        "type": "string",
                        "pattern": "^\\d{2}:\\d{2}$",
                        "description": "End time in MM:SS format"
                    }
                },
                "required": [
                    "question",
                    "question_type",
                    "sub_task_type",
                    "golden_answer",
                    "confidence",
                    "audio_cue",
                    "visual_cue",
                    "start_timestamp",
                    "end_timestamp"
                ],
                "additionalProperties": False
            },
            "minItems": 1,
            "maxItems": 3,
            "description": "Array of 1-3 questions"
        }
    },
    "required": ["questions"],
    "additionalProperties": False
}


# Question type normalization mapping
QUESTION_TYPE_MAPPING = {
    # Canonical forms (as-is)
    "Needle": "Needle",
    "Audio-Visual Stitching": "Audio-Visual Stitching",
    "Temporal Understanding": "Temporal Understanding",
    "Sequential": "Sequential",
    "Subscene": "Subscene",
    "General Holistic Reasoning": "General Holistic Reasoning",
    "Inference": "Inference",
    "Context": "Context",
    "Referential Grounding": "Referential Grounding",
    "Counting": "Counting",
    "Comparative": "Comparative",
    "Object Interaction Reasoning": "Object Interaction Reasoning",
    "Tackling Spurious Correlations": "Tackling Spurious Correlations",

    # Common variations (GPT-4o often returns these)
    "NEEDLE": "Needle",
    "AUDIO-VISUAL STITCHING": "Audio-Visual Stitching",
    "TEMPORAL UNDERSTANDING": "Temporal Understanding",
    "SEQUENTIAL": "Sequential",
    "SUBSCENE": "Subscene",
    "GENERAL HOLISTIC REASONING": "General Holistic Reasoning",
    "INFERENCE": "Inference",
    "CONTEXT": "Context",
    "REFERENTIAL GROUNDING": "Referential Grounding",
    "COUNTING": "Counting",
    "COMPARATIVE": "Comparative",
    "OBJECT INTERACTION REASONING": "Object Interaction Reasoning",
    "TACKLING SPURIOUS CORRELATIONS": "Tackling Spurious Correlations",

    # Abbreviated forms
    "Audio-Visual": "Audio-Visual Stitching",
    "AUDIO-VISUAL": "Audio-Visual Stitching",
    "Temporal": "Temporal Understanding",
    "TEMPORAL": "Temporal Understanding",
    "General": "General Holistic Reasoning",
    "GENERAL": "General Holistic Reasoning",
    "Object Interaction": "Object Interaction Reasoning",
    "OBJECT INTERACTION": "Object Interaction Reasoning",
    "Spurious": "Tackling Spurious Correlations",
    "SPURIOUS": "Tackling Spurious Correlations",
}


def normalize_question_type(raw_type) -> str:
    """
    Normalize question type from GPT-4o to canonical format.

    Handles:
    - Uppercase variations (SUBSCENE → Subscene)
    - Abbreviated forms (Audio-Visual → Audio-Visual Stitching)
    - Arrays of types (["Type1", "Type2"] → "Type1; Type2")
    - Unknown types (logs warning, returns original)

    Args:
        raw_type: Either a string or a list of strings

    Returns canonical type(s) from SPECIALIST_PROMPTS keys, joined with "; " if multiple.
    """
    # Handle array/list of types (new format from JSON schema)
    if isinstance(raw_type, list):
        if not raw_type:
            logger.warning("Empty question_type array received")
            return "Unknown"

        # Normalize each type individually
        normalized_types = []
        for single_type in raw_type:
            normalized = _normalize_single_type(single_type)
            normalized_types.append(normalized)

        # ✅ Enforce Guidelines rule: Sequential + Temporal Understanding co-occurrence
        normalized_types = enforce_sequential_temporal_rule(normalized_types)

        # Join with "; " to match official format: "Temporal Understanding; Sequential; Counting"
        return "; ".join(normalized_types)

    # Handle string type (backward compatibility)
    return _normalize_single_type(raw_type)


def _normalize_single_type(raw_type: str) -> str:
    """
    Normalize a single question type string.

    Args:
        raw_type: A single question type string

    Returns:
        Normalized canonical type
    """
    # Try direct mapping first
    if raw_type in QUESTION_TYPE_MAPPING:
        return QUESTION_TYPE_MAPPING[raw_type]

    # Try case-insensitive match against canonical types
    for canonical_type in SPECIALIST_PROMPTS.keys():
        if raw_type.lower() == canonical_type.lower():
            return canonical_type

    # Unknown type - log warning and return original
    logger.warning(f"Unknown question type '{raw_type}' - using as-is. "
                  f"Expected one of: {list(SPECIALIST_PROMPTS.keys())}")
    return raw_type


def enforce_sequential_temporal_rule(question_types: List[str]) -> List[str]:
    """
    Enforce Guidelines rule: "Sequential always go hand in hand with Temporal Understanding"
    (Guidelines_ Prompt Creation.docx, line 153)

    If a question has "Sequential" type but not "Temporal Understanding",
    automatically add "Temporal Understanding" to maintain compliance.

    Args:
        question_types: List of question type strings

    Returns:
        Updated list with rule enforced
    """
    # Check if Sequential is present
    has_sequential = "Sequential" in question_types
    has_temporal = "Temporal Understanding" in question_types

    # If Sequential present but Temporal Understanding missing, add it
    if has_sequential and not has_temporal:
        logger.debug("Enforcing Sequential+Temporal rule: Adding 'Temporal Understanding'")
        # Add Temporal Understanding right after Sequential (preserves order)
        seq_index = question_types.index("Sequential")
        question_types.insert(seq_index + 1, "Temporal Understanding")

    return question_types


def calculate_question_similarity(q1: str, q2: str) -> float:
    """
    Calculate similarity score between two questions using SequenceMatcher.

    Args:
        q1: First question text
        q2: Second question text

    Returns:
        Similarity score between 0.0 (completely different) and 1.0 (identical)
    """
    # Normalize for comparison (lowercase, strip)
    q1_normalized = q1.lower().strip()
    q2_normalized = q2.lower().strip()

    # Use SequenceMatcher for similarity
    similarity = SequenceMatcher(None, q1_normalized, q2_normalized).ratio()

    return similarity


def check_duplicate_questions(questions: List[Dict], similarity_threshold: float = 0.7) -> Tuple[bool, List[str]]:
    """
    Check if any questions are too similar (likely duplicates).

    Args:
        questions: List of question dicts with 'question' field
        similarity_threshold: Similarity score above which questions are considered duplicates (0.7 = 70% similar)

    Returns:
        Tuple of (has_duplicates: bool, duplicate_warnings: List[str])
    """
    warnings = []
    has_duplicates = False

    for i in range(len(questions)):
        for j in range(i + 1, len(questions)):
            q1_text = questions[i].get('question', '')
            q2_text = questions[j].get('question', '')

            similarity = calculate_question_similarity(q1_text, q2_text)

            if similarity > similarity_threshold:
                has_duplicates = True
                warning = (f"⚠️  Duplicate detected (similarity={similarity:.2f}): "
                          f"\n   Q{i+1}: {q1_text[:80]}..."
                          f"\n   Q{j+1}: {q2_text[:80]}...")
                warnings.append(warning)
                logger.warning(warning)

    return has_duplicates, warnings


def check_hedging_language(questions: List[Dict]) -> Tuple[bool, List[str]]:
    """
    Check for prohibited hedging/weak language in questions and answers.

    Hedging language weakens answers and makes them vague. We ban:
    - "appears to", "appears"
    - "seems to", "seems"
    - "looks like"
    - "could be"
    - "may be", "might be"
    - "suggests", "suggesting"
    - "indicates", "indicating"
    - "likely", "probably", "possibly"

    Args:
        questions: List of question dicts with 'question' and 'golden_answer' fields

    Returns:
        Tuple of (has_violations: bool, violation_warnings: List[str])
    """
    # ✅ FIXED: Using centralized HEDGING_PATTERNS from validation_patterns.py

    warnings = []
    has_violations = False

    for i, q_dict in enumerate(questions):
        question = q_dict.get('question', '')
        answer = q_dict.get('golden_answer', '')
        question_id = q_dict.get('question_id', f'Q{i+1}')

        # Check question
        for pattern in HEDGING_PATTERNS:
            matches = re.finditer(pattern, question, re.IGNORECASE)
            for match in matches:
                has_violations = True
                warning = (f"⚠️  HEDGING in question {question_id}: '{match.group()}' in "
                          f"\"{question[:80]}...\"")
                warnings.append(warning)
                logger.warning(warning)

        # Check answer (more critical)
        for pattern in HEDGING_PATTERNS:
            matches = re.finditer(pattern, answer, re.IGNORECASE)
            for match in matches:
                has_violations = True
                warning = (f"⚠️  HEDGING in answer {question_id}: '{match.group()}' in "
                          f"\"{answer[:80]}...\"")
                warnings.append(warning)
                logger.warning(warning)

    return has_violations, warnings


def check_pronoun_usage(questions: List[Dict]) -> Tuple[bool, List[str]]:
    """
    Check for prohibited pronoun usage in questions and answers.

    Pronouns should be avoided as they create ambiguity. We ban:
    - "he", "she", "him", "her", "his", "hers"
    - "they", "them", "their", "theirs"

    Args:
        questions: List of question dicts with 'question' and 'golden_answer' fields

    Returns:
        Tuple of (has_violations: bool, violation_warnings: List[str])
    """
    # ✅ FIXED: Using centralized PRONOUN_PATTERNS from validation_patterns.py

    warnings = []
    has_violations = False

    for i, q_dict in enumerate(questions):
        question = q_dict.get('question', '')
        answer = q_dict.get('golden_answer', '')
        question_id = q_dict.get('question_id', f'Q{i+1}')

        # Check question
        for pattern in PRONOUN_PATTERNS:
            matches = re.finditer(pattern, question, re.IGNORECASE)
            for match in matches:
                has_violations = True
                warning = (f"⚠️  PRONOUN in question {question_id}: '{match.group()}' in "
                          f"\"{question[:80]}...\"")
                warnings.append(warning)
                logger.warning(warning)

        # Check answer (more critical)
        for pattern in PRONOUN_PATTERNS:
            matches = re.finditer(pattern, answer, re.IGNORECASE)
            for match in matches:
                has_violations = True
                warning = (f"⚠️  PRONOUN in answer {question_id}: '{match.group()}' in "
                          f"\"{answer[:80]}...\"")
                warnings.append(warning)
                logger.warning(warning)

    return has_violations, warnings


# ============================================================================
# Visual Coherence & Scene Detection Constants
# ============================================================================
# These thresholds control imagehash-based scene cut detection and static scene filtering

# Scene cut detection - Adaptive thresholds based on video motion characteristics
SCENE_CUT_THRESHOLD_DEFAULT = 20  # Default when highlights unavailable (balanced)
SCENE_CUT_THRESHOLD_HIGH_MOTION = 28  # Sports, action videos (higher = more tolerant)
SCENE_CUT_THRESHOLD_MEDIUM_MOTION = 22  # Moderate activity (balanced)
SCENE_CUT_THRESHOLD_LOW_MOTION = 15  # Interviews, static content (stricter)

# Visual activity thresholds for adaptive scene detection
VISUAL_ACTIVITY_HIGH_THRESHOLD = 0.7  # Above this = high motion (sports, action)
VISUAL_ACTIVITY_MEDIUM_THRESHOLD = 0.4  # Above this = medium motion

# Static scene detection
MIN_AVG_IMAGEHASH_DIFF = 3  # Minimum average imagehash difference to indicate meaningful visual change
                             # Below this = scene is too static (no progression)

# Cluster validation
MIN_FRAMES_FOR_START_END_CHECK = 3  # Need at least 3 frames to compare start vs end


# ============================================================================
# Frame Clustering Constants
# ============================================================================
# Control how individual frames are grouped into temporal clusters

MAX_CLUSTER_TIME_GAP_SECONDS = 10.0  # Maximum time gap between frames in same cluster
MAX_CLUSTER_SIZE = 4  # Maximum number of frames per cluster
MIN_CLUSTER_SIZE = 2  # Minimum number of frames required to form a cluster


# ============================================================================
# Audio Processing Constants
# ============================================================================
# Control audio snippet truncation and duplicate detection

AUDIO_SNIPPET_MAX_LENGTH = 60  # Truncate audio transcripts to 60 chars in prompts (save tokens)
AUDIO_DUPLICATE_SIMILARITY_THRESHOLD = 0.85  # Perceptual hash similarity for duplicate detection
                                              # (0.85 = 85% similar, higher = stricter)


# ============================================================================
# Phase 8 Configuration
# ============================================================================
# Configuration
PHASE8_CONFIG = {
    # Frame selection
    'min_frame_priority': 0.75,
    'min_question_types': 2,
    'max_frames': 40,

    # Output control
    'max_tokens_per_frame': 1800,  # ✅ FIX #8: Increased to prevent truncation in cluster questions

    # Question targets
    'questions_per_frame': {
        'high_priority': 4,   # priority >= 0.90
        'medium_priority': 3, # priority 0.75-0.89
        'low_priority': 2     # priority 0.70-0.74
    },

    # Quality control
    'target_questions': 32,  # ✅ OPTIMIZED: Target 30-35 questions (middle of range)
    'generate_buffer': 1.5,  # Generate 48 questions, keep 32
    'validation_threshold': 0.8,

    # ✅ HYBRID MODE: Spatial distribution across full video
    'cluster_only_mode': False,  # Use hybrid mode (clusters + distributed singles)
    'questions_per_cluster': 3,  # Generate only 3 BEST questions per cluster
    'target_cluster_questions': 12,  # 3 per cluster × 4 clusters = 12
    'target_single_questions': 20,  # Distributed across 6 time segments
    'num_time_segments': 6,  # Divide video into 6 segments for full coverage

    # Quality control - Duplicate detection
    'duplicate_similarity_threshold': 0.7,  # Similarity score above which questions are flagged as duplicates (0.7 = 70% similar)

    # Model - GPT-4o for all frames (30% cheaper than Claude, excellent vision)
    'model': 'gpt-4o-2024-11-20'
}


@dataclass
class GeneratedQuestion:
    """A single generated question with metadata"""
    question_id: str
    question: str
    answer: str
    question_type: str
    frame_id: str
    timestamp: float
    audio_cue: str
    visual_cue: str
    confidence: float
    evidence: str
    model: str
    tokens: Dict  # input_share, output, total
    cost: float

    # ✅ FIX #5: Add optional fields for cluster questions
    frame_sequence: Optional[List[str]] = field(default=None)  # For multi-frame questions
    start_timestamp: Optional[float] = field(default=None)
    end_timestamp: Optional[float] = field(default=None)

    # ✅ Add sub-task type (auto-detected or from LLM)
    sub_task_type: Optional[str] = field(default=None)


def build_cluster_prompt(cluster_data: Dict, config: Dict) -> str:
    """
    ✅ FIX #2: Build multi-frame prompt for temporal clusters.
    
    Args:
        cluster_data: {
            'frames': List[Dict],  # Frame metadata
            'audio_start': str,     # Audio at start
            'audio_end': str,       # Audio at end
            'start_timestamp': float,
            'end_timestamp': float
        }
        config: Phase 8 config
    
    Returns:
        Prompt string for multi-frame question generation
    """
    frames = cluster_data['frames']
    start_ts = cluster_data['start_timestamp']
    end_ts = cluster_data['end_timestamp']
    duration = end_ts - start_ts

    # ✅ HYBRID MODE: Use curated question types (3 best) if provided
    question_types = cluster_data.get('selected_question_types',
                                      frames[0].get('question_types', ['Sequential', 'Temporal Understanding']))

    # Create audio snippets for prompt (truncate long audio to save tokens)
    audio_start = cluster_data.get('audio_start', '')
    audio_end = cluster_data.get('audio_end', '')
    audio_start_snippet = (audio_start[:AUDIO_SNIPPET_MAX_LENGTH] + "...") if len(audio_start) > AUDIO_SNIPPET_MAX_LENGTH else audio_start
    audio_end_snippet = (audio_end[:AUDIO_SNIPPET_MAX_LENGTH] + "...") if len(audio_end) > AUDIO_SNIPPET_MAX_LENGTH else audio_end

    # Pre-compute timestamp strings for use in prompt examples
    start_ts_mmss = seconds_to_mmss(start_ts)
    end_ts_mmss = seconds_to_mmss(end_ts)
    
    prompt = f"""You are generating adversarial MULTI-FRAME questions about a video SEQUENCE.

═══════════════════════════════════════════════════════════════
SEQUENCE CONTEXT:
═══════════════════════════════════════════════════════════════

**Timeline:** {start_ts:.1f}s to {end_ts:.1f}s (duration: {duration:.1f}s)

**Frames Shown:** {len(frames)} frames from this sequence
- Frame 1: {frames[0]['timestamp']:.1f}s (start)
- Frame {len(frames)}: {frames[-1]['timestamp']:.1f}s (end)

**Audio Context:**
Start: "{audio_start_snippet}"
End: "{audio_end_snippet}"

═══════════════════════════════════════════════════════════════
⚠️  CRITICAL: NO HALLUCINATIONS ALLOWED ⚠️
═══════════════════════════════════════════════════════════════

Before generating ANY question, carefully observe ONLY what is actually visible/audible across these frames:

FORBIDDEN HALLUCINATIONS (these will be rejected):
✗ NO inventing interactions that don't exist
  → "gripping", "holding", "using" when objects are just positioned near each other
  → VERIFY: Is there actual physical contact? Actual hand-on-object interaction?

✗ NO inventing progression that doesn't exist
  → Don't describe motion/transformation that isn't clearly visible across frames
  → VERIFY: Can I point to the exact frames where this change occurs?

✗ NO inventing states/orientations that aren't visible
  → "overturned", "upside down", "rotated" when object is in normal position
  → VERIFY: Look at the actual orientation carefully in each frame

✗ NO inventing details not present in the frames
  → Colors, numbers, text that aren't actually visible
  → Actions, motions, expressions that aren't actually happening
  → Audio that isn't in the transcript

✗ NO assuming relationships or context beyond what's shown
  → "about to", "preparing to", "intending to" (unless motion clearly indicates)
  → VERIFY: What is ACTUALLY happening in these frames?

HALLUCINATION EXAMPLES TO AVOID:
✗ BAD: "Person grips the shark's tail and pulls it across frames"
  → If person and shark are just near each other, this is a hallucination
✓ GOOD: "Person moves from left to right while shark remains stationary"

✗ BAD: "Object rotates 180 degrees from upright to inverted"
  → If object stays in same orientation, this is a hallucination
✓ GOOD: "Object maintains upright position throughout sequence"

✗ BAD: "Person transitions from standing to preparing to jump"
  → If person just shifts weight slightly, this is a hallucination
✓ GOOD: "Person shifts weight from left foot to right foot"

VERIFICATION CHECKLIST - Before submitting each question:
☐ Is the progression I'm describing actually visible across frames? (not assumed)
☐ Are the interactions I'm describing actually happening? (not just spatial proximity)
☐ Are the details I'm mentioning actually visible/audible? (not imagined)
☐ Am I describing what I SEE happening, not what might happen next?
☐ Can I point to specific frames showing each step of the progression?

═══════════════════════════════════════════════════════════════
ASSIGNED QUESTION TYPES:
═══════════════════════════════════════════════════════════════

Generate 3 questions from these types: {', '.join(question_types)}

IMPORTANT: Generate 3 SEPARATE QUESTIONS (not 3 types on 1 question).

Each question should focus on ONE primary type, but can include up to 2 additional
types if the question naturally spans multiple categories.

Example Multi-Type Assignment:
- Primary type: "Temporal Understanding"
- Secondary type: "Audio-Visual Stitching" (if question uses audio for temporal anchor)
- Result: question_type: ["Temporal Understanding", "Audio-Visual Stitching"]

Still generate 3 distinct questions total, each focusing on different aspects.

═══════════════════════════════════════════════════════════════
MINIMUM TEMPORAL WINDOWS BY QUESTION TYPE (CRITICAL)
═══════════════════════════════════════════════════════════════

Each question type requires a MINIMUM temporal window:

SIMPLE OBSERVATION (10-15s minimum):
• Needle: 10s
• Counting: 10s
• Referential Grounding: 10s

MODERATE COMPLEXITY (20-25s minimum):
• Audio-Visual Stitching: 20s
• Context: 20s
• General Holistic Reasoning: 20s
• Inference: 20s

HIGH COMPLEXITY (30-40s minimum):
• Sequential: 30s
• Temporal Understanding: 30s
• Subscene: 30s
• Object Interaction Reasoning: 30s

VERY HIGH COMPLEXITY (40-60s minimum):
• Comparative: 40s
• Tackling Spurious Correlations: 40s

⚠️ VALIDATION: Your generated questions' start_timestamp to end_timestamp MUST span ≥ minimum for question_type.

This cluster has {duration:.1f}s available. Ensure your questions use appropriate temporal windows:
- If generating Sequential (30s min): Use at least 30s of this {duration:.1f}s cluster
- If generating Comparative (40s min): Use at least 40s of this {duration:.1f}s cluster
- If generating Audio-Visual Stitching (20s min): Use at least 20s of this {duration:.1f}s cluster

Example for Sequential question in this {duration:.1f}s cluster:
✓ VALID: start_timestamp: {start_ts:.1f}, end_timestamp: {min(end_ts, start_ts + 35):.1f}  (spans ≥30s)
✗ INVALID: start_timestamp: {start_ts:.1f}, end_timestamp: {start_ts + 15:.1f}  (spans <30s) → REJECT

═══════════════════════════════════════════════════════════════
AUDIO MODALITY REQUIREMENTS (MANDATORY FOR DIVERSITY)
═══════════════════════════════════════════════════════════════

Your 3 questions MUST use at least 2 different audio modalities from:

1. SPEECH (dialogue, narration, phrases, words)
   Example: "when narrator says 'watch this'"

2. MUSIC (tempo, tone, starts/stops)
   Example: "when tempo increases from 80 to 120 BPM"

3. SOUND EFFECTS (impacts, whooshes, clicks, mechanical)
   Example: "when impact sound occurs"

4. SILENCE (pauses, gaps, scene boundaries)
   Example: "during the 2-second silence"

✅ VALID Distribution:
- Q1: Speech ("when narrator says...")
- Q2: Music ("when tempo increases...")
- Q3: Sound effect ("when impact occurs...")
→ ACCEPTED: 3 different modalities used

❌ INVALID Distribution:
- Q1: Speech ("when person says...")
- Q2: Speech ("when narrator mentions...")
- Q3: Speech ("when dialogue includes...")
→ REJECTED: All speech-only, no diversity

MANDATORY MINIMUM: At least 1 speech + 1 non-speech question.

═══════════════════════════════════════════════════════════════
⚠️ MULTI-FRAME VALIDATION (QUESTIONS WILL BE AUTO-REJECTED IF SINGLE-FRAME)
═══════════════════════════════════════════════════════════════

EACH question will be REJECTED if answerable from a SINGLE FRAME.

VALIDATION TEST (apply to each question before finalizing):
1. Cover Frame 1 → Can you answer with remaining frames?
   • If YES → Question is multi-frame ✓
   • If NO → Question requires only Frame 1 → REJECT ✗

2. Cover Frame 2 → Can you answer with remaining frames?
   • If YES → Question is multi-frame ✓
   • If NO → Question requires only Frame 2 → REJECT ✗

3. Repeat for all frames. Only if ALL frames contribute → ACCEPT

REQUIRED MULTI-FRAME INDICATORS (use at least 2 per question):
□ "from start to end"
□ "progression of" / "sequence of"
□ "changes from X to Y"
□ "first... then... finally"
□ "before X and after Y"
□ "throughout the clip"
□ "across frames 1-N"
□ "develops over time"

EXAMPLES:
✓ MULTI-FRAME (requires all frames):
  Q: "How does the object's position change from start to end?"
  → Requires Frame 1 (start) AND Frame N (end) to answer

✗ SINGLE-FRAME (answerable from one frame):
  Q: "What color is the object?"
  → Can answer from ANY single frame → REJECT

If your question lacks multi-frame indicators → likely single-frame → REJECT and rewrite

═══════════════════════════════════════════════════════════════

**Multi-Frame Type Definitions:**

1. **Sequential**: Order of actions in sequence
   - "What is the order in which X, Y, Z occur during this sequence?"
   - "What action happens AFTER X but BEFORE Y?"

2. **Temporal Understanding**: Before/after relationships
   - "What changes between the start and end of this sequence?"
   - "What happens immediately before the final action?"

3. **Comparative**: Changes over time
   - "How does the person's position change from start to end?"
   - "Compare the background at start vs end of sequence"

4. **Subscene**: Describe continuous action
   - "Describe the complete sequence of actions shown"
   - "What progression occurs during this sequence?"

5. **Audio-Visual Stitching**: Audio-visual progression
   - "How does the audio change relate to the visual progression?"
   - "What audio cue marks the transition between actions?"

6. **Inference**: Causal reasoning across sequence
   - "Based on the sequence, why does the person perform action X?"
   - "What can be inferred about the person's intent from the progression?"

7. **Object Interaction Reasoning**: How objects are used over time
   - "How does the person's interaction with the object evolve during this sequence?"

═══════════════════════════════════════════════════════════════
CRITICAL REQUIREMENTS - EVERY QUESTION MUST FOLLOW:
═══════════════════════════════════════════════════════════════

1. SEQUENCE REQUIREMENTS:
   ✓ Questions MUST require watching the ENTIRE SEQUENCE (not single moment)
   ✓ Use temporal language: "during sequence", "from start to end", "progression"
   ✓ Reference BOTH start and end frames when possible
   ✓ Cannot be answered from any single frame alone

2. EACH QUESTION MUST:
   ✓ Require BOTH audio AND visual to answer (Rule #1 - CRITICAL!)
   ✓ Cannot be answered from audio alone OR visual alone
   ✓ Clear and unambiguous
   ✓ Specific to this sequence (not generic)

   FORBIDDEN IN QUESTIONS:
   ✗ NO pronouns: he/she/they/him/her/his/their
     → Use: "the person", "the individual", "the object"
   ✗ NO names: no proper names, team names, celebrity names, character names
     → Use: descriptors like "person on left", "individual in blue"
   ✗ NO timestamp questions: "at what time", "what timestamp"
     → Use: "when you hear X", "between audio cues X and Y"

   ADVERSARIAL REQUIREMENT (CRITICAL!):
   ✗ REJECT simple observation questions that require no reasoning
     → "What color changes?" = TOO SIMPLE, REJECT
     → "Does something move?" = TOO SIMPLE, REJECT
   ✓ Questions must require CAREFUL ATTENTION to sequence progression
   ✓ Require reasoning, integration, or precise detail tracking
   ✓ Multi-step understanding across the temporal sequence

   FORBIDDEN GENERIC PATTERNS (WORST OFFENDERS - AUTO-REJECT):
   ✗ "Describe the sequence..." = LAZY, REJECT
   ✗ "What happens during this sequence?" = TOO VAGUE, REJECT
   ✗ "What occurs between..." = TOO GENERIC, REJECT
   ✗ "Explain what is shown..." = LAZY, REJECT
   ✗ "What takes place..." = TOO VAGUE, REJECT
   ✓ Instead: Ask about SPECIFIC progression aspects
     → "How does the person's grip on the object change from start to end?"
     → "What transformation occurs in the object's orientation across frames?"
     → "How does the spatial relationship between objects evolve?"

3. EACH ANSWER MUST:
   ✓ Describe the SEQUENCE/PROGRESSION, not a single moment
   ✓ Reference BOTH audio and visual explicitly
   ✓ Rich and complete (50-80 words, 3-4 sentences)
   ✓ Descriptive and specific (not yes/no)
   ✓ Based on evidence visible + audible across frames
   ✓ Use proper grammar and capitalization
   ✓ Complete sentences ending with period
   ✓ SPECIFIC and DEFINITIVE statements only (no hedging)

   EXAMPLE ANSWER (65 words, 3-part structure):
   "The person moves from left to right across the frames while the shark remains
   stationary on the right side (direct answer). This movement occurs as the narrator
   says 'watch the approach' and background music tempo increases from 80 to 120 BPM
   (audio-visual connection). The person's posture shifts from upright to slightly
   crouched, with the distance decreasing from approximately 10 studs to 3 studs
   (supporting detail)."

   GESTURE INTERPRETATION (ALLOWED):
   When describing gestures/hand movements, you MAY use metaphorical interpretations:
   ✓ "as if counting" → ALLOWED
   ✓ "like framing a shot" → ALLOWED
   ✓ "shaped like a box" → ALLOWED
   Use reasonable interpretations based on visible hand/body positions.

   VISUAL SCOPE REQUIREMENTS (MANDATORY):
   When describing visual effects or changes, MUST specify scope:
   ✓ Use: "entire frame" OR "background only" OR "foreground only" OR "specific region: [location]"

   Examples:
   ✗ BAD: "The scene becomes grainy" (unclear scope)
   ✓ GOOD: "The entire frame, including the person and background, displays visible grain"
   ✓ GOOD: "The background becomes grainy while the person remains in clear focus"

   FORBIDDEN IN ANSWERS (WORST OFFENDERS - AUTO-REJECT):
   ✗ NO pronouns: he/she/they/him/her
     → Use: "the person", "the individual"
   ✗ NO names (real or character names)
   ✗ NO vague words: something, someone, might, maybe, possibly
   ✗ NO filler words: um, uh, like, basically, literally, kind of

   ⚠️  CRITICAL: NO HEDGING LANGUAGE (these destroy answer quality):
   ✗ "appears to" = WEAK, REJECT
   ✗ "seems to" = WEAK, REJECT
   ✗ "looks like" = WEAK, REJECT
   ✗ "could be" = WEAK, REJECT
   ✗ "may be" = WEAK, REJECT
   ✗ "suggests" = WEAK, REJECT
   ✗ "indicates" = WEAK, REJECT
   ✓ Instead: Use DEFINITIVE statements about what IS happening
     → BAD: "The person appears to move from left to right"
     → GOOD: "The person moves from left to right"
     → BAD: "The object seems to rotate"
     → GOOD: "The object rotates 90 degrees clockwise"

4. VISUAL_CUE FIELD REQUIREMENTS:
   ✓ Describe objective visual PROGRESSION across frames
   ✓ Specific frame-by-frame changes: colors, positions, orientations, actions
   ✓ Precise details (not interpretations): "object moves from left to center to right"
   ✓ Observable transformations: "posture changes from standing to crouching"
   ✓ DEFINITIVE statements only (no hedging)

   FORBIDDEN IN VISUAL_CUE (WORST OFFENDERS):
   ✗ NO interpretations: "looks angry" → use "furrowed eyebrows, gritted teeth"
   ✗ NO pronouns: "he/she" → use "the person"
   ✗ NO vague terms: "something changes"
   ✗ NO single-frame descriptions → must show progression
   ✗ NO hedging: "appears to move", "seems to rotate", "looks like it changes"
     → Use DEFINITIVE: "moves", "rotates", "changes"

   VISUAL_CUE EXAMPLES:
   ✗ BAD: "Object appears to move from left to right"
   ✓ GOOD: "Object moves from left to right across frames"
   ✗ BAD: "Person seems to change posture"
   ✓ GOOD: "Person transitions from standing to crouching position"

5. AUDIO_CUE FIELD REQUIREMENTS:
   ✓ Describe audio from transcript (no {{placeholders}})
   ✓ Quote exact phrases if speech: "'hello world' at start" → "'goodbye' at end"
   ✓ Specific sounds: "click sound" → "whoosh sound" → "thud sound"
   ✓ Audio timing relative to visual: "footsteps coincide with movement"

   FORBIDDEN IN AUDIO_CUE:
   ✗ NO placeholders: {{{{audio}}}}, {{{{audio_snippet}}}}
   ✗ NO vague descriptions: "sound occurs"
   ✗ Must be specific to actual audio heard

6. CONFIDENCE SCORE:
   ✓ Range: 0.5 to 0.99 (no 1.0 - nothing is 100% certain)
   ✓ Higher for clear, unambiguous sequences
   ✓ Lower for complex or subtle progressions

7. DIVERSITY REQUIREMENT (CRITICAL - ALL 3 QUESTIONS MUST BE DIFFERENT):
   ✗ NO duplicate questions (same question worded differently)
   ✗ NO asking about the same aspect twice
   ✗ NO repeating the same question type 3 times

   ✓ Each question must focus on DIFFERENT aspect:
     → Question 1: Object interaction progression
     → Question 2: Spatial relationship changes
     → Question 3: Audio-visual synchronization

   ✓ Use different question types (Sequential, Temporal, Comparative, etc.)
   ✓ Cover different elements (person, object, background, audio)
   ✓ Different reasoning patterns (order, transformation, comparison)

   DIVERSITY CHECK - Before submitting:
   ☐ Do all 3 questions ask about different aspects?
   ☐ Would a person need to observe different things to answer each?
   ☐ Are the questions complementary (not redundant)?

   FORBIDDEN DUPLICATES (AUTO-REJECT):
   ✗ BAD: Q1: "How does position change?" + Q2: "What movement occurs?" = SAME THING
   ✗ BAD: Q1: "How does grip evolve?" + Q2: "How does holding change?" = SAME THING
   ✗ BAD: 3 Sequential questions = NO DIVERSITY
   ✓ GOOD: Q1: Sequential (action order) + Q2: Comparative (size change) + Q3: Audio-Visual (sound timing)

**BAD (single-moment):**
❌ "What is the person wearing?"
❌ "How many objects are visible?"
❌ "What color is the background?"

**GOOD (sequence-based + adversarial):**
✅ "What sequence of actions does the person perform from start to end?"
✅ "How does the person's posture change during this sequence?"
✅ "What transformation occurs in the object's position between the start and end?"

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT:
═══════════════════════════════════════════════════════════════

Return 3 questions in strict JSON format (schema enforced).

Example:
{{
  "questions": [
    {{
      "question": "From start to end of this sequence, how does the person's interaction with the object evolve?",
      "question_type": "Sequential",
      "golden_answer": "The person first picks up the object (start), examines it while rotating (middle), then places it on the table (end).",
      "confidence": 0.92,
      "audio_cue": "Soft thud sound at end when object is placed",
      "visual_cue": "Object in person's hands at start, being rotated in middle frames, resting on surface at end",
      "start_timestamp": "{start_ts_mmss}",
      "end_timestamp": "{end_ts_mmss}"
    }},
    {{
      "question": "What changes occur in the person's posture between the beginning and end of this sequence?",
      "question_type": "Temporal Understanding",
      "golden_answer": "The person starts standing upright, transitions to leaning forward while examining the object, and ends in a seated position.",
      "confidence": 0.88,
      "audio_cue": "Chair scraping sound at the end marks the transition to seated",
      "visual_cue": "Standing posture at start, leaning forward in middle frames, seated position at end",
      "start_timestamp": "{start_ts_mmss}",
      "end_timestamp": "{end_ts_mmss}"
    }},
    {{
      "question": "How does the background scene change during this sequence?",
      "question_type": "Comparative",
      "golden_answer": "The background starts empty, a second person appears in the middle frame moving across, and the background returns to empty by the end.",
      "confidence": 0.85,
      "audio_cue": "Footsteps heard during middle portion when second person crosses",
      "visual_cue": "Empty background at start, figure passing through middle frames, empty background at end",
      "start_timestamp": "{start_ts_mmss}",
      "end_timestamp": "{end_ts_mmss}"
    }}
  ]
}}

═══════════════════════════════════════════════════════════════
FINAL VALIDATION CHECKLIST - VERIFY BEFORE SUBMITTING:
═══════════════════════════════════════════════════════════════

FOR EACH QUESTION:
☐ Requires ENTIRE SEQUENCE to answer (not answerable from single frame)?
☐ Requires BOTH audio AND visual (cannot answer with just one)?
☐ NO generic patterns ("describe", "what happens", "what occurs")?
☐ SPECIFIC aspect asked (grip, posture, position, etc.)?
☐ NO pronouns (he/she/they)?
☐ NO names (real or character)?
☐ NO "at what time" questions?

FOR EACH ANSWER:
☐ Describes PROGRESSION (start → middle → end)?
☐ References BOTH audio and visual explicitly?
☐ 50-80 words (3-4 sentences)?
☐ DEFINITIVE statements (NO "appears to", "seems to", "looks like")?
☐ NO vague words (something, might, maybe)?
☐ Complete sentences with proper grammar?

FOR DIVERSITY:
☐ All 3 questions ask about DIFFERENT aspects?
☐ No duplicate questions (same thing worded differently)?
☐ Uses different question types?

FOR VISUAL_CUE:
☐ Describes frame-by-frame PROGRESSION?
☐ DEFINITIVE statements (NO "appears to", "seems to")?
☐ NO pronouns, NO vague terms?

FOR AUDIO_CUE:
☐ Specific audio from transcript (NO placeholders)?
☐ Exact quotes or specific sounds?

Generate your questions now:"""

    return prompt


# ═══════════════════════════════════════════════════════════════════════════════
# ✅ QUICK WIN #1: System Message for Critical Rules
# ═══════════════════════════════════════════════════════════════════════════════

CRITICAL_RULES_SYSTEM_MESSAGE = """
You are a video question generation expert specializing in adversarial multimodal reasoning.

═══════════════════════════════════════════════════════════════════════════════
CRITICAL RULES (VIOLATIONS = AUTO-REJECT)
═══════════════════════════════════════════════════════════════════════════════

Your generated questions undergo AUTOMATIC VALIDATION. Questions are REJECTED if:

1. AUDIO TIMESTAMPS → Silent segments (RMS < 0.01)
2. AUDIO TIMESTAMPS → Scene cuts (energy change > 3.0)
3. TEMPORAL WINDOWS → Shorter than 10 seconds
4. AUDIO DIVERSITY → All speech-based (need 1 speech + 1 non-speech minimum)
5. ANSWER LENGTH → Under 250 characters (~50 words)
6. ANSWER LENGTH → Over 400 characters (~80 words)
7. AUDIO-VISUAL → Can be answered with single modality

═══════════════════════════════════════════════════════════════════════════════
VALIDATION EXAMPLES (Learn from these)
═══════════════════════════════════════════════════════════════════════════════

✅ VALID Audio Timestamps:
- Start: 02:08, End: 02:28 (20s window, continuous speech + music)
- Start: 01:45, End: 02:15 (30s window, dialogue with background sounds)
- Start: 00:30, End: 00:50 (20s window, sound effects + speech)

❌ INVALID Audio Timestamps (AUTO-REJECTED):
- Start: 02:08, End: 02:12 (4s window - TOO SHORT, need 10s minimum)
- Start: 05:30, End: 05:45 (silent segment detected, RMS=0.003 < 0.01)
- Start: 10:15, End: 10:35 (scene cut at 10:20, energy spike 4.2 > 3.0)

✅ VALID Audio Diversity (at least 2 modalities):
- Q1: Speech-based ("when narrator says 'watch this'...")
- Q2: Music-based ("when tempo increases to 120 BPM...")
- Q3: Sound effect-based ("when impact sound occurs...")

❌ INVALID Audio Diversity (AUTO-REJECTED):
- Q1: Speech-based ("when person says...")
- Q2: Speech-based ("when narrator mentions...")
- Q3: Speech-based ("when dialogue includes...")
→ REJECTED: All 3 questions use speech only (no diversity)

AUDIO MODALITY DISTRIBUTION (MANDATORY):
Your questions MUST use at least 2 different audio modalities from:
1. SPEECH (dialogue, narration, commentary, phrases)
2. MUSIC (tempo changes, tone shifts, music starts/stops)
3. SOUND EFFECTS (impacts, whooshes, clicks, mechanical sounds)
4. SILENCE (dramatic pauses, scene boundaries, audio gaps)

✅ VALID Answer Length (50-80 words):
"The person moves from left to right across the frames while the shark remains
stationary on the right side. This movement occurs as the narrator says 'watch
the approach' and background music tempo increases from 80 to 120 BPM. The
person's posture shifts from upright to slightly crouched, suggesting preparation
for interaction, with the distance between person and shark decreasing from
approximately 10 studs to 3 studs across the sequence." (72 words)

❌ INVALID Answer Length:
"The person moves left to right near the shark." (8 words - TOO SHORT)
"The person moves from left to right..." (150 words of excessive detail - TOO LONG)

═══════════════════════════════════════════════════════════════════════════════

If you violate these rules, the question is automatically rejected and regeneration
is required. This wastes API calls and costs money.

FOLLOW ALL RULES ON FIRST GENERATION TO AVOID REJECTION.

Your task: Generate 2-3 high-quality questions that pass all validation checks.
"""


class Phase8VisionGenerator:
    """Generate questions directly from frames using GPT-4o Vision"""

    def __init__(self, openai_api_key: str, anthropic_api_key: Optional[str] = None, config: Dict = None):
        """
        Initialize Phase 8 generator.

        Args:
            openai_api_key: OpenAI API key
            anthropic_api_key: Anthropic API key (for hedging fixer)
            config: Optional config override (defaults to PHASE8_CONFIG)
        """
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.config = config or PHASE8_CONFIG
        self.cost_tracker = CostTracker()
        self.cluster_metadata = {}  # Maps cluster index to Phase 5 metadata

        # ✅ NEW: Audio path for validators (set in generate_questions)
        self.audio_path = None

        # ✅ NEW: Track duplicate removal stats
        self.duplicate_stats = {
            'rejected_invalid_audio': 0,
            'audio_duplicates_removed': 0
        }

        # Initialize hedging fixer if anthropic key provided
        self.hedging_fixer = None
        if anthropic_api_key and HEDGING_FIXER_AVAILABLE:
            try:
                self.hedging_fixer = HedgingFixer(anthropic_api_key=anthropic_api_key)
                logger.info("Hedging fixer initialized (will auto-fix weak language)")
            except Exception as e:
                logger.warning(f"Could not initialize hedging fixer: {e}")
        else:
            logger.info("No Anthropic API key - hedging detection only (no auto-fix)")

        # Initialize hallucination validator
        self.hallucination_validator = None
        if HALLUCINATION_VALIDATOR_AVAILABLE:
            try:
                self.hallucination_validator = HallucinationValidator(openai_api_key=openai_api_key)
                logger.info("✅ Hallucination validator initialized (Critical Gap #3)")
            except Exception as e:
                logger.warning(f"Could not initialize hallucination validator: {e}")

        # ✅ FIX BUG-002: Initialize video_duration (set properly in generate_questions)
        self.video_duration = 0.0

        # ✅ FIX BUG-003: Initialize scene_cut_threshold (used in cluster visual coherence validation)
        self.scene_cut_threshold = SCENE_CUT_THRESHOLD_DEFAULT

    def _select_best_cluster_question_types(self, cluster_metadata: Optional[Dict]) -> List[str]:
        """
        Select question types for cluster based on Pass 2A/2B ontology assignment.

        ROUTING LOGIC (Strict Ontology Binding):
        1. PRIMARY ontology MUST be first (from Pass 2A/2B assignment)
        2. SECONDARY ontologies fill remaining slots (priority order)
        3. Fallback to Temporal Understanding only if NO assignment

        Args:
            cluster_metadata: Validated moment metadata with question_types from Pass 2A/2B
                             Format: question_types = [primary_ontology, ...secondary_ontologies]
                             Can be None if metadata is missing or invalid.

        Returns:
            List of exactly 3 question type strings (bound to assigned ontologies)
        """
        # ✅ FIXED: Handle None cluster_metadata to prevent silent failures
        if cluster_metadata is None:
            logger.warning("⚠️  cluster_metadata is None, using empty dict")
            cluster_metadata = {}

        suggested_types = cluster_metadata.get('question_types', [])

        # ✅ CRITICAL: Primary ontology is FIRST in suggested_types
        # This comes from validated_moments: [primary_ontology, ...secondary_ontologies]
        if not suggested_types:
            logger.warning("⚠️  No ontology assigned by Pass 2A/2B, defaulting to Temporal Understanding")
            return ['Temporal Understanding', 'Sequential', 'Comparative']

        selected = []

        # ✅ STEP 1: PRIMARY ONTOLOGY (MANDATORY - from Pass 2A/2B)
        primary_ontology = suggested_types[0]  # First item is primary
        selected.append(primary_ontology)
        logger.debug(f"      Primary ontology: {primary_ontology}")

        # ✅ STEP 2: SECONDARY ONTOLOGIES (from Pass 2A/2B if available)
        for secondary in suggested_types[1:]:
            if len(selected) >= 3:
                break
            if secondary not in selected:
                selected.append(secondary)
                logger.debug(f"      Secondary ontology: {secondary}")

        # ✅ STEP 3: Fill remaining slots with compatible cluster types if needed
        if len(selected) < 3:
            # Compatible multi-frame types (avoid redundancy with primary)
            CLUSTER_TYPE_PRIORITY = [
                'Temporal Understanding',      # Core multi-frame type
                'Sequential',                  # Distinct from Temporal (order)
                'Comparative',                 # Frame-to-frame comparison
                'Audio-Visual Stitching',      # A-V progression
                'Object Interaction Reasoning',# Action sequences
                'Subscene',                    # Sub-scene reasoning
                'Inference',                   # Causal reasoning
            ]

            for qtype in CLUSTER_TYPE_PRIORITY:
                if len(selected) >= 3:
                    break
                if qtype not in selected:
                    # Skip Subscene if we already have Temporal or Sequential (redundant)
                    if qtype == 'Subscene' and ('Temporal Understanding' in selected or 'Sequential' in selected):
                        continue

                selected.append(qtype)
                if len(selected) >= 3:
                    break

        # Fallback 1: If we don't have 3 yet, add remaining suggested types (excluding Subscene if redundant)
        if len(selected) < 3:
            for qtype in suggested_types:
                if qtype not in selected:
                    # Still try to avoid Subscene if we have Temporal or Sequential
                    if qtype == 'Subscene' and ('Temporal Understanding' in selected or 'Sequential' in selected):
                        continue
                    selected.append(qtype)
                    if len(selected) >= 3:
                        break

        # Fallback 2: If STILL don't have 3, allow Subscene as last resort
        if len(selected) < 3 and 'Subscene' in suggested_types and 'Subscene' not in selected:
            selected.append('Subscene')

        # Final fallback: Use defaults if still not enough (shouldn't happen with Phase 5 metadata)
        defaults = ['Temporal Understanding', 'Sequential', 'Comparative']
        while len(selected) < 3:
            for qtype in defaults:
                if qtype not in selected:
                    selected.append(qtype)
                    break

        logger.debug(f"Selected {len(selected)} cluster question types: {selected}")
        return selected[:3]  # Ensure exactly 3

    def _validate_question_ontology_match(
        self,
        question: 'GeneratedQuestion',
        assigned_ontologies: List[str],
        moment_mode: str = "cluster"
    ) -> tuple[bool, str]:
        """
        Validate that generated question matches assigned ontology from Pass 2A/2B.

        CRITICAL ROUTING VALIDATION:
        - Ensures Phase 8 respects Pass 2A/2B ontology assignments
        - Prevents mismatched questions (e.g., Temporal when AVStitching was assigned)
        - Enforces temporal window requirements per question type

        Args:
            question: Generated question object
            assigned_ontologies: List of ontologies assigned by Pass 2A/2B [primary, ...secondary]
            moment_mode: Mode from validated moment ("cluster", "precise", "micro_temporal", "inference")

        Returns:
            (is_valid, rejection_reason)
        """
        q_type = question.question_type

        # Validation 1: Question type must be in assigned ontologies
        if q_type not in assigned_ontologies:
            return False, f"Question type '{q_type}' not in assigned ontologies {assigned_ontologies}"

        # Validation 2: Temporal window must meet minimum for question type
        from processing.ontology_types import get_min_temporal_window
        min_window = get_min_temporal_window(q_type)
        actual_window = question.end_timestamp - question.start_timestamp

        if actual_window < min_window:
            return False, f"{q_type} requires {min_window}s window, got {actual_window:.1f}s"

        # ✅ GAP #10 FIX: Enhanced multi-frame validation for cluster questions
        # Validates that cluster questions actually require multiple frames (not single-frame answerable)
        CLUSTER_TYPES = [
            "Sequential", "Comparative", "Temporal Understanding",
            "Subscene", "Object Interaction Reasoning",
            "Audio-Visual Stitching", "Tackling Spurious Correlations"  # ✅ ADDED missing cluster types
        ]

        if q_type in CLUSTER_TYPES and moment_mode == "cluster":
            # ✅ ENHANCED: Check both question AND answer for multi-frame indicators
            multi_frame_keywords = [
                # Temporal progression
                "sequence", "progression", "before", "after", "then", "next",
                "first", "second", "third", "last", "final", "initially", "eventually",
                # State changes
                "changes", "transforms", "develops", "transitions", "shifts", "evolves",
                "becomes", "turns into", "grows", "decreases", "increases",
                # Causal/sequential
                "leads to", "results in", "causes", "because", "followed by",
                "following", "subsequently", "as a result", "consequently",
                # Comparative
                "compared to", "different from", "in contrast", "whereas",
                "earlier", "later", "previously", "afterwards",
                # Multi-step
                "steps", "stages", "phases", "process", "over time", "throughout",
                # Counting/progression
                "how many times", "repeatedly", "multiple", "several occasions"
            ]

            # Check both question and answer
            q_text = question.question.lower()
            a_text = question.golden_answer.lower()
            combined_text = q_text + " " + a_text

            has_multi_frame_indicator = any(kw in combined_text for kw in multi_frame_keywords)

            if not has_multi_frame_indicator:
                return False, f"{q_type} cluster question lacks multi-frame indicators (may be single-frame answerable)"

        # Validation 4: Primary ontology should be prioritized
        # Warn (but don't reject) if question doesn't match primary
        primary_ontology = assigned_ontologies[0] if assigned_ontologies else None
        if primary_ontology and q_type != primary_ontology:
            logger.debug(f"      ⚠️  Question type '{q_type}' != primary '{primary_ontology}' (using secondary)")

        return True, ""

    def _create_time_segments(self, video_duration: float, num_segments: int) -> List[Dict]:
        """
        Divide video into time segments for spatial distribution.

        Args:
            video_duration: Total video duration in seconds
            num_segments: Number of segments to create (default: 6)

        Returns:
            List of segment dicts with {start, end, duration}
        """
        segment_duration = video_duration / num_segments
        segments = []

        for i in range(num_segments):
            start = i * segment_duration
            end = (i + 1) * segment_duration if i < num_segments - 1 else video_duration

            segments.append({
                'segment_id': i,
                'start': start,
                'end': end,
                'duration': end - start,
                'has_cluster': False,  # Will be updated
                'cluster_count': 0,
                'target_singles': 0  # Will be calculated
            })

        return segments

    def _assign_frames_to_segments(
        self,
        frames: List[Dict],
        clusters: List[List[Dict]],
        segments: List[Dict]
    ) -> Dict:
        """
        Assign frames and clusters to time segments for distributed selection.

        Args:
            frames: All available frames from Phase 5
            clusters: List of cluster frame lists
            segments: Time segments created by _create_time_segments

        Returns:
            Dict with segment assignments and target questions per segment
        """
        # Track which frames are in clusters (by frame_id or timestamp)
        frames_in_clusters = set()
        for cluster in clusters:
            for frame in cluster:
                # Use frame_id if available (Phase 6+), otherwise timestamp (Phase 5)
                frame_key = frame.get('frame_id', frame.get('timestamp'))
                frames_in_clusters.add(frame_key)

        # Assign clusters to segments
        for cluster in clusters:
            cluster_start = cluster[0]['timestamp']
            cluster_end = cluster[-1]['timestamp']
            cluster_midpoint = (cluster_start + cluster_end) / 2

            for segment in segments:
                if segment['start'] <= cluster_midpoint < segment['end']:
                    segment['has_cluster'] = True
                    segment['cluster_count'] += 1
                    break

        # Assign frames to segments
        for segment in segments:
            segment['frames'] = []
            segment['single_frames'] = []

        for frame in frames:
            timestamp = frame['timestamp']

            for segment in segments:
                if segment['start'] <= timestamp < segment['end']:
                    segment['frames'].append(frame)
                    # Check if frame is in a cluster (by frame_id or timestamp)
                    frame_key = frame.get('frame_id', frame.get('timestamp'))
                    if frame_key not in frames_in_clusters:
                        segment['single_frames'].append(frame)
                    break

        # Calculate target singles per segment based on cluster presence
        target_single_questions = self.config['target_single_questions']
        segments_without_clusters = sum(1 for s in segments if not s['has_cluster'])
        segments_with_clusters = sum(1 for s in segments if s['has_cluster'])

        if segments_without_clusters > 0:
            # Segments without clusters get more singles
            singles_for_empty = int(target_single_questions * 0.6 / segments_without_clusters) if segments_without_clusters else 0
            singles_for_filled = int(target_single_questions * 0.4 / segments_with_clusters) if segments_with_clusters else 0

            for segment in segments:
                if segment['has_cluster']:
                    segment['target_singles'] = singles_for_filled
                else:
                    segment['target_singles'] = singles_for_empty
        else:
            # All segments have clusters, distribute evenly
            singles_per_segment = target_single_questions // len(segments)
            for segment in segments:
                segment['target_singles'] = singles_per_segment

        return {
            'segments': segments,
            'total_single_frames': sum(len(s['single_frames']) for s in segments),
            'frames_in_clusters': len(frames_in_clusters)
        }

    def _calculate_adaptive_threshold(self, highlights: Optional[List[Dict]] = None) -> int:
        """
        Calculate scene cut threshold based on video motion characteristics.

        Uses Phase 3 highlights to determine average visual activity level.
        Higher visual activity (sports, action) = higher threshold (fewer false positives)
        Lower visual activity (interview, static) = lower threshold (stricter validation)

        Args:
            highlights: Optional Phase 3 highlights containing visual_score

        Returns:
            int: Threshold value for imagehash difference (15-28)
        """
        if not highlights or len(highlights) == 0:
            logger.info(f"No highlights provided, using default threshold: {SCENE_CUT_THRESHOLD_DEFAULT}")
            return SCENE_CUT_THRESHOLD_DEFAULT

        # Calculate average visual activity from Phase 3 highlights
        visual_scores = [h.get('visual_score', 0.5) for h in highlights]
        avg_visual_activity = sum(visual_scores) / len(visual_scores)

        # Adaptive threshold based on visual activity
        if avg_visual_activity > VISUAL_ACTIVITY_HIGH_THRESHOLD:  # High motion (sports, action)
            threshold = SCENE_CUT_THRESHOLD_HIGH_MOTION
            motion_level = "HIGH"
        elif avg_visual_activity > VISUAL_ACTIVITY_MEDIUM_THRESHOLD:  # Medium motion
            threshold = SCENE_CUT_THRESHOLD_MEDIUM_MOTION
            motion_level = "MEDIUM"
        else:  # Low motion (interview, static)
            threshold = SCENE_CUT_THRESHOLD_LOW_MOTION
            motion_level = "LOW"

        logger.info(f"🎯 Adaptive Threshold Calculation:")
        logger.info(f"   Visual Activity Level: {avg_visual_activity:.3f} ({motion_level})")
        logger.info(f"   Scene Cut Threshold: {threshold}")
        logger.info(f"   Rationale: {'High motion videos need higher thresholds to avoid false positives' if motion_level == 'HIGH' else 'Medium motion uses balanced threshold' if motion_level == 'MEDIUM' else 'Low motion uses strict threshold'}")

        return threshold

    def generate_questions(
        self,
        phase5_output: Dict,
        audio_analysis: Dict,
        frames_dir: Path,
        video_id: str,
        highlights: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Main entry point: Generate questions from Phase 5 output.

        Args:
            phase5_output: Full Phase 5 output containing:
                - 'selection_plan': List of selected frames
                - 'dense_clusters': List of Claude-validated cluster metadata
                - 'coverage': Question type coverage stats
            audio_analysis: Audio transcript from Phase 1
            frames_dir: Directory containing extracted frame images
            video_id: Video identifier
            highlights: Optional Phase 3 highlights for adaptive threshold calculation

        Returns:
            {
                'questions': List[GeneratedQuestion],
                'cost_summary': Dict,
                'frame_stats': List[FrameTokenStats],
                'question_stats': List[QuestionTokenStats]
            }
        """
        logger.info("=" * 80)
        logger.info("PHASE 8: DIRECT VISION QUESTION GENERATION")
        logger.info("=" * 80)

        # ✅ NEW: Setup audio path for validators
        # Try multiple possible audio file locations
        possible_audio_paths = [
            frames_dir.parent / f"{video_id}_audio.wav",  # outputs/video_X_audio.wav
            frames_dir.parent.parent / f"{video_id}_audio.wav",  # outputs_dir/video_X_audio.wav
            Path(f"uploads/{video_id}.mp3"),  # Original audio
        ]

        self.audio_path = None
        for audio_path in possible_audio_paths:
            if audio_path.exists():
                self.audio_path = audio_path
                logger.info(f"✓ Audio file found: {self.audio_path}")
                break

        if not self.audio_path:
            logger.warning(f"⚠️  Audio file not found in:")
            for p in possible_audio_paths:
                logger.warning(f"   - {p}")
            logger.warning("   Audio validation will be skipped")

        # Calculate adaptive threshold based on video characteristics
        self.scene_cut_threshold = self._calculate_adaptive_threshold(highlights)

        # Extract frames and clusters from Phase 5 output
        phase5_frames = phase5_output.get('selection_plan', [])
        phase5_clusters_metadata = phase5_output.get('dense_clusters', [])

        logger.info(f"\n📦 Phase 5 Output:")
        logger.info(f"   Frames: {len(phase5_frames)}")
        logger.info(f"   Validated clusters: {len(phase5_clusters_metadata)}")

        # Step 0: Map timestamps to frame_ids from frames_metadata.json
        logger.info("\n📋 Step 0: Loading frame metadata and mapping frame_ids...")
        phase5_frames = self._add_frame_ids(phase5_frames, frames_dir, video_id)
        logger.info(f"   Mapped {len(phase5_frames)} frames to frame_ids")

        # Step 1: Filter and rank frames
        logger.info("\n📊 Step 1: Filtering and ranking frames...")
        filtered_frames = self._filter_frames(phase5_frames)
        logger.info(f"   Filtered: {len(phase5_frames)} → {len(filtered_frames)} frames")
        logger.info(f"   (priority >= {self.config['min_frame_priority']}, types >= {self.config['min_question_types']})")

        # Step 2: Select top frames
        top_frames = self._select_top_frames(filtered_frames)
        logger.info(f"\n🎯 Step 2: Selected top {len(top_frames)} frames for processing")

        # Inform user if we only have clusters (no individual frames)
        if len(top_frames) == 0 and len(phase5_clusters_metadata) > 0:
            logger.warning(f"⚠️  No individual frames selected, but {len(phase5_clusters_metadata)} clusters available")
            logger.info("   Will generate questions from clusters only")

        # Check if we have any frames OR clusters to process
        if len(top_frames) == 0 and len(phase5_clusters_metadata) == 0:
            logger.error("❌ No frames or clusters selected for processing!")
            logger.error("   This usually means:")
            logger.error("   1. Phase 5 output format is incorrect")
            logger.error("   2. Filtering criteria too strict (try lowering min_frame_priority)")
            logger.error("   3. No frames have question_types assigned")

            # Return empty result
            return {
                'questions': [],
                'cost_summary': self.cost_tracker.get_summary(),
                'frame_stats': [],
                'question_stats': [],
                'metadata': {
                    'video_id': video_id,
                    'frames_processed': 0,
                    'frames_filtered_from': len(phase5_frames),
                    'model_used': self.config['model'],
                    'error': 'No frames selected for processing'
                }
            }

        # Step 3: Assign model (GPT-4o for all frames)
        frame_models = self._assign_models(top_frames)
        model = self.config['model']
        logger.info(f"\n🤖 Step 3: Model assignment")
        logger.info(f"   Using {model} for all {len(frame_models)} frames")

        # Step 4: Hybrid question generation with spatial distribution
        logger.info(f"\n🎬 Step 4: Hybrid question generation (spatial distribution across full video)")
        logger.info(f"   Strategy: 3 best questions per cluster + distributed singles across 6 time segments")

        all_questions = []

        # Step 4a: Get video duration and create time segments
        video_duration = audio_analysis.get('duration', 0)
        if video_duration == 0:
            logger.warning("Video duration not found in audio_analysis, estimating from frames...")
            video_duration = max(f['timestamp'] for f in phase5_frames) if phase5_frames else 600

        # Store video duration for use in temporal window enforcement
        self.video_duration = video_duration

        logger.info(f"\n📊 Video Analysis:")
        logger.info(f"   Duration: {video_duration:.1f}s ({video_duration/60:.1f} minutes)")

        num_segments = self.config.get('num_time_segments', 6)
        segments = self._create_time_segments(video_duration, num_segments)
        logger.info(f"   Created {len(segments)} time segments (~{segments[0]['duration']:.1f}s each)")

        # Step 4b: Use Phase 5's Claude-validated clusters
        if phase5_clusters_metadata:
            clusters, cluster_metadata_dict = self._build_clusters_from_phase5(phase5_clusters_metadata, frames_dir, video_id)
            self.cluster_metadata = cluster_metadata_dict  # Store for use in _generate_cluster_questions
            logger.info(f"   Using {len(clusters)} Claude-validated clusters from Phase 5")
        else:
            # Fallback: naive detection if Phase 5 didn't provide clusters
            clusters = self._identify_clusters(top_frames)
            self.cluster_metadata = {}  # No metadata for naive clusters
            logger.warning(f"   ⚠️  Phase 5 clusters not found, falling back to naive detection")
            logger.info(f"   Identified {len(clusters)} clusters")

        # Step 4c: Assign frames and clusters to segments
        segment_data = self._assign_frames_to_segments(phase5_frames, clusters, segments)
        segments = segment_data['segments']

        logger.info(f"\n🗺️  Spatial Distribution:")
        for seg in segments:
            cluster_marker = "■" * seg['cluster_count'] if seg['has_cluster'] else ""
            single_marker = "○" * seg['target_singles']
            logger.info(f"   Segment {seg['segment_id']} ({seg['start']:.0f}-{seg['end']:.0f}s): {cluster_marker} {single_marker} ({seg['cluster_count']} clusters, {seg['target_singles']} singles)")

        # Step 4d: Generate multi-frame questions from ALL clusters (3 per cluster)
        logger.info(f"\n   Stage 1: Generating cluster questions (3 best per cluster)...")
        cluster_questions = []

        num_clusters_to_process = len(clusters)  # Process ALL clusters
        # ✅ INDEXING STRATEGY: Loop uses 1-based indexing for user-friendly logging (Cluster 1, 2, 3...)
        # cluster_metadata dict uses 0-based keys (0, 1, 2...) matching list indices
        # Methods receive 1-based cluster_index and convert to 0-based when accessing dict
        for i, cluster in enumerate(clusters, 1):
            cluster_duration = cluster[-1]['timestamp'] - cluster[0]['timestamp']
            logger.info(f"   [{i}/{num_clusters_to_process}] Cluster: {len(cluster)} frames ({cluster[0]['timestamp']:.1f}s-{cluster[-1]['timestamp']:.1f}s)")

            try:
                questions = self._generate_cluster_questions(
                    cluster=cluster,
                    cluster_index=i,
                    audio_analysis=audio_analysis,
                    frames_dir=frames_dir,
                    model=model
                )
                cluster_questions.extend(questions)
                logger.info(f"      ✓ Generated {len(questions)} questions (total: {len(cluster_questions)})")

            except Exception as e:
                logger.error(f"      ✗ Failed to generate cluster questions: {e}")
                continue

        all_questions.extend(cluster_questions)
        logger.info(f"\n   ✓ Stage 1 complete: {len(cluster_questions)} cluster questions generated")

        # Step 4e: Generate distributed single-frame questions across segments
        logger.info(f"\n   Stage 2: Generating distributed single-frame questions...")
        single_questions = []

        for seg in segments:
            if len(seg['single_frames']) == 0:
                logger.info(f"   Segment {seg['segment_id']}: No single frames available (skip)")
                continue

            # Select top frames from this segment
            segment_frames = sorted(seg['single_frames'], key=lambda f: f.get('priority', 0), reverse=True)
            num_to_select = min(seg['target_singles'], len(segment_frames))

            logger.info(f"   Segment {seg['segment_id']} ({seg['start']:.0f}-{seg['end']:.0f}s): Selecting {num_to_select}/{len(segment_frames)} frames")

            for i, frame in enumerate(segment_frames[:num_to_select], 1):
                frame_id = frame['frame_id']
                frame_model = frame_models.get(frame_id, model)

                logger.info(f"      [{i}/{num_to_select}] Processing {frame_id} (t={frame['timestamp']:.1f}s)")

                try:
                    questions = self._generate_frame_questions(
                        frame=frame,
                        audio_analysis=audio_analysis,
                        frames_dir=frames_dir,
                        model=frame_model
                    )
                    single_questions.extend(questions)
                    logger.info(f"         ✓ Generated {len(questions)} questions")

                except Exception as e:
                    logger.error(f"         ✗ Failed: {e}")
                    continue

        all_questions.extend(single_questions)
        logger.info(f"\n   ✓ Stage 2 complete: {len(single_questions)} single-frame questions generated")
        logger.info(f"\n   📊 Total questions: {len(all_questions)} ({len(cluster_questions)} cluster + {len(single_questions)} single)")
        logger.info(f"   📍 Coverage: {len(segments)} segments across {video_duration:.1f}s (100% timeline coverage)")

        # Step 4f: Validate against hallucinations (Critical Gap #3)
        if self.hallucination_validator:
            logger.info(f"\n✓ Step 4f: Validating {len(all_questions)} questions for hallucinations...")
            # Convert to dicts for validation
            questions_as_dicts = []
            for q in all_questions:
                q_dict = {
                    'question_id': q.question_id,
                    'question': q.question,
                    'golden_answer': q.answer,
                    'question_type': q.question_type,
                    'frame_id': q.frame_id
                }
                questions_as_dicts.append((q, q_dict))  # Keep original for later

            validated_dicts = [q_dict for _, q_dict in questions_as_dicts]
            valid_dicts, validation_stats = self.hallucination_validator.validate_question_batch(
                validated_dicts,
                frames_dir
            )

            # Map back to original GeneratedQuestion objects
            valid_ids = {q['question_id'] for q in valid_dicts}
            all_questions = [q for q, q_dict in questions_as_dicts if q_dict['question_id'] in valid_ids]

            logger.info(f"   ✅ Hallucination validation complete: {len(all_questions)} valid questions")
            if validation_stats['rejections'] > 0:
                logger.info(f"   ⚠️  Rejected {validation_stats['rejections']} questions for hallucination issues")

        # Step 5: Validate and select best questions
        logger.info(f"\n✓ Step 5: Selecting best {self.config['target_questions']} questions...")
        final_questions = self._select_best_questions(all_questions)
        logger.info(f"   Selected {len(final_questions)} questions")

        # Step 6: Generate summary
        cost_summary = self.cost_tracker.get_summary()
        self.cost_tracker.print_summary()

        result = {
            'questions': final_questions,
            'cost_summary': cost_summary,
            'frame_stats': [f.to_dict() for f in self.cost_tracker.frame_stats],
            'question_stats': [q.to_dict() for q in self.cost_tracker.question_stats],
            'metadata': {
                'video_id': video_id,
                'frames_processed': len(top_frames),
                'frames_filtered_from': len(phase5_frames),
                'model_used': model,
                # ✅ NEW: Add audio validation stats
                'audio_validation': {
                    'rejected_invalid_audio': self.duplicate_stats['rejected_invalid_audio'],
                    'audio_duplicates_removed': self.duplicate_stats['audio_duplicates_removed']
                }
            }
        }

        logger.info("=" * 80)
        logger.info("✅ PHASE 8 COMPLETE")
        logger.info("=" * 80)

        return result

    def _add_frame_ids(self, phase5_frames: List[Dict], frames_dir: Path, video_id: str) -> List[Dict]:
        """
        Add frame_id to each Phase 5 frame by matching timestamps to frames_metadata.json.

        Args:
            phase5_frames: Frames from Phase 5 selection_plan (have timestamp, no frame_id)
            frames_dir: Directory containing extracted frames
            video_id: Video identifier

        Returns:
            phase5_frames with frame_id added to each frame
        """
        # Load frames metadata
        # Note: frames_dir already includes video_id (e.g., outputs/video_XXX/frames/video_id/)
        metadata_path = frames_dir / "frames_metadata.json"

        if not metadata_path.exists():
            logger.warning(f"   frames_metadata.json not found at {metadata_path}")
            logger.warning("   Generating frame_ids from timestamp as fallback")

            # Fallback: Generate frame_id as single_XXX based on index
            for i, frame in enumerate(phase5_frames):
                frame['frame_id'] = f"single_{i:03d}"
            return phase5_frames

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Build timestamp -> (frame_id, image_path) mapping
        timestamp_to_frame_info = {}
        for frame_meta in metadata.get('frames', []):
            timestamp = frame_meta.get('timestamp')
            frame_id = frame_meta.get('frame_id')
            image_path = frame_meta.get('image_path')
            if timestamp is not None and frame_id:
                timestamp_to_frame_info[timestamp] = {
                    'frame_id': frame_id,
                    'image_path': image_path
                }

        logger.info(f"   Loaded {len(timestamp_to_frame_info)} frame mappings from metadata")

        # Match Phase 5 frames to frame_ids and image_paths
        matched_count = 0
        for frame in phase5_frames:
            timestamp = frame.get('timestamp')

            # Exact match
            if timestamp in timestamp_to_frame_info:
                frame['frame_id'] = timestamp_to_frame_info[timestamp]['frame_id']
                frame['image_path'] = timestamp_to_frame_info[timestamp]['image_path']
                matched_count += 1
            else:
                # Try finding closest match (within 0.1s tolerance)
                closest_timestamp = None
                min_diff = float('inf')

                for ts in timestamp_to_frame_info.keys():
                    diff = abs(ts - timestamp)
                    if diff < min_diff and diff <= 0.1:
                        min_diff = diff
                        closest_timestamp = ts

                if closest_timestamp:
                    frame['frame_id'] = timestamp_to_frame_info[closest_timestamp]['frame_id']
                    frame['image_path'] = timestamp_to_frame_info[closest_timestamp]['image_path']
                    matched_count += 1
                    logger.debug(f"   Matched t={timestamp:.1f}s to {frame['frame_id']} (closest: {closest_timestamp:.1f}s)")
                else:
                    # No match found - generate fallback ID
                    logger.warning(f"   No frame_id found for timestamp {timestamp:.1f}s, using fallback")
                    frame['frame_id'] = f"unknown_{int(timestamp*10):04d}"
                    frame['image_path'] = None

        logger.info(f"   Successfully matched {matched_count}/{len(phase5_frames)} frames")

        return phase5_frames

    def _filter_frames(self, frames: List[Dict]) -> List[Dict]:
        """
        Filter frames based on quality criteria with adaptive thresholds.

        Keeps frames with:
        - priority >= min_frame_priority (0.75 for Claude frames)
        - priority >= 0.65 for gap-fills (relaxed threshold)
        - len(question_types) >= min_question_types

        Adaptive threshold ensures gap-fills have a chance to fill spatial coverage gaps.
        """
        filtered = []
        base_threshold = self.config['min_frame_priority']  # 0.75
        gap_fill_threshold = base_threshold * 0.87  # 0.65 (relaxed for gap-fills)

        # Track stats for logging
        claude_frames_total = 0
        claude_frames_passed = 0
        gap_fills_total = 0
        gap_fills_passed = 0

        for frame in frames:
            priority = frame.get('priority', 0)
            question_types = frame.get('question_types', [])
            is_gap_fill = frame.get('is_gap_fill', False)

            # Track frame counts
            if is_gap_fill:
                gap_fills_total += 1
            else:
                claude_frames_total += 1

            # Adaptive threshold based on frame type
            threshold = gap_fill_threshold if is_gap_fill else base_threshold

            if priority >= threshold and len(question_types) >= self.config['min_question_types']:
                filtered.append(frame)
                if is_gap_fill:
                    gap_fills_passed += 1
                else:
                    claude_frames_passed += 1

        # Log adaptive threshold usage
        if gap_fills_total > 0:
            logger.info(f"   Adaptive thresholds: Claude frames {base_threshold:.2f}, Gap-fills {gap_fill_threshold:.2f}")
            logger.info(f"   Claude frames: {claude_frames_passed}/{claude_frames_total} passed")
            logger.info(f"   Gap-fills: {gap_fills_passed}/{gap_fills_total} passed (relaxed threshold)")

        return filtered

    def _select_top_frames(self, frames: List[Dict]) -> List[Dict]:
        """
        Select top N frames based on composite score.

        Score = priority × num_question_types × type_difficulty_avg
        """
        # Calculate scores
        scored_frames = []

        for frame in frames:
            priority = frame.get('priority', 0.5)
            question_types = frame.get('question_types', [])

            # Get average difficulty of question types
            type_priorities = get_question_type_priority(question_types)
            avg_type_priority = sum(type_priorities.values()) / len(type_priorities) if type_priorities else 0.5

            # Composite score
            score = priority * len(question_types) * avg_type_priority

            scored_frames.append({
                **frame,
                'score': score
            })

        # Sort by score and take top N
        scored_frames.sort(key=lambda f: f['score'], reverse=True)
        top_frames = scored_frames[:self.config['max_frames']]

        return top_frames

    def _assign_models(self, frames: List[Dict]) -> Dict[str, str]:
        """
        Assign GPT-4o to all frames.

        GPT-4o provides excellent vision capabilities at lower cost than Claude,
        so we use it for all frames regardless of complexity.
        """
        model = self.config['model']
        return {frame['frame_id']: model for frame in frames}

    def _identify_clusters(self, frames: List[Dict]) -> List[List[Dict]]:
        """
        ✅ FIX #1: Group frames into dense clusters based on temporal proximity.

        A cluster = 2-4 consecutive frames within 10 seconds.

        Args:
            frames: List of frame dicts with 'timestamp'

        Returns:
            List of clusters, where each cluster is a list of MIN_CLUSTER_SIZE-MAX_CLUSTER_SIZE frames
        """
        if len(frames) < MIN_CLUSTER_SIZE:
            return []

        # ✅ FIX #1: CRITICAL - Sort by timestamp first
        sorted_frames = sorted(frames, key=lambda f: f['timestamp'])

        clusters = []
        current_cluster = [sorted_frames[0]]

        for i in range(1, len(sorted_frames)):
            time_gap = sorted_frames[i]['timestamp'] - current_cluster[-1]['timestamp']

            # If within time gap and cluster not too large, add to current cluster
            if time_gap <= MAX_CLUSTER_TIME_GAP_SECONDS and len(current_cluster) < MAX_CLUSTER_SIZE:
                current_cluster.append(sorted_frames[i])
            else:
                # Save cluster if it has minimum required frames
                if len(current_cluster) >= MIN_CLUSTER_SIZE:
                    clusters.append(current_cluster)
                current_cluster = [sorted_frames[i]]

        # Don't forget last cluster
        if len(current_cluster) >= MIN_CLUSTER_SIZE:
            clusters.append(current_cluster)

        return clusters

    def _build_clusters_from_phase5(
        self,
        cluster_metadata: List[Dict],
        frames_dir: Path,
        video_id: str
    ) -> Tuple[List[List[Dict]], Dict[int, Dict]]:
        """
        Build cluster frame lists from Phase 5's Claude-validated cluster metadata.

        ARCHITECTURE:
        - Phase 5 (Claude): Identifies cluster ranges (start, end, frame_count)
        - Phase 6: Extracts evenly-spaced frames for each cluster
        - Phase 8 (this method): Loads extracted cluster frames from Phase 6

        Args:
            cluster_metadata: Phase 5's dense_clusters array
            frames_dir: Directory containing extracted frames
            video_id: Video identifier

        Returns:
            Tuple of (clusters, metadata_dict) where:
            - clusters: List of clusters, where each cluster is a list of frame dicts
            - metadata_dict: Dict mapping cluster index to Phase 5 metadata
        """
        if not cluster_metadata:
            logger.warning("No cluster metadata provided from Phase 5")
            return [], {}

        # Load frames_metadata.json from Phase 6
        # Note: frames_dir already includes video_id (e.g., outputs/video_XXX/frames/video_id/)
        metadata_path = frames_dir / "frames_metadata.json"
        if not metadata_path.exists():
            logger.error(f"frames_metadata.json not found at {metadata_path}")
            return [], {}

        with open(metadata_path, 'r') as f:
            frames_metadata = json.load(f)

        all_extracted_frames = frames_metadata.get('frames', [])

        clusters = []
        cluster_metadata_dict = {}  # Maps cluster index to Phase 5 metadata

        for cluster_idx, cluster_meta in enumerate(cluster_metadata):
            start_ts = cluster_meta['start']
            end_ts = cluster_meta['end']
            expected_count = cluster_meta['frame_count']
            reason = cluster_meta.get('reason', 'Unknown')

            # Find cluster frames extracted by Phase 6
            # Phase 6 tags cluster frames with frame_type="cluster" and cluster_id
            cluster_id = f"cluster_{cluster_idx:02d}"

            cluster_frames = [
                f for f in all_extracted_frames
                if f.get('frame_type') == 'cluster' and f.get('cluster_id') == cluster_id
            ]

            # Sort by timestamp
            cluster_frames.sort(key=lambda f: f['timestamp'])

            # Validate we have enough frames
            if len(cluster_frames) < MIN_CLUSTER_SIZE:
                logger.warning(f"⚠️  Cluster {start_ts:.1f}s-{end_ts:.1f}s only has {len(cluster_frames)} frames (need {MIN_CLUSTER_SIZE}), skipping")
                logger.warning(f"   Reason: {reason}")
                continue

            duration = end_ts - start_ts
            logger.debug(f"   Built cluster: {start_ts:.1f}s-{end_ts:.1f}s ({duration:.1f}s, {len(cluster_frames)} frames)")
            logger.debug(f"   Reason: {reason[:80]}...")

            # ✅ HYBRID MODE: Store Phase 5 metadata separately (can't set attributes on list objects)
            cluster_metadata_dict[len(clusters)] = cluster_meta

            clusters.append(cluster_frames)

        logger.info(f"   ✅ Built {len(clusters)} clusters from Phase 5 metadata")
        return clusters, cluster_metadata_dict

    # ✅ MODIFIED: Trust Phase 5 validation FIRST
    def _validate_cluster_visual_coherence(
        self,
        cluster: List[Dict],
        frames_dir: Path,
        cluster_index: int = None
    ) -> Tuple[bool, str]:
        """
        Validate cluster has visual coherence.
        
        ✅ MODIFIED: Three-tier validation approach:
        1. PRIORITY: Trust Phase 5's Claude validation (scene_type consistency check)
        2. FALLBACK: Histogram comparison (fast, no dependencies)
        3. OPTIONAL: ImageHash (detailed, if available)
        
        This preserves spatial distribution by avoiding false positives from ImageHash.

        Args:
            cluster: List of 2-4 frame dicts with 'image_path'
            frames_dir: Directory containing frame images
            cluster_index: Index of cluster (1-based from loop, converted to 0-based for dict lookup)

        Returns:
            (is_valid: bool, rejection_reason: str)
        """
        # ✅ TIER 1: Check Phase 5 validation FIRST (highest priority)
        cluster_metadata = {}
        if cluster_index is not None and hasattr(self, 'cluster_metadata'):
            # ✅ Convert 1-based cluster_index (from loop) to 0-based dict key
            cluster_metadata = self.cluster_metadata.get(cluster_index - 1, {})
            # ✅ FIXED: Handle None values in cluster_metadata dict
            if cluster_metadata is None:
                cluster_metadata = {}

        validation = cluster_metadata.get('validation', {})

        if validation:
            # Phase 5 provided validation - trust it!
            logger.info(f"   ✓ Using Phase 5 validation (cluster {cluster_index})")

            if validation.get('same_scene_type') and not validation.get('is_scene_cut'):
                logger.info(f"      ✅ Phase 5: scene_type consistent, not a scene cut")
                return True, ""

            if validation.get('is_scene_cut'):
                logger.warning(f"      ❌ Phase 5 marked as scene cut")
                return False, "Marked as scene cut by Phase 5"

            if not validation.get('same_scene_type'):
                logger.warning(f"      ❌ Phase 5 detected different scene_types")
                return False, "Different scene_types detected by Phase 5"

        # ✅ TIER 2: Phase 5 validation missing - use ImageHash as fallback
        # (This only happens for legacy clusters or if Phase 5 didn't validate)
        logger.warning(f"   ⚠️  Phase 5 validation missing for cluster {cluster_index}, using ImageHash fallback")
        
        try:
            # Verify imagehash is available
            pass
        except NameError:
            logger.warning("imagehash not installed, accepting cluster (no validation possible)")
            return True, ""  # Accept if no validation possible

        # Load all images in cluster and compute hashes
        hashes = []
        for i, frame in enumerate(cluster):
            image_path = Path(frame.get('image_path'))

            if not image_path.exists():
                return False, f"Image not found: {image_path}"

            try:
                img = Image.open(image_path)
                img_hash = imagehash.average_hash(img)
                hashes.append(img_hash)
            except Exception as e:
                logger.warning(f"Failed to hash frame {i}: {e}")
                return False, f"Failed to hash frame {i}: {e}"

        if len(hashes) < MIN_CLUSTER_SIZE:
            return False, f"Not enough frames to validate (need {MIN_CLUSTER_SIZE})"

        # Check 1: Scene cut detection (consecutive frame differences)
        # Use adaptive threshold based on video characteristics
        threshold = self.scene_cut_threshold
        max_consecutive_diff = 0
        for i in range(len(hashes) - 1):
            diff = hashes[i] - hashes[i+1]
            max_consecutive_diff = max(max_consecutive_diff, diff)

            # ✅ ADAPTIVE THRESHOLD: Dynamically adjusted based on video motion
            # High motion (sports) = 28, Medium = 22, Low (interview) = 15
            if diff > threshold:  # Scene cut threshold (adaptive)
                logger.debug(f"Scene cut detected: frames {i}→{i+1} (diff={diff}, threshold={threshold})")
                return False, f"Scene cut detected between frames {i} and {i+1} (hash_diff={diff}, threshold={threshold})"

        # Check 2: Static scene detection (average change too low)
        total_diff = sum(hashes[i] - hashes[i+1] for i in range(len(hashes)-1))
        avg_diff = total_diff / (len(hashes) - 1)

        if avg_diff < MIN_AVG_IMAGEHASH_DIFF:  # Too static threshold
            logger.debug(f"Too static: avg_diff={avg_diff:.1f}")
            return False, f"Too static - no meaningful progression (avg_diff={avg_diff:.1f}, threshold={MIN_AVG_IMAGEHASH_DIFF})"

        # Check 3: Overall coherence (start vs end shouldn't be too different)
        # Use same adaptive threshold for start-to-end validation
        if len(hashes) >= MIN_FRAMES_FOR_START_END_CHECK:
            start_end_diff = hashes[0] - hashes[-1]
            if start_end_diff > threshold:  # Too much change threshold (adaptive)
                logger.debug(f"Start-to-end too different: diff={start_end_diff}, threshold={threshold}")
                return False, f"Start and end too different - possible multiple scenes (diff={start_end_diff}, threshold={threshold})"

        # Passed all checks
        logger.debug(f"✓ Cluster validated: max_diff={max_consecutive_diff}, avg_diff={avg_diff:.1f}, threshold={threshold}")
        return True, ""

    def _generate_cluster_questions(
        self,
        cluster: List[Dict],
        cluster_index: int,
        audio_analysis: Dict,
        frames_dir: Path,
        model: str
    ) -> List[GeneratedQuestion]:
        """
        Generate multi-frame questions for a cluster of frames.

        Args:
            cluster: List of 2-4 consecutive frames
            cluster_index: Index of this cluster (1-based for logging, converted to 0-based for dict access)
            audio_analysis: Audio analysis data
            frames_dir: Directory containing frame images
            model: Model to use (GPT-4o)

        Returns:
            List of GeneratedQuestion objects (1-3 questions)
        """
        # ✅ MODIFIED: Validate cluster visual coherence (now trusts Phase 5 first)
        is_valid, rejection_reason = self._validate_cluster_visual_coherence(
            cluster,
            frames_dir,
            cluster_index
        )

        if not is_valid:
            logger.warning(f"      ❌ Cluster {cluster_index} REJECTED: {rejection_reason}")
            logger.warning(f"         Frames: {cluster[0]['timestamp']:.1f}s - {cluster[-1]['timestamp']:.1f}s ({len(cluster)} frames)")
            return []  # Skip this cluster entirely

        logger.info(f"      ✅ Cluster {cluster_index} passed visual validation")

        # EXISTING CODE BELOW (unchanged):
        first_frame = cluster[0]
        last_frame = cluster[-1]

        # Get audio cues for first and last frame
        audio_start = self._find_audio_cue(first_frame['timestamp'], audio_analysis)
        audio_end = self._find_audio_cue(last_frame['timestamp'], audio_analysis)

        # ✅ FIXED: Validate audio cues BEFORE making API call (saves cost)
        # Skip cluster if either audio cue is invalid
        is_valid_start, start_rejection = validate_audio_cue_content(audio_start)
        is_valid_end, end_rejection = validate_audio_cue_content(audio_end)

        if not is_valid_start:
            logger.debug(f"      ⚠️  Skipping cluster {cluster_index} - start audio invalid: {start_rejection}")
            return []
        if not is_valid_end:
            logger.debug(f"      ⚠️  Skipping cluster {cluster_index} - end audio invalid: {end_rejection}")
            return []

        # ✅ HYBRID MODE: Select best 3 question types for this cluster
        # Convert 1-based cluster_index (from loop) to 0-based dict key
        cluster_metadata = self.cluster_metadata.get(cluster_index - 1, {})
        # ✅ FIXED: Ensure cluster_metadata is not None (defensive programming)
        if cluster_metadata is None:
            cluster_metadata = {}
        selected_question_types = self._select_best_cluster_question_types(cluster_metadata)
        logger.debug(f"      Selected question types: {selected_question_types}")

        # Build cluster data
        cluster_data = {
            'frames': cluster,
            'audio_start': audio_start,
            'audio_end': audio_end,
            'start_timestamp': first_frame['timestamp'],
            'end_timestamp': last_frame['timestamp'],
            'selected_question_types': selected_question_types  # Override with curated types
        }

        # ✅ FIX #2 & #3: Use the build_cluster_prompt function defined above
        prompt = build_cluster_prompt(cluster_data, self.config)

        # Load all images in cluster
        images_b64 = []
        for frame in cluster:
            image_path_str = frame.get('image_path')
            if not image_path_str:
                logger.warning(f"      No image_path in frame {frame.get('frame_id')}, skipping cluster")
                return []

            image_path = Path(image_path_str)
            if not image_path.exists():
                logger.warning(f"      Image not found: {image_path}, skipping cluster")
                return []

            with open(image_path, 'rb') as f:
                image_b64 = base64.b64encode(f.read()).decode('utf-8')
                images_b64.append(image_b64)

        # Call Vision API with multiple images
        try:
            response = self._call_vision_api_multi_image(
                prompt=prompt,
                images=images_b64,
                model=model
            )

            # ✅ FIX #6: Add token tracking for clusters
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

            # Calculate cost
            token_usage = TokenUsage(model, input_tokens, output_tokens)
            cost = token_usage.total_cost

            # Parse questions from response
            questions = self._parse_questions(response, first_frame['timestamp'], cluster_index=cluster_index)

            # ✅ FIX #5 & #6: Add cluster metadata and token tracking to questions
            cluster_id = f"cluster_{first_frame['frame_id']}_{last_frame['frame_id']}"
            
            # Track frame-level tokens for cluster
            frame_stat = FrameTokenStats(
                frame_id=cluster_id,
                timestamp=first_frame['timestamp'],
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                questions_generated=len(questions),
                cost=cost
            )
            self.cost_tracker.add_frame(frame_stat)

            for i, q in enumerate(questions):
                # Add cluster metadata
                q.frame_sequence = [f['frame_id'] for f in cluster]
                q.start_timestamp = first_frame['timestamp']
                q.end_timestamp = last_frame['timestamp']
                q.frame_id = cluster_id

                # ✅ Critical Gap #2: Enforce minimum temporal window by question type
                center_timestamp = (first_frame['timestamp'] + last_frame['timestamp']) / 2.0
                q.start_timestamp, q.end_timestamp = enforce_temporal_window(
                    start_seconds=q.start_timestamp,
                    end_seconds=q.end_timestamp,
                    question_type=q.question_type,
                    video_duration=self.video_duration,
                    frame_timestamp=center_timestamp
                )

                # ✅ Auto-detect sub-task type if not already set
                if not hasattr(q, 'sub_task_type') or not q.sub_task_type:
                    q.sub_task_type = detect_sub_task_types(q.question, q.answer)

                # Update token tracking
                q.tokens = {
                    'input_share': input_tokens // len(questions),
                    'output': output_tokens // len(questions),
                    'total': (input_tokens + output_tokens) // len(questions)
                }
                q.cost = cost / len(questions)
                
                # Track question-level tokens
                q_stat = QuestionTokenStats(
                    question_id=q.question_id,
                    question_type=q.question_type,
                    frame_id=cluster_id,
                    model=model,
                    input_tokens_share=input_tokens // len(questions),
                    output_tokens=output_tokens // len(questions),
                    cost=cost / len(questions)
                )
                self.cost_tracker.add_question(q_stat)

            # ✅ CRITICAL: Validate questions match assigned ontologies from Pass 2A/2B
            assigned_ontologies = cluster_metadata.get('question_types', [])
            validated_questions = []
            rejected_count = 0

            for q in questions:
                is_valid, rejection_reason = self._validate_question_ontology_match(
                    question=q,
                    assigned_ontologies=assigned_ontologies,
                    moment_mode="cluster"
                )

                if is_valid:
                    validated_questions.append(q)
                else:
                    rejected_count += 1
                    logger.warning(f"      ❌ Rejected question: {rejection_reason}")
                    logger.warning(f"         Question: {q.question[:80]}...")
                    logger.warning(f"         Type: {q.question_type}, Assigned: {assigned_ontologies}")

            if rejected_count > 0:
                logger.info(f"      Ontology validation: {len(validated_questions)}/{len(questions)} questions passed")

            return validated_questions

        except Exception as e:
            logger.error(f"      Error generating cluster questions: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def _call_vision_api_multi_image(
        self,
        prompt: str,
        images: List[str],
        model: str
    ):
        """
        ✅ FIX #4: Call GPT-4o Vision API with multiple images using structured outputs.

        Args:
            prompt: The prompt text
            images: List of base64-encoded images
            model: Model name (gpt-4o-2024-11-20)

        Returns:
            OpenAI response object with structured JSON
        """
        # Build content with multiple images
        content = [{"type": "text", "text": prompt}]

        for i, image_b64 in enumerate(images, 1):
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_b64}",
                    "detail": "high"
                }
            })

        # ✅ FIX #4: Use self.openai_client (not self.client)
        # ✅ QUICK WIN #1: Add system message for critical rules (+15% compliance)
        # ✅ QUICK WIN #2: Temperature 0.1 for strict rule adherence (+20% compliance)
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": CRITICAL_RULES_SYSTEM_MESSAGE},
                {"role": "user", "content": content}
            ],
            max_tokens=self.config['max_tokens_per_frame'],
            temperature=0.1,  # ✅ Changed from 0.7 → 0.1 for better rule compliance
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "adversarial_questions",
                    "schema": QUESTION_SCHEMA,
                    "strict": True
                }
            }
        )

        return response

    def _parse_questions(
        self,
        response,
        default_timestamp: float,
        cluster_index: Optional[int] = None
    ) -> List[GeneratedQuestion]:
        """
        Parse structured JSON response from vision API into GeneratedQuestion objects.

        ✅ MODIFIED: Now actually removes duplicates (not just logs warnings)

        Args:
            response: OpenAI API response with structured JSON
            default_timestamp: Default timestamp to use if parsing fails
            cluster_index: Optional cluster index for unique multi-frame question IDs

        Returns:
            List of GeneratedQuestion objects (deduplicated)
        """
        try:
            content = response.choices[0].message.content
            result = json.loads(content)
            questions_data = result.get('questions', [])

            # ✅ MODIFIED: ENFORCE duplicate removal (not just warn)
            if len(questions_data) > 1:
                threshold = self.config.get('duplicate_similarity_threshold', 0.7)
                has_duplicates, duplicate_warnings = check_duplicate_questions(questions_data, similarity_threshold=threshold)
                
                if has_duplicates:
                    logger.warning(f"      ⚠️  QUALITY ISSUE: Duplicate questions detected in cluster")
                    for warning in duplicate_warnings:
                        logger.warning(f"      {warning}")

                    # ✅ NEW: Actually remove duplicates (keep highest confidence)
                    original_count = len(questions_data)
                    deduplicated = []

                    for i, q1 in enumerate(questions_data):
                        is_duplicate = False

                        # Check if this question is too similar to any already-kept question
                        for j, q2 in enumerate(deduplicated):
                            similarity = calculate_question_similarity(q1['question'], q2['question'])

                            if similarity > threshold:
                                # Duplicate found - keep the one with higher confidence
                                if q1.get('confidence', 0) > q2.get('confidence', 0):
                                    # Replace lower-confidence with higher-confidence
                                    deduplicated[j] = q1
                                    logger.debug(f"         Replaced Q{j+1} with Q{i+1} (higher confidence: {q1.get('confidence', 0):.2f} > {q2.get('confidence', 0):.2f})")
                                is_duplicate = True
                                break

                        # If not a duplicate of any kept question, add it
                        if not is_duplicate:
                            deduplicated.append(q1)

                    questions_data = deduplicated
                    removed_count = original_count - len(questions_data)

                    if removed_count > 0:
                        logger.info(f"      🔧 Removed {removed_count} duplicate question(s) (kept highest confidence)")
                        logger.info(f"      ✓ Deduplicated: {original_count} → {len(questions_data)} questions")

            # ✅ ENHANCED: QUALITY CHECK (hedging + pronouns + audio cues) with auto-fix
            has_hedging, hedging_warnings = check_hedging_language(questions_data)
            has_pronouns, pronoun_warnings = check_pronoun_usage(questions_data)
            # ✅ Critical Gap #4: Check audio cue quality
            has_audio_issues, audio_warnings = check_audio_cue_quality(questions_data)

            # ✅ P1 Fix #3: Check audio modality diversity (Guidelines requirement)
            audio_diversity_valid, diversity_warnings = validate_audio_modality_diversity(
                questions_data, min_modalities=2
            )
            if not audio_diversity_valid:
                logger.warning(f"      ⚠️  AUDIO DIVERSITY: Insufficient audio modality diversity")
                for warning in diversity_warnings:
                    logger.warning(f"      {warning}")

            if has_hedging or has_pronouns or has_audio_issues:
                issues = []
                if has_hedging:
                    issues.append("hedging")
                    logger.warning(f"      ⚠️  QUALITY ISSUE: Hedging language detected")
                    for warning in hedging_warnings:
                        logger.warning(f"      {warning}")

                if has_pronouns:
                    issues.append("pronouns")
                    logger.warning(f"      ⚠️  QUALITY ISSUE: Pronoun usage detected")
                    for warning in pronoun_warnings:
                        logger.warning(f"      {warning}")

                if has_audio_issues:
                    issues.append("audio_cues")
                    logger.warning(f"      ⚠️  QUALITY ISSUE: Audio cue quality issues detected (Critical Gap #4)")
                    for warning in audio_warnings:
                        logger.warning(f"      {warning}")
                    # Reject questions with audio cue issues (can't auto-fix these easily)
                    questions_data = [q for q in questions_data if validate_audio_cue_content(q.get('audio_cue', ''))[0]]
                    logger.info(f"      🔧 Filtered out questions with invalid audio cues: {len(audio_warnings)} removed")

                # ✅ AUTO-FIX: Use quality fixer to repair affected questions (hedging/pronouns only)
                if self.hedging_fixer and (has_hedging or has_pronouns):
                    logger.info(f"      🔧 Auto-fixing quality issues ({', '.join(issues)}) with Claude Haiku...")
                    questions_data, fix_stats = self.hedging_fixer.fix_quality_issues(
                        questions_data,
                        has_hedging=has_hedging,
                        has_pronouns=has_pronouns,
                        hedging_warnings=hedging_warnings,
                        pronoun_warnings=pronoun_warnings
                    )
                    # Update cost tracking properly (total_cost is a read-only property)
                    if fix_stats['fixes_applied'] > 0:
                        fixer_stat = FrameTokenStats(
                            frame_id="quality_fixer",
                            timestamp=0.0,  # Not frame-specific
                            model="claude-haiku-4-5-20251001",
                            input_tokens=fix_stats['input_tokens'],
                            output_tokens=fix_stats['output_tokens'],
                            questions_generated=fix_stats['fixes_applied'],
                            cost=fix_stats['cost']
                        )
                        self.cost_tracker.add_frame(fixer_stat)
                    logger.info(f"      ✅ Fixed {fix_stats['fixes_applied']} questions (hedging: {fix_stats['hedging_fixes']}, pronouns: {fix_stats['pronoun_fixes']}, cost: ${fix_stats['cost']:.4f})")
                else:
                    logger.warning(f"      ⚠️  No quality fixer available - questions kept as-is")

            generated_questions = []

            for i, q_data in enumerate(questions_data):
                # Convert timestamps from MM:SS to seconds
                start_mmss = q_data.get('start_timestamp', seconds_to_mmss(default_timestamp))
                end_mmss = q_data.get('end_timestamp', seconds_to_mmss(default_timestamp))
                start_seconds = mmss_to_seconds(start_mmss)
                end_seconds = mmss_to_seconds(end_mmss)

                # ✅ FIX #7 (Part 2): Validate and clamp negative timestamps from LLM
                if start_seconds < 0:
                    logger.warning(f"      Negative start timestamp {start_mmss} detected, clamping to 00:00")
                    start_seconds = 0.0
                    start_mmss = "00:00"
                if end_seconds < 0:
                    logger.warning(f"      Negative end timestamp {end_mmss} detected, clamping to 00:00")
                    end_seconds = 0.0
                    end_mmss = "00:00"

                # Normalize question type
                raw_type = q_data.get('question_type', 'Unknown')
                normalized_type = normalize_question_type(raw_type)

                # ✅ NEW: Enforce minimum temporal windows by question type
                # (using get_min_temporal_window imported at top)
                min_window = get_min_temporal_window(normalized_type)
                actual_window = end_seconds - start_seconds

                if actual_window < min_window:
                    logger.warning(
                        f"      Question {i+1}: Window too short ({actual_window:.1f}s < {min_window:.1f}s required for {normalized_type})"
                    )

                    # Expand window symmetrically around center
                    center = (start_seconds + end_seconds) / 2 if end_seconds > start_seconds else default_timestamp
                    half_window = min_window / 2

                    start_seconds = max(0, center - half_window)
                    end_seconds = center + half_window

                    logger.info(
                        f"      Expanded window: {start_seconds:.1f}s - {end_seconds:.1f}s ({min_window:.1f}s total)"
                    )

                # ✅ FIX #9: Generate unique question IDs using cluster_index
                if cluster_index is not None:
                    question_id = f"cluster_{cluster_index:02d}_q{i+1:02d}"
                else:
                    question_id = f"cluster_q{i+1:02d}"  # Fallback for single frames

                # Create question object
                question = GeneratedQuestion(
                    question_id=question_id,
                    question=q_data.get('question', ''),
                    answer=q_data.get('golden_answer', ''),
                    question_type=normalized_type,
                    frame_id='cluster',  # Will be updated by caller
                    timestamp=start_seconds,
                    audio_cue=q_data.get('audio_cue', ''),
                    visual_cue=q_data.get('visual_cue', ''),
                    confidence=q_data.get('confidence', 0.8),
                    evidence=q_data.get('visual_cue', ''),
                    model=response.model,
                    tokens={},  # Will be updated by caller
                    cost=0.0,   # Will be updated by caller
                    # ✅ CRITICAL FIX: Set validated temporal window (prevents fallback to timestamp ± 1-2)
                    start_timestamp=start_seconds,
                    end_timestamp=end_seconds
                )

                generated_questions.append(question)

            return generated_questions

        except Exception as e:
            logger.error(f"      Error parsing questions from response: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def _generate_frame_questions(
        self,
        frame: Dict,
        audio_analysis: Dict,
        frames_dir: Path,
        model: str
    ) -> List[GeneratedQuestion]:
        """
        Generate questions for a single frame using GPT-4o Vision.

        Returns:
            List of GeneratedQuestion objects
        """
        frame_id = frame['frame_id']
        timestamp = frame['timestamp']

        # Find audio cue near this timestamp
        audio_cue = self._find_audio_cue(timestamp, audio_analysis)

        # ✅ FIXED: Validate audio cue BEFORE making API call (saves cost)
        # Skip generation if audio cue is invalid (placeholder, empty, vague)
        is_valid_audio, audio_rejection = validate_audio_cue_content(audio_cue)
        if not is_valid_audio:
            logger.debug(f"      ⚠️  Skipping frame {frame_id} - {audio_rejection}")
            return []  # Skip this frame entirely

        # Build specialist prompt
        frame_data = {
            'timestamp': timestamp,
            'question_types': frame.get('question_types', []),
            'audio_cue': audio_cue,
            'priority': frame.get('priority', 0.5),
            'frame_id': frame_id
        }
        prompt = build_specialist_prompt(frame_data, self.config)

        # Load frame image using the path from frames_metadata
        image_path_str = frame.get('image_path')

        if not image_path_str:
            # Fallback: Try to construct path
            logger.warning(f"      No image_path in frame data, trying to construct...")
            image_path = frames_dir / f"{frame_id}_{timestamp}s.jpg"
            if not image_path.exists():
                image_path = frames_dir / f"{frame_id}.jpg"
        else:
            # Use the path from frames_metadata (absolute path)
            image_path = Path(image_path_str)

        if not image_path.exists():
            logger.warning(f"      Frame image not found: {image_path}")
            return []

        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        # Call GPT-4o Vision API with structured outputs
        # ✅ QUICK WIN #1: System message for critical rules (+15% compliance)
        # ✅ QUICK WIN #2: Temperature 0.1 for strict rule adherence (+20% compliance)
        # ✅ RETRY LOGIC: Handle rate limiting with exponential backoff
        try:
            def make_api_call():
                return self.openai_client.chat.completions.create(
                    model=model,
                    max_tokens=self.config['max_tokens_per_frame'],
                    temperature=0.1,  # ✅ Changed from 0.7 → 0.1 for better rule compliance
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "adversarial_questions",
                            "schema": QUESTION_SCHEMA,
                            "strict": True
                        }
                    },
                    messages=[
                        {
                            "role": "system",
                            "content": CRITICAL_RULES_SYSTEM_MESSAGE
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_data}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }
                    ]
                )

            response = retry_with_exponential_backoff(make_api_call)

            # Extract token usage
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

            # Calculate cost
            token_usage = TokenUsage(model, input_tokens, output_tokens)
            cost = token_usage.total_cost

            # Parse response - structured outputs guarantee valid JSON
            content = response.choices[0].message.content

            try:
                result = json.loads(content)
                questions_data = result.get('questions', [])
            except json.JSONDecodeError as e:
                logger.error(f"      Failed to parse structured JSON response: {e}")
                logger.error(f"      Response: {content[:2000]}")
                return []

            # Track frame-level tokens
            frame_stat = FrameTokenStats(
                frame_id=frame_id,
                timestamp=timestamp,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                questions_generated=len(questions_data),
                cost=cost
            )
            self.cost_tracker.add_frame(frame_stat)

            # Convert to GeneratedQuestion objects and track per-question tokens
            generated_questions = []

            for i, q_data in enumerate(questions_data):
                question_id = f"{frame_id}_q{i+1:02d}"

                # Estimate tokens for this question (proportional)
                q_output_tokens = len(json.dumps(q_data)) * 0.3  # Rough estimate: 1 token ≈ 3.3 chars
                q_input_share = input_tokens // len(questions_data)

                # Calculate question cost
                q_token_usage = TokenUsage(model, q_input_share, int(q_output_tokens))
                q_cost = q_token_usage.total_cost

                # Normalize question type (schema already enforces exact types via enum)
                raw_type = q_data.get('question_type', 'Unknown')
                normalized_type = normalize_question_type(raw_type)
                if raw_type != normalized_type:
                    logger.debug(f"Normalized type '{raw_type}' → '{normalized_type}'")

                # ✅ Auto-detect sub-task type if not set by LLM
                sub_task_type = q_data.get('sub_task_type')
                if not sub_task_type:
                    sub_task_type = detect_sub_task_types(
                        q_data.get('question', ''),
                        q_data.get('golden_answer', '')
                    )
                    if sub_task_type:
                        logger.debug(f"Auto-detected sub-task type: {sub_task_type}")

                # Convert timestamps from MM:SS to seconds
                start_mmss = q_data.get('start_timestamp', seconds_to_mmss(timestamp))
                end_mmss = q_data.get('end_timestamp', seconds_to_mmss(timestamp))
                start_seconds = mmss_to_seconds(start_mmss)
                end_seconds = mmss_to_seconds(end_mmss)

                # ✅ Critical Gap #2: Enforce minimum temporal window by question type
                start_seconds, end_seconds = enforce_temporal_window(
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    question_type=normalized_type,
                    video_duration=self.video_duration,
                    frame_timestamp=timestamp
                )

                question = GeneratedQuestion(
                    question_id=question_id,
                    question=q_data.get('question', ''),
                    answer=q_data.get('golden_answer', ''),  # ← Schema field name
                    question_type=normalized_type,
                    frame_id=frame_id,
                    timestamp=start_seconds,  # Use converted start timestamp
                    audio_cue=q_data.get('audio_cue', ''),  # ← From schema, not frame audio
                    visual_cue=q_data.get('visual_cue', ''),  # ← Schema field name
                    confidence=q_data.get('confidence', 0.8),
                    evidence=q_data.get('visual_cue', ''),  # Use visual_cue for evidence
                    model=model,
                    tokens={
                        'input_share': q_input_share,
                        'output': int(q_output_tokens),
                        'total': q_input_share + int(q_output_tokens)
                    },
                    cost=q_cost,
                    sub_task_type=sub_task_type,  # ✅ Auto-detected or from LLM
                    # ✅ CRITICAL FIX: Set validated temporal window (prevents fallback to timestamp ± 1-2)
                    start_timestamp=start_seconds,
                    end_timestamp=end_seconds
                )

                generated_questions.append(question)

                # Track question-level tokens
                q_stat = QuestionTokenStats(
                    question_id=question_id,
                    question_type=question.question_type,
                    frame_id=frame_id,
                    model=model,
                    input_tokens_share=q_input_share,
                    output_tokens=int(q_output_tokens),
                    cost=q_cost
                )
                self.cost_tracker.add_question(q_stat)

            # ✅ CRITICAL: Validate questions match assigned ontologies from Pass 2A/2B
            assigned_ontologies = frame.get('question_types', [])
            moment_mode = frame.get('mode', 'precise')  # Default to precise for single frames
            validated_questions = []
            rejected_count = 0

            for q in generated_questions:
                is_valid, rejection_reason = self._validate_question_ontology_match(
                    question=q,
                    assigned_ontologies=assigned_ontologies,
                    moment_mode=moment_mode
                )

                if is_valid:
                    validated_questions.append(q)
                else:
                    rejected_count += 1
                    logger.warning(f"      ❌ Rejected question: {rejection_reason}")
                    logger.warning(f"         Question: {q.question[:80]}...")
                    logger.warning(f"         Type: {q.question_type}, Assigned: {assigned_ontologies}")

            if rejected_count > 0:
                logger.info(f"      Ontology validation: {len(validated_questions)}/{len(generated_questions)} questions passed")

            return validated_questions

        except Exception as e:
            logger.error(f"      GPT-4o API call failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def _describe_audio_event(self, event: Dict) -> str:
        """Convert audio event to natural language description"""
        event_type = event.get('type', 'unknown')
        subtype = event.get('subtype', '')
        characteristics = event.get('characteristics', {})

        if event_type == 'background_music':
            tempo = characteristics.get('tempo', 0)
            if tempo > 0:
                return f"{subtype} music (tempo: {tempo:.0f} BPM)"
            else:
                return f"{subtype} music"

        elif event_type == 'sound_effect':
            intensity = characteristics.get('intensity', 'medium')
            return f"{intensity} {subtype} sound effect"

        elif event_type == 'crowd_sound':
            intensity = characteristics.get('intensity', 'medium')
            return f"{intensity} {subtype}"

        elif event_type == 'music_change':
            change_mag = characteristics.get('change_magnitude', 0)
            direction = subtype.replace('tempo_', '')
            return f"Music tempo {direction} by {change_mag:.0f} BPM"

        return f"{event_type} ({subtype})"

    def _find_audio_cue(self, timestamp: float, audio_analysis: Dict) -> str:
        """
        Find comprehensive audio cue near timestamp (transcript + events)

        Combines:
        - Speech transcript (closest segment)
        - Audio events within 30-second window (music, sound effects, silence)

        Returns natural language audio description
        """
        # 1. Get transcript text
        segments = audio_analysis.get('segments', [])
        transcript_text = ""

        closest_segment = None
        min_distance = float('inf')

        for segment in segments:
            seg_start = segment.get('start', 0)
            seg_end = segment.get('end', 0)
            seg_mid = (seg_start + seg_end) / 2

            distance = abs(seg_mid - timestamp)

            if distance < min_distance:
                min_distance = distance
                closest_segment = segment

        if closest_segment:
            transcript_text = closest_segment.get('text', '')

        # 2. Get audio events nearby (30-second window)
        audio_events = audio_analysis.get('audio_events', [])

        # ✅ VALIDATION: Log audio events presence
        if not audio_events:
            logger.warning(f"⚠️  No audio_events in audio_analysis at timestamp {timestamp:.1f}s! Only using transcript.")
        else:
            logger.debug(f"✓ Found {len(audio_events)} total audio events in audio_analysis")

        nearby_events = []

        for event in audio_events:
            event_start = event.get('start', 0)
            event_end = event.get('end', 0)
            event_mid = (event_start + event_end) / 2

            # Check if event overlaps with timestamp window
            if abs(event_mid - timestamp) <= 30.0:  # 30-second window
                nearby_events.append(self._describe_audio_event(event))

        # 3. Combine into rich audio cue
        audio_cue_parts = []

        # Add transcript if present and not filtered
        if transcript_text and not transcript_text.startswith('['):
            audio_cue_parts.append(f"'{transcript_text}'")

        # Add audio events
        audio_cue_parts.extend(nearby_events)

        # ✅ VALIDATION: Log what was found
        if nearby_events:
            logger.debug(f"✓ Found {len(nearby_events)} audio events near timestamp {timestamp:.1f}s: {nearby_events}")

        # Return combined or fallback
        if audio_cue_parts:
            return "; ".join(audio_cue_parts)
        else:
            return transcript_text if transcript_text else "No significant audio"

    def _select_best_questions(self, questions: List[GeneratedQuestion]) -> List[GeneratedQuestion]:
        """
        Select best N questions from candidate pool with audio validation.

        Strategy:
        1. Validate audio timestamps (remove silent/scene-cut segments)
        2. Detect audio duplicates (remove questions using same audio)
        3. Round-robin selection for type diversity
        4. Max 2 questions per frame

        ✅ FIX #2: Round-robin selection for better type diversity
        ✅ FIX #3: Max 2 questions per frame (prevents redundancy)
        ✅ NEW: Audio timestamp validation + duplicate detection
        """
        target = self.config['target_questions']

        if len(questions) <= target:
            return questions

        # ✅ STEP 1: Validate audio timestamps
        logger.info(f"\n🔍 Step 1: Validating audio timestamps for {len(questions)} questions...")

        validated_questions = []
        rejected_count = 0

        # Get audio path from self (set in generate_questions)
        if self.audio_path and self.audio_path.exists():
            try:
                validator = AudioTimestampValidator(self.audio_path)

                for i, q in enumerate(questions):
                    start_ts = q.start_timestamp if q.start_timestamp is not None else q.timestamp
                    end_ts = q.end_timestamp if q.end_timestamp is not None else (start_ts + 10.0)

                    is_valid, confidence, reason = validator.validate_timestamp(start_ts, end_ts)

                    if is_valid:
                        validated_questions.append(q)
                    else:
                        rejected_count += 1
                        logger.debug(f"   ❌ Rejected Q{i} ({q.question_id}): {reason}")

                logger.info(f"   ✓ Validated: {len(validated_questions)}/{len(questions)} questions")
                logger.info(f"   ✗ Rejected: {rejected_count} (silent/scene-cuts)")

                # Track stats
                self.duplicate_stats['rejected_invalid_audio'] = rejected_count

            except Exception as e:
                logger.warning(f"   ⚠️  Audio validation failed: {e}")
                logger.warning("   Proceeding without validation")
                validated_questions = questions
        else:
            logger.warning("   ⚠️  No audio path available, skipping validation")
            validated_questions = questions

        # ✅ STEP 2: Detect audio duplicates
        logger.info(f"\n🔍 Step 2: Detecting duplicate audio segments...")

        if len(validated_questions) > 1 and self.audio_path and self.audio_path.exists():
            try:
                # Convert to dict format for duplicate detection
                q_dicts = []
                for q in validated_questions:
                    start_ts = q.start_timestamp if q.start_timestamp is not None else q.timestamp
                    end_ts = q.end_timestamp if q.end_timestamp is not None else (start_ts + 10.0)
                    q_dicts.append({
                        'start_timestamp': start_ts,
                        'end_timestamp': end_ts
                    })

                duplicates = detect_duplicate_audio_segments(
                    self.audio_path,
                    q_dicts,
                    similarity_threshold=AUDIO_DUPLICATE_SIMILARITY_THRESHOLD
                )

                if duplicates:
                    logger.warning(f"   ⚠️  Found {len(duplicates)} audio duplicate pairs")

                    # Remove lower-confidence question from each duplicate pair
                    to_remove = set()
                    for idx1, idx2, similarity in duplicates:
                        q1 = validated_questions[idx1]
                        q2 = validated_questions[idx2]

                        logger.debug(f"      Duplicate: Q{idx1} ↔ Q{idx2} (similarity={similarity:.2f})")
                        logger.debug(f"         Q{idx1}: confidence={q1.confidence:.2f}")
                        logger.debug(f"         Q{idx2}: confidence={q2.confidence:.2f}")

                        # Remove lower-confidence question
                        if q1.confidence < q2.confidence:
                            to_remove.add(idx1)
                            logger.debug(f"         → Removing Q{idx1} (lower confidence)")
                        else:
                            to_remove.add(idx2)
                            logger.debug(f"         → Removing Q{idx2} (lower confidence)")

                    # Filter out removed questions
                    deduplicated_questions = [
                        q for i, q in enumerate(validated_questions)
                        if i not in to_remove
                    ]

                    logger.info(f"   ✓ Removed {len(to_remove)} duplicate audio segments")
                    logger.info(f"   ✓ Remaining: {len(deduplicated_questions)} questions")

                    # Track stats
                    self.duplicate_stats['audio_duplicates_removed'] = len(to_remove)

                    validated_questions = deduplicated_questions
                else:
                    logger.info(f"   ✓ No audio duplicates found")

            except Exception as e:
                logger.warning(f"   ⚠️  Duplicate detection failed: {e}")
                logger.warning("   Proceeding without duplicate detection")
        else:
            logger.info(f"   ⚠️  Skipping duplicate detection (not enough questions or no audio)")

        # ✅ STEP 3: Round-robin selection
        logger.info(f"\n🎯 Step 3: Round-robin selection for type diversity...")

        # ✅ ROUND-ROBIN: Group questions by type
        questions = validated_questions  # Use validated questions from now on
        questions_by_type = {}
        for q in questions:
            qtype = q.question_type
            if qtype not in questions_by_type:
                questions_by_type[qtype] = []
            questions_by_type[qtype].append(q)

        # Sort each type's questions by confidence
        for qtype in questions_by_type:
            questions_by_type[qtype].sort(key=lambda q: q.confidence, reverse=True)

        # ✅ ROUND-ROBIN SELECTION: Take max 3 per type
        selected = []
        frame_counts = {}  # Track questions per frame
        max_per_type = 3
        max_per_frame = 2  # ✅ FIX #3: Max 2 questions per frame

        # ✅ IMPROVED ROUND-ROBIN: Try all questions of a type before giving up
        # Round 1: Take top 1 from each type
        for qtype in sorted(questions_by_type.keys()):
            # Try to find ANY question of this type that fits frame constraint
            for q in questions_by_type[qtype][:]:  # Iterate over copy
                # Check frame diversity
                if frame_counts.get(q.frame_id, 0) >= max_per_frame:
                    continue  # Try next question of same type

                selected.append(q)
                frame_counts[q.frame_id] = frame_counts.get(q.frame_id, 0) + 1
                questions_by_type[qtype].remove(q)
                break  # Found one, move to next type

        # Round 2: Take top 2 from each type
        for qtype in sorted(questions_by_type.keys()):
            type_count = len([q for q in selected if q.question_type == qtype])
            if type_count >= max_per_type:
                continue

            # Try to find ANY question of this type that fits frame constraint
            for q in questions_by_type[qtype][:]:  # Iterate over copy
                # Check frame diversity
                if frame_counts.get(q.frame_id, 0) >= max_per_frame:
                    continue  # Try next question of same type

                selected.append(q)
                frame_counts[q.frame_id] = frame_counts.get(q.frame_id, 0) + 1
                questions_by_type[qtype].remove(q)
                break  # Found one, move to next type

        # Round 3: Take top 3 from each type
        for qtype in sorted(questions_by_type.keys()):
            type_count = len([q for q in selected if q.question_type == qtype])
            if type_count >= max_per_type:
                continue

            # Try to find ANY question of this type that fits frame constraint
            for q in questions_by_type[qtype][:]:  # Iterate over copy
                # Check frame diversity
                if frame_counts.get(q.frame_id, 0) >= max_per_frame:
                    continue  # Try next question of same type

                selected.append(q)
                frame_counts[q.frame_id] = frame_counts.get(q.frame_id, 0) + 1
                questions_by_type[qtype].remove(q)
                break  # Found one, move to next type

        # If still short, fill with highest confidence remaining
        if len(selected) < target:
            remaining = [q for qtype_list in questions_by_type.values() for q in qtype_list]
            remaining.sort(key=lambda q: q.confidence, reverse=True)

            for q in remaining:
                # Check type limit
                type_count = len([sq for sq in selected if sq.question_type == q.question_type])
                if type_count >= max_per_type:
                    continue

                # Check frame limit
                if frame_counts.get(q.frame_id, 0) >= max_per_frame:
                    continue

                selected.append(q)
                frame_counts[q.frame_id] = frame_counts.get(q.frame_id, 0) + 1

                if len(selected) >= target:
                    break

        # Log type distribution
        logger.info(f"   Question type distribution after round-robin selection:")
        type_counts = {}
        for q in selected[:target]:
            type_counts[q.question_type] = type_counts.get(q.question_type, 0) + 1
        for qtype, count in sorted(type_counts.items()):
            logger.info(f"      {qtype}: {count}/{max_per_type}")

        logger.info(f"\n   ✓ Selected {len(selected[:target])}/{target} questions ({len(type_counts)} types)")

        return selected[:target]