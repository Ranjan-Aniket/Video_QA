"""
Pass 2A: Sonnet 4.5 - Easy/Medium Ontology Selection

Processes ~260 frames and identifies moments for 9 ontology types:
1. Needle (small details, OCR, graphics)
2. Counting (repeated elements)
3. Context (background, setting)
4. Referential (audio-visual sync)
5. Temporal (before/after)
6. Sequential (ordered events)
7. Comparative (differences)
8. ObjectInteraction (transformations)
9. Subscene (bounded segments)

Also attempts Inference and AVStitching, flagging hard cases for Opus 4.

Cost: ~$1.10 (Sonnet 4.5)
Output: 4-mode structured moments with cues and protected windows
"""

import json
import base64
from typing import Dict, List, Optional, Any
from pathlib import Path
from loguru import logger
import anthropic
import os
from dataclasses import dataclass, asdict
from processing.ontology_types import (
    normalize_type,
    OFFICIAL_TYPES,
    get_sub_task_type,
    get_min_temporal_window,
    MODE_DURATION_RANGES,
    MODE_PROTECTED_RADII
)
from processing.audio_validators import validate_audio_modality_diversity
from processing.moment_validators import (
    validate_temporal_window_for_type,
    validate_moment_duration,
    validate_cue_quality,
    validate_protected_radius
)


@dataclass
class Moment:
    """Represents a detected moment"""
    frame_ids: List[int]
    timestamps: List[float]
    mode: str  # "precise", "micro_temporal", "inference_window", "cluster"
    duration: float
    visual_cues: List[str]
    audio_cues: List[str]
    correspondence: str  # How audio and visual semantically align
    primary_ontology: str
    secondary_ontologies: List[str]
    adversarial_features: List[str]
    priority: float
    protected_window: Dict[str, float]  # {start, end, radius}
    frame_extraction: Dict  # {method, frames, anchor}
    confidence: float

    def to_dict(self):
        """Convert to dict for JSON serialization"""
        return asdict(self)


class Pass2ASonnetSelector:
    """
    Sonnet 4.5 selector for 9 easy/medium ontology types
    """

    def __init__(self):
        """Initialize Pass 2A selector"""
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = "claude-sonnet-4-5-20250929"

        # ✅ PROMPT CACHING: Build cacheable system prompt once (reused across ALL videos)
        self._cached_system_prompt = None  # Will be built on first use

        # Ontology definitions (✅ FIXED: Using official type names from ontology_types.py)
        self.ontology_definitions = {
            "Needle": "Small details - OCR text, brief graphics, specific numbers, badges, small objects",
            "Counting": "Repeated elements - count objects, count actions, count occurrences over time",
            "Context": "Background details - setting, environment, visual elements behind main action",
            "Referential Grounding": "Audio-visual sync - speech that directly references visible action (pronouns, pointing)",
            "Temporal Understanding": "Before/after - clear state changes, transitions, transformations",
            "Sequential": "Ordered events - 3+ events that have a specific order (first, then, finally)",
            "Comparative": "Differences - two states or elements to compare (X vs Y, bigger/smaller)",
            "Object Interaction Reasoning": "Transformations - object changes state due to action (open/close, fill/empty)",
            "Subscene": "Bounded segments - 5-15 second segments worth captioning as complete mini-stories"
        }

        # Mode definitions with duration guidelines (CORRECTED per Guidelines PDF)
        # ✅ P0 GAP #1 & #11 FIX: Align mode assignments with minimum temporal window requirements
        # Mode duration should accommodate the minimum temporal window needed for questions
        self.mode_definitions = {
            "precise": {
                "duration": "1-5s",
                "types": ["Needle", "Referential Grounding"],
                "description": "Exact timestamp critical, single frame or 2-3 frame burst",
                "min_windows": "10s (precise extraction, wide context window)"
            },
            "micro_temporal": {
                "duration": "3-8s",
                "types": ["Comparative", "Temporal Understanding", "Object Interaction Reasoning"],
                "description": "Short observations - quick comparisons, state changes, object transformations",
                "min_windows": "10s (short temporal window requirements)"
            },
            "inference_window": {
                "duration": "8-15s",
                "types": ["Inference", "General Holistic Reasoning", "Context", "Counting"],  # ✅ ADDED: Context (20s) and Counting (10s)
                "description": "Broader context needed, multi-step reasoning, scene understanding",
                "min_windows": "10-20s (medium temporal window requirements)"
            },
            "cluster": {
                "duration": "15-60s",
                "types": ["Sequential", "Subscene",
                         "Audio-Visual Stitching", "Tackling Spurious Correlations"],
                "description": "Extended continuous action, progression over time, temporal changes",
                "min_windows": "20-40s (high temporal window requirements)"
            }
        }

        # Frame content requirements by ontology type (P0 Critical Issue #2)
        # ✅ FIXED: Using official type names from ontology_types.py + added 7 missing types
        self.frame_requirements = {
            # OCR Recognition sub-task
            "Needle": {
                "check": lambda f: bool(f.get('text_detected') or f.get('ocr_text')),
                "error": "No visible text for Needle question"
            },
            "Referential Grounding": {  # ✅ FIXED: was "Referential"
                "check": lambda f: bool(f.get('text_detected') or f.get('ocr_text')),
                "error": "No visible text for Referential Grounding question"
            },

            # Counting sub-task
            "Counting": {
                "check": lambda f: len(f.get('objects', [])) >= 2,
                "error": "Need 2+ objects for Counting question"
            },

            # Human Behavior sub-task
            "Object Interaction Reasoning": {  # ✅ FIXED: was "ObjectInteraction"
                "check": lambda f: any('person' in str(o).lower() for o in f.get('objects', [])),
                "error": "No person detected for Object Interaction question"
            },

            # Scene Recognition sub-task
            "Context": {
                "check": lambda f: f.get('scene_type') not in [None, 'unknown', ''],
                "error": "No clear scene for Context question"
            },
            "Subscene": {
                "check": lambda f: f.get('scene_type') not in [None, 'unknown', ''],
                "error": "No clear scene for Subscene question"
            },

            # ✅ NEW: Temporal Understanding sub-task (requires visible change/state)
            "Temporal Understanding": {
                "check": lambda f: len(f.get('objects', [])) > 0 or bool(f.get('scene_type')),
                "error": "No visible content for Temporal Understanding question"
            },

            # ✅ NEW: Causal Reasoning sub-task
            "Sequential": {
                "check": lambda f: len(f.get('objects', [])) > 0 or bool(f.get('scene_type')),
                "error": "No visible content for Sequential question"
            },
            "Inference": {
                "check": lambda f: len(f.get('objects', [])) > 0 or any('person' in str(o).lower() for o in f.get('objects', [])),
                "error": "No visible action/object for Inference question"
            },

            # ✅ NEW: Holistic Reasoning sub-task
            "General Holistic Reasoning": {
                "check": lambda f: f.get('scene_type') not in [None, 'unknown', ''],
                "error": "No clear scene for General Holistic Reasoning question"
            },

            # ✅ NEW: Comparative sub-task (requires objects to compare)
            "Comparative": {
                "check": lambda f: len(f.get('objects', [])) >= 2,
                "error": "Need 2+ objects/elements for Comparative question"
            },

            # ✅ NEW: Audio-Visual Alignment sub-task (requires visual content)
            "Audio-Visual Stitching": {
                "check": lambda f: len(f.get('objects', [])) > 0 or bool(f.get('scene_type')),
                "error": "No visible content for Audio-Visual Stitching question"
            },
            "Tackling Spurious Correlations": {
                "check": lambda f: len(f.get('objects', [])) > 0 or bool(f.get('scene_type')),
                "error": "No visible content for Spurious Correlation question"
            },
        }

        # ✅ FIXED: Import from ontology_types.py (single source of truth)
        # Mode duration ranges (P1 High Issue #3)
        self.mode_duration_ranges = MODE_DURATION_RANGES

        # Protected window radii (P2 Medium Issue #6)
        self.protected_radii = MODE_PROTECTED_RADII

        # Hedging patterns (P1 High Issue #4)
        self.hedging_patterns = [
            r'\bappears?\s+to\b', r'\bseems?\s+to\b', r'\blooks?\s+like\b',
            r'\bcould\s+be\b', r'\bmay\s+be\b', r'\bmight\s+be\b',
            r'\bsuggests?\b', r'\blikely\b', r'\bprobably\b', r'\bpossibly\b'
        ]

        # Pronoun patterns (P1 High Issue #4)
        self.pronoun_patterns = [
            r'\bhe\b', r'\bshe\b', r'\bhim\b', r'\bher\b', r'\bhis\b', r'\bhers\b',
            r'\bthey\b', r'\bthem\b', r'\btheir\b', r'\btheirs\b'
        ]

        # Minimum temporal windows by question type (P0 Critical Bug Fix #2)
        # ✅ FIXED: Now importing from centralized ontology_types.py instead of defining locally

    def _build_cached_system_prompt(self) -> str:
        """
        Build CACHEABLE system prompt with all static guidelines.
        This will be reused across ALL videos (~90% of prompt tokens).

        Returns:
            System prompt string
        """
        return f"""You are analyzing video frames to identify moments for adversarial Q&A generation.

TASK: Identify moments for the following 9 ontology types:

{json.dumps(self.ontology_definitions, indent=2)}

ALSO ATTEMPT (flag if uncertain):
- Inference: Unstated cause/purpose/intent (flag if confidence < 0.7)
- AVStitching: Editing intent (flag if unclear)

=== 3-MODE FRAMEWORK ===

{json.dumps({k: v for k, v in self.mode_definitions.items() if k != 'micro_temporal'}, indent=2)}

=== PROTECTED WINDOWS ===

Each mode has protected windows to avoid overlap:
- Mode 1 (precise): ±5s radius
- Mode 2 (inference_window): ±2s radius
- Mode 3 (cluster): no protection (encompasses other frames)

=== OUTPUT FORMAT ===

For each detected moment, provide:

{{
  "frame_ids": [123, 124],
  "timestamps": [45.2, 45.7],
  "mode": "precise",  // precise | inference_window | cluster (micro_temporal reserved for future use)
  "duration": 2.5,
  "visual_cues": ["Text 'UNION LABEL' visible on shirt", "Speaker pointing at label"],
  "audio_cues": ["Speaker says 'look at the union label'"],
  "correspondence": "Speech directly references visible text and pointing gesture",
  "primary_ontology": "Needle",
  "secondary_ontologies": ["Referential"],
  "adversarial_features": [
    "Text partially occluded",
    "Easy to misread '9' as 'g'",
    "Requires matching speech timing to text visibility"
  ],
  "priority": 0.95,
  "protected_window": {{
    "start": 40.2,
    "end": 50.2,
    "radius": 5.0
  }},
  "frame_extraction": {{
    "method": "burst",
    "frames": [45.0, 45.2, 45.5],
    "anchor": 45.2
  }},
  "confidence": 0.95
}}

FRAME EXTRACTION METHODS:
- "burst": 2-3 frames around anchor (for precise moments)
- "uniform_sample": Evenly spaced frames in window
- "keyframe_sample": Key action points in window
- "adaptive_sample": Action-density weighted sampling

=== ADVERSARIAL CRITERIA ===

Each moment must:
1. Have TRUE audio-visual correspondence (speech semantically describes visual)
2. Require BOTH audio and visual to answer questions
3. NOT be answerable from single modality alone
4. Have specific adversarial features that make it challenging

=== CRITICAL CUE QUALITY GUIDELINES ===

❌ NO HEDGING LANGUAGE in visual_cues or audio_cues:
- FORBIDDEN: "appears to", "seems to", "looks like", "could be", "may be", "might be"
- FORBIDDEN: "suggests", "likely", "probably", "possibly"
- USE DEFINITIVE LANGUAGE: "shows", "displays", "contains", "indicates", "demonstrates"

Examples:
✗ BAD: "The person appears to be holding a red object"
✓ GOOD: "Person holds red mug"
✗ BAD: "Text seems to say 'UNION'"
✓ GOOD: "Text reads 'UNION'"

❌ NO PRONOUNS in visual_cues or audio_cues:
- FORBIDDEN: "he", "she", "him", "her", "his", "hers", "they", "them", "their"
- USE DESCRIPTORS: "person", "speaker", "individual", "figure", "woman", "man"

Examples:
✗ BAD: "He picks up the ball"
✓ GOOD: "Person picks up ball"
✗ BAD: "She points at the screen"
✓ GOOD: "Speaker points at screen"

❌ NO PROPER NAMES in visual_cues or audio_cues:
- FORBIDDEN: Person names, brand names, team names, character names
- USE GENERIC DESCRIPTORS: "person", "speaker", "brand logo", "team", "character"

Examples:
✗ BAD: "LeBron James scores"
✓ GOOD: "Player scores"
✗ BAD: "Nike logo visible"
✓ GOOD: "Brand logo visible"

═══════════════════════════════════════════════════════════════════════════════
MINIMUM TEMPORAL WINDOWS BY QUESTION TYPE (CRITICAL)
═══════════════════════════════════════════════════════════════════════════════

Each question type requires a MINIMUM temporal window (protected_window span):

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

⚠️ VALIDATION: protected_window.start to protected_window.end MUST span ≥ minimum for primary_ontology.

Example for Sequential:
✓ VALID: {{"start": 10.0, "end": 42.0}}  // 32s span ≥ 30s minimum
✗ INVALID: {{"start": 10.0, "end": 25.0}}  // 15s span < 30s minimum → REJECT

=== AUDIO MODALITY DIVERSITY ===

Selected moments should use at least 2 different audio modalities:
1. SPEECH: Dialogue, narration, words, phrases
2. MUSIC: Tempo, tone, melody, starts/stops
3. SOUND EFFECTS: Impacts, whooshes, mechanical sounds, clicks
4. SILENCE: Pauses, gaps, scene boundaries

This ensures audio-visual diversity and prevents speech-only moments.

=== HALLUCINATION PREVENTION ===

FORBIDDEN HALLUCINATIONS (will be rejected):
✗ NO inventing interactions that don't exist in the video
✗ NO inventing object states/orientations not visible in frames
✗ NO inventing progression/causality not shown
✗ NO inventing audio content not in transcript
✗ NO assuming actions before/after the visible window

ONLY describe what is DIRECTLY OBSERVABLE in:
- The provided frames (visual)
- The provided transcript segments (audio)

=== SPECIAL CASES ===

- If OCR text detected → ALWAYS create Mode 1 precise moment
- If counting opportunity (3+ identical objects) → Create Mode 1 moment
- If before/after pattern → Flag for Opus 4 (Comparative)
- If semantic mismatch → Flag for Opus 4 (Spurious)

Return JSON with moments and flagged_for_opus4 array."""

    def validate_frame_for_ontology(self, frame: dict, ontology_type: str) -> tuple[bool, str]:
        """
        Validate frame has required content for ontology type (P0 Critical Issue #2).

        Args:
            frame: Frame metadata dict
            ontology_type: Ontology type (e.g., "Needle", "Counting")

        Returns:
            (is_valid, error_message)
        """
        requirement = self.frame_requirements.get(ontology_type)

        if requirement is None:
            return True, ""  # No special requirements for this type

        try:
            if not requirement['check'](frame):
                return False, requirement['error']
        except Exception as e:
            logger.warning(f"Frame validation error for {ontology_type}: {e}")
            return True, ""  # Allow if check fails (graceful degradation)

        return True, ""

    def validate_temporal_window_for_type(self, moment: dict) -> tuple[bool, str]:
        """
        Validate temporal window meets minimum for question type (P0 Critical Bug Fix #2).

        Args:
            moment: Moment dict with protected_window and primary_ontology

        Returns:
            (is_valid, error_message)
        """
        # ✅ FIXED: Use shared validation module (eliminates duplication with Pass 2B)
        return validate_temporal_window_for_type(moment)

    def validate_moment_duration(self, moment: dict) -> tuple[bool, str]:
        """
        Validate moment duration matches mode requirements (P1 High Issue #3).

        Args:
            moment: Moment dict with 'mode' and 'duration' fields

        Returns:
            (is_valid, error_message)
        """
        # ✅ FIXED: Use shared validation module (eliminates duplication with Pass 2B)
        return validate_moment_duration(moment, self.mode_duration_ranges)

    def validate_cue_quality(self, moment: dict) -> tuple[bool, list[str]]:
        """
        Validate visual_cues and audio_cues for quality issues (P1 High Issue #4).

        Checks for:
        - Hedging language (appears, seems, possibly, etc.)
        - Pronouns (he, she, they, etc.)

        Args:
            moment: Moment dict with 'visual_cues' and 'audio_cues' fields

        Returns:
            (is_valid, list of issues found)
        """
        # ✅ FIXED: Use shared validation module (eliminates duplication with Pass 2B)
        return validate_cue_quality(moment, self.hedging_patterns, self.pronoun_patterns)

    def validate_protected_radius(self, moment: dict) -> tuple[bool, str]:
        """
        Validate protected window radius matches mode (P2 Medium Issue #6).

        Args:
            moment: Moment dict with 'mode' and 'protected_window' fields

        Returns:
            (is_valid, error_message)
        """
        # ✅ FIXED: Use shared validation module (eliminates duplication with Pass 2B)
        return validate_protected_radius(moment, self.protected_radii)

    def encode_frame(self, frame_path: str) -> str:
        """
        Encode frame to base64 for Claude Vision API

        Args:
            frame_path: Path to frame image

        Returns:
            Base64 encoded image
        """
        with open(frame_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def prepare_frames_for_api(
        self,
        selected_frames: List[Dict],
        frames_dir: Path
    ) -> List[Dict]:
        """
        Prepare frame images for Claude Vision API

        Args:
            selected_frames: Frame metadata from Pass 1
            frames_dir: Directory containing extracted frames

        Returns:
            List of dicts with frame metadata + base64 images
        """
        prepared_frames = []

        for frame in selected_frames:
            frame_id = frame['frame_id']
            timestamp = frame['timestamp']

            # ✅ FIX BUG-004: Handle both integer and string frame_ids
            # Integer frame_ids: format as frame_000123.jpg
            # String frame_ids: use as-is (e.g., "single_001.jpg")
            if isinstance(frame_id, int):
                frame_file = frames_dir / f"frame_{frame_id:06d}.jpg"
            else:
                # String frame_id - use directly (may already have .jpg extension)
                frame_file = frames_dir / (f"{frame_id}.jpg" if not frame_id.endswith('.jpg') else frame_id)

            if not frame_file.exists():
                logger.warning(f"Frame file not found: {frame_file}")
                continue

            # Encode frame
            frame_b64 = self.encode_frame(str(frame_file))

            prepared_frames.append({
                **frame,
                'image_base64': frame_b64
            })

        return prepared_frames

    def build_prompt(
        self,
        frames: List[Dict],
        audio_analysis: Dict,
        clip_analysis: Dict
    ) -> str:
        """
        Build DYNAMIC user prompt for Sonnet 4.5 (video-specific data only).
        Static guidelines are in system prompt with caching.

        Args:
            frames: Selected frames with metadata
            audio_analysis: Audio analysis results
            clip_analysis: CLIP analysis results

        Returns:
            Dynamic prompt string with video-specific data
        """
        prompt = f"""=== AUDIO TRANSCRIPT (with timestamps) ===

"""
        # Add transcript
        if 'segments' in audio_analysis:
            for seg in audio_analysis['segments'][:100]:  # Limit for context
                prompt += f"[{seg['start']:.1f}s] {seg.get('speaker', 'SPEAKER')}: {seg['text']}\n"

        prompt += f"""

=== AUDIO EVENTS (non-speech) ===

"""
        # Add audio events (music, sound effects, crowd sounds, music changes)
        audio_events = audio_analysis.get('audio_events', [])
        if audio_events:
            # Group by type for better organization
            music_events = [e for e in audio_events if e['type'] == 'background_music']
            sound_effects = [e for e in audio_events if e['type'] == 'sound_effect']
            crowd_sounds = [e for e in audio_events if e['type'] == 'crowd_sound']
            music_changes = [e for e in audio_events if e['type'] == 'music_change']

            if music_events:
                prompt += "MUSIC:\n"
                for event in music_events[:20]:  # Limit to avoid context bloat
                    prompt += f"  [{event['start']:.1f}s-{event['end']:.1f}s] {event['subtype']} music"
                    if 'characteristics' in event:
                        tempo = event['characteristics'].get('tempo')
                        if tempo:
                            prompt += f" (tempo: {tempo:.0f} BPM)"
                    prompt += "\n"

            if sound_effects:
                prompt += "SOUND EFFECTS:\n"
                for event in sound_effects[:20]:
                    prompt += f"  [{event['start']:.1f}s] {event['subtype']} (intensity: {event['characteristics'].get('intensity', 'unknown')})\n"

            if crowd_sounds:
                prompt += "CROWD SOUNDS:\n"
                for event in crowd_sounds[:10]:
                    prompt += f"  [{event['start']:.1f}s-{event['end']:.1f}s] {event['subtype']} (intensity: {event['characteristics'].get('intensity', 'unknown')})\n"

            if music_changes:
                prompt += "MUSIC CHANGES:\n"
                for event in music_changes[:10]:
                    before = event['characteristics'].get('before_tempo', 0)
                    after = event['characteristics'].get('after_tempo', 0)
                    prompt += f"  [{event['start']:.1f}s] {event['subtype']} ({before:.0f}→{after:.0f} BPM)\n"
        else:
            prompt += "(No non-speech audio events detected)\n"

        prompt += f"""

=== FRAME METADATA ===

Total frames in this batch: {len(frames)}

"""
        # Add frame metadata (images will be sent separately in content blocks)
        for i, frame in enumerate(frames):  # Process all frames in batch
            frame_id = frame['frame_id']
            ts = frame['timestamp']
            scene = frame.get('scene_type', 'unknown')
            ocr = frame.get('text_detected', '')
            obj_count = len(frame.get('objects', []))

            # ✅ ISSUE #1 FIX: Include CLIP ontology scores for moment prioritization
            ontology_scores = clip_analysis.get('ontology_scores', {}).get(frame_id, {})
            if ontology_scores:
                top_ontology = max(ontology_scores, key=ontology_scores.get)
                top_score = ontology_scores[top_ontology]
                clip_info = f", CLIP: {top_ontology}={top_score:.2f}"
            else:
                clip_info = ""

            prompt += f"Frame {i+1}: ID={frame_id}, time={ts:.1f}s, scene={scene}, objects={obj_count}"
            if ocr:
                prompt += f", OCR='{ocr[:50]}'"
            prompt += clip_info
            prompt += "\n"

        # Static guidelines are now in system prompt with caching
        return prompt

    def call_sonnet_45(
        self,
        frames: List[Dict],
        audio_analysis: Dict,
        clip_analysis: Dict
    ) -> Dict:
        """
        Call Sonnet 4.5 with vision to analyze frames

        Args:
            frames: Prepared frames with base64 images
            audio_analysis: Audio analysis
            clip_analysis: CLIP analysis

        Returns:
            Parsed response with moments and flags
        """
        logger.info(f"Calling Sonnet 4.5 with {len(frames)} frames...")

        # ✅ PROMPT CACHING: Build system prompt once (lazy init)
        if self._cached_system_prompt is None:
            self._cached_system_prompt = self._build_cached_system_prompt()

        # Build dynamic user prompt (video-specific data)
        dynamic_prompt = self.build_prompt(frames, audio_analysis, clip_analysis)

        # Build user content blocks (dynamic text + images)
        user_content = [{"type": "text", "text": dynamic_prompt}]

        # Add all frame images in this batch (batching handled by caller)
        for frame in frames:
            if 'image_base64' in frame:
                user_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": frame['image_base64']
                    }
                })

        try:
            # ✅ PROMPT CACHING: Use system messages with cache_control
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8000,
                system=[{
                    "type": "text",
                    "text": self._cached_system_prompt,
                    "cache_control": {"type": "ephemeral"}
                }],
                messages=[
                    {"role": "user", "content": user_content}
                ]
            )

            # Parse response
            response_text = response.content[0].text

            # Extract JSON
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                logger.error("No JSON found in Sonnet 4.5 response")
                return {"moments": [], "flagged_for_opus4": [], "coverage": {}}

            result = json.loads(json_match.group(0))

            # Track cost
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = (input_tokens * 3 / 1_000_000) + (output_tokens * 15 / 1_000_000)
            logger.info(f"Sonnet 4.5 cost: ${cost:.4f}")

            result['cost'] = cost
            result['tokens'] = {
                'input': input_tokens,
                'output': output_tokens
            }

            return result

        except Exception as e:
            logger.error(f"Sonnet 4.5 call failed: {e}")
            return {"moments": [], "flagged_for_opus4": [], "coverage": {}, "error": str(e)}

    def _batch_frames(self, frames: List[Dict], batch_size: int = 20):
        """
        Split frames into batches to avoid API limits

        Args:
            frames: List of frames to batch
            batch_size: Maximum frames per batch (default: 50)

        Yields:
            Batches of frames
        """
        for i in range(0, len(frames), batch_size):
            yield frames[i:i+batch_size]

    def _check_protected_window_overlap(self, new_moment: Dict[str, Any], existing_moments: List[Dict[str, Any]]) -> tuple[bool, Optional[str]]:
        """
        Check if new moment's protected window overlaps with existing moments

        Args:
            new_moment: Moment dict to validate
            existing_moments: List of already accepted moment dicts

        Returns:
            (is_valid, rejection_reason)
        """
        new_window = new_moment.get('protected_window', {})
        new_start = new_window.get('start')
        new_end = new_window.get('end')

        if new_start is None or new_end is None:
            # No protected window defined - accept
            return True, None

        new_duration = new_end - new_start
        new_mode = new_moment.get('mode')

        # ✅ GAP #7 FIX: Cluster moments CAN overlap with other modes,
        # but need minimum separation from OTHER cluster moments
        if new_mode == 'cluster':
            # Check only against other cluster moments
            for existing in existing_moments:
                if existing.get('mode') != 'cluster':
                    continue  # Cluster can overlap with non-cluster modes

                existing_window = existing.get('protected_window', {})
                existing_start = existing_window.get('start')
                existing_end = existing_window.get('end')

                if existing_start is None or existing_end is None:
                    continue

                # Calculate overlap percentage
                overlap_start = max(new_start, existing_start)
                overlap_end = min(new_end, existing_end)

                if overlap_start < overlap_end:
                    overlap_duration = overlap_end - overlap_start
                    existing_duration = existing_end - existing_start

                    # Calculate overlap as percentage of shorter moment
                    min_duration = min(new_duration, existing_duration)
                    overlap_percent = (overlap_duration / min_duration) * 100

                    # ✅ ALLOW up to 50% overlap, REJECT >50% overlap
                    # This prevents near-duplicate clusters while allowing partial overlap
                    if overlap_percent > 50.0:
                        existing_ontology = existing.get('primary_ontology', 'unknown')
                        return False, f"Cluster-to-cluster overlap too high ({overlap_percent:.0f}% > 50% max) with {existing_ontology} at [{existing_start:.1f}s - {existing_end:.1f}s]"

            # Passed all cluster-to-cluster checks
            return True, None

        # Non-cluster moments: check against all existing moments
        for existing in existing_moments:
            # Mode 4 existing moments don't block new non-cluster moments
            if existing.get('mode') == 'cluster':
                continue

            existing_window = existing.get('protected_window', {})
            existing_start = existing_window.get('start')
            existing_end = existing_window.get('end')

            if existing_start is None or existing_end is None:
                continue

            # Check for overlap: two windows overlap if one starts before the other ends
            overlaps = (new_start < existing_end) and (new_end > existing_start)

            if overlaps:
                existing_ontology = existing.get('primary_ontology', 'unknown')
                existing_mode = existing.get('mode', 'unknown')
                return False, f"Overlaps with {existing_ontology} ({existing_mode}) moment at [{existing_start:.1f}s - {existing_end:.1f}s]"

        return True, None

    def _validate_and_filter_moments(self, all_moments: List[Dict], selected_frames: List[Dict]) -> tuple[List[Dict], List[Dict]]:
        """
        Validate moments and filter out those with violations.

        Checks:
        - Protected window overlap
        - Mode duration matching requirements
        - Cue quality (hedging/pronouns)
        - Protected radius matching mode
        - Frame content matching ontology type

        Args:
            all_moments: List of all moment dicts from LLM
            selected_frames: Original frames with metadata for content validation

        Returns:
            (valid_moments, rejected_moments)
        """
        # Sort by priority (highest first) to keep best moments when conflicts arise
        sorted_moments = sorted(all_moments, key=lambda m: m.get('priority', 0.5), reverse=True)

        valid_moments = []
        rejected_moments = []

        # Create frame lookup for content validation
        frame_lookup = {f['frame_id']: f for f in selected_frames}

        validation_stats = {
            'protected_window': 0,
            'mode_duration': 0,
            'cue_quality': 0,
            'protected_radius': 0,
            'frame_content': 0,
            'temporal_window': 0,  # P0 Bug Fix #2
        }

        for moment in sorted_moments:
            rejection_reason = None

            # ✅ OPTIMIZATION: Fail-fast validation order (cheapest/most-likely-to-fail first)

            # ✅ Validation #1: Frame content for ontology type (CHEAPEST - just dict lookup)
            # Most likely to fail, so check first to avoid wasting time on other validations
            ontology_type = moment.get('primary_ontology', '')
            frame_ids = moment.get('frame_ids', [])

            # Check if ANY of the frames in this moment has required content
            frame_content_valid = False
            for frame_id in frame_ids:
                frame = frame_lookup.get(frame_id)
                if frame:
                    is_valid, reason = self.validate_frame_for_ontology(frame, ontology_type)
                    if is_valid:
                        frame_content_valid = True
                        break

            if not frame_content_valid and frame_ids:
                # Only reject if we have frames and none passed validation
                rejection_reason = f"Frame content: {reason}"
                validation_stats['frame_content'] += 1

            # ✅ Validation #2: Temporal window for question type (CHEAP - just math)
            if rejection_reason is None:
                is_valid, reason = self.validate_temporal_window_for_type(moment)
                if not is_valid:
                    rejection_reason = f"Temporal window: {reason}"
                    validation_stats['temporal_window'] += 1

            # ✅ Validation #3: Mode duration (CHEAP - simple math)
            if rejection_reason is None:
                is_valid, reason = self.validate_moment_duration(moment)
                if not is_valid:
                    rejection_reason = f"Mode duration: {reason}"
                    validation_stats['mode_duration'] += 1

            # ✅ Validation #4: Cue quality (MEDIUM - regex pattern matching)
            if rejection_reason is None:
                is_valid, issues = self.validate_cue_quality(moment)
                if not is_valid:
                    rejection_reason = f"Cue quality: {'; '.join(issues[:2])}"  # Show first 2 issues
                    validation_stats['cue_quality'] += 1

            # ✅ Validation #5: Protected radius (MEDIUM - math calculation)
            if rejection_reason is None:
                is_valid, reason = self.validate_protected_radius(moment)
                if not is_valid:
                    rejection_reason = f"Protected radius: {reason}"
                    validation_stats['protected_radius'] += 1

            # ✅ Validation #6: Protected window overlap (MOST EXPENSIVE - iterates all valid_moments)
            # Check last since it's most expensive and least likely to fail
            if rejection_reason is None:
                is_valid, reason = self._check_protected_window_overlap(moment, valid_moments)
                if not is_valid:
                    rejection_reason = f"Protected window: {reason}"
                    validation_stats['protected_window'] += 1

            # Accept or reject based on all validations
            if rejection_reason is None:
                # ✅ P1 #6: Assign sub-task type based on ontology
                ontology_type = moment.get('primary_ontology', '')
                if ontology_type:
                    moment['sub_task_type'] = get_sub_task_type(ontology_type)
                valid_moments.append(moment)
            else:
                logger.warning(f"Rejected moment (ontology={moment.get('primary_ontology')}, "
                              f"timestamp={moment.get('timestamps', [0])[0]:.1f}s): {rejection_reason}")
                rejected_moments.append({
                    'moment': moment,
                    'rejection_reason': rejection_reason
                })

        # ✅ GAP #4 FIX: Validate audio modality diversity across all moments (batch-level check)
        if valid_moments:
            # Convert moments to expected format (list of dicts with 'audio_cue' field)
            moments_for_audio_check = []
            for moment in valid_moments:
                audio_cues_list = moment.get('audio_cues', [])
                # Join all audio cues for this moment
                combined_audio = ' '.join(audio_cues_list) if audio_cues_list else ''
                if combined_audio:  # Only include moments with audio cues
                    moments_for_audio_check.append({
                        'audio_cue': combined_audio,
                        'moment_id': f"moment_{moment.get('timestamps', [0])[0]:.1f}s"
                    })

            # Validate audio modality diversity
            if moments_for_audio_check:
                audio_diversity_valid, diversity_warnings = validate_audio_modality_diversity(
                    moments_for_audio_check, min_modalities=2
                )
                if not audio_diversity_valid:
                    logger.warning(f"⚠️  AUDIO DIVERSITY: Insufficient audio modality diversity in Pass 2A moments")
                    for warning in diversity_warnings:
                        logger.warning(f"  {warning}")
                else:
                    logger.info(f"✅ Audio modality diversity check passed ({len(moments_for_audio_check)} moments checked)")

        # Log validation statistics
        logger.info(f"Moment validation: {len(valid_moments)} valid, {len(rejected_moments)} rejected")
        logger.info(f"  Rejection breakdown: protected_window={validation_stats['protected_window']}, "
                   f"duration={validation_stats['mode_duration']}, "
                   f"cue_quality={validation_stats['cue_quality']}, "
                   f"radius={validation_stats['protected_radius']}, "
                   f"frame_content={validation_stats['frame_content']}")

        return valid_moments, rejected_moments

    def process_frames(
        self,
        selected_frames: List[Dict],
        audio_analysis: Dict,
        clip_analysis: Dict,
        frames_dir: Path
    ) -> Dict:
        """
        Run full Pass 2A processing

        Args:
            selected_frames: Frames from Pass 1 (~260 frames)
            audio_analysis: Audio analysis
            clip_analysis: CLIP analysis
            frames_dir: Directory with extracted frame images

        Returns:
            {
                'mode1_precise': [...],
                'mode2_micro_temporal': [...],
                'mode3_inference_window': [...],
                'mode4_clusters': [...],
                'flagged_for_opus4': [...],
                'coverage': {...},
                'cost': float
            }
        """
        logger.info("=" * 60)
        logger.info("PASS 2A: Sonnet 4.5 Easy/Medium Ontology Selection")
        logger.info("=" * 60)

        # Prepare frames with images
        prepared_frames = self.prepare_frames_for_api(selected_frames, frames_dir)

        if not prepared_frames:
            logger.error("No frames prepared for API call")
            return {
                'mode1_precise': [],
                'mode2_micro_temporal': [],
                'mode3_inference_window': [],
                'mode4_clusters': [],
                'flagged_for_opus4': [],
                'coverage': {},
                'cost': 0
            }

        # Batch frames to avoid API limits (reduced to 20 to prevent timeouts)
        batches = list(self._batch_frames(prepared_frames, batch_size=20))
        logger.info(f"Processing {len(prepared_frames)} frames in {len(batches)} batches...")

        # Call Sonnet 4.5 for each batch and merge results
        all_moments = []
        all_flagged = []
        total_cost = 0.0
        coverage_totals = {}

        for batch_idx, batch in enumerate(batches):
            logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} ({len(batch)} frames)...")
            response = self.call_sonnet_45(batch, audio_analysis, clip_analysis)

            # Merge moments
            all_moments.extend(response.get('moments', []))
            all_flagged.extend(response.get('flagged_for_opus4', []))

            # Sum costs
            total_cost += response.get('cost', 0)

            # Merge coverage counts
            for ontology, count in response.get('coverage', {}).items():
                coverage_totals[ontology] = coverage_totals.get(ontology, 0) + count

        logger.info(f"Total Sonnet 4.5 cost across {len(batches)} batches: ${total_cost:.4f}")

        # ✅ CRITICAL GAP FIX: Enforce Sequential+Temporal rule BEFORE validation
        # Guidelines: "Sequential always go hand in hand with Temporal Understanding"
        # This must happen before validation to ensure moments aren't rejected due to missing Temporal Understanding
        for moment in all_moments:
            # Normalize first
            if 'primary_ontology' in moment:
                moment['primary_ontology'] = normalize_type(moment['primary_ontology'])
            if 'secondary_ontologies' in moment:
                moment['secondary_ontologies'] = [normalize_type(t) for t in moment['secondary_ontologies']]

            # Enforce co-occurrence rule
            primary = moment.get('primary_ontology', '')
            secondaries = moment.get('secondary_ontologies', [])
            all_types = [primary] + secondaries
            if "Sequential" in all_types and "Temporal Understanding" not in all_types:
                if 'secondary_ontologies' not in moment:
                    moment['secondary_ontologies'] = []
                moment['secondary_ontologies'].append("Temporal Understanding")
                logger.debug(f"Enforcing Sequential+Temporal rule (pre-validation): Added Temporal Understanding to moment at {moment.get('timestamps', [0])[0]:.1f}s")

        # Validate protected windows and filter overlapping moments
        valid_moments, rejected_moments = self._validate_and_filter_moments(all_moments, selected_frames)

        # ✅ FIXED: Recalculate coverage from VALID moments (not raw LLM responses)
        # This ensures coverage reflects actual output after validation/rejection
        coverage_actual = {}
        for moment in valid_moments:
            primary = moment.get('primary_ontology', 'Unknown')
            coverage_actual[primary] = coverage_actual.get(primary, 0) + 1

        # Log discrepancy if validation rejected many moments
        total_raw = sum(coverage_totals.values())
        total_actual = sum(coverage_actual.values())
        if total_raw > total_actual:
            logger.info(f"   Coverage adjusted: {total_raw} raw moments → {total_actual} valid moments "
                       f"({total_raw - total_actual} rejected)")

        # Create merged response with validated moments
        response = {
            'moments': valid_moments,
            'rejected_moments': rejected_moments,
            'flagged_for_opus4': all_flagged,
            'coverage': coverage_actual,  # ✅ Use actual coverage, not raw LLM counts
            'cost': total_cost
        }

        # Organize moments by mode
        mode1_moments = []
        mode2_moments = []
        mode3_moments = []
        mode4_moments = []

        for moment_data in response.get('moments', []):
            mode = moment_data.get('mode', 'precise')

            # NOTE: Normalization and Sequential+Temporal rule enforcement already done at lines 996-1014
            # Just extract the already-normalized values
            primary_ontology = moment_data.get('primary_ontology', '')
            secondary_ontologies = moment_data.get('secondary_ontologies', [])

            # Convert to Moment dataclass
            moment = Moment(
                frame_ids=moment_data.get('frame_ids', []),
                timestamps=moment_data.get('timestamps', []),
                mode=mode,
                duration=moment_data.get('duration', 0),
                visual_cues=moment_data.get('visual_cues', []),
                audio_cues=moment_data.get('audio_cues', []),
                correspondence=moment_data.get('correspondence', ''),
                primary_ontology=primary_ontology,  # Already normalized
                secondary_ontologies=secondary_ontologies,  # Already normalized and rule-enforced
                adversarial_features=moment_data.get('adversarial_features', []),
                priority=moment_data.get('priority', 0.5),
                protected_window=moment_data.get('protected_window', {}),
                frame_extraction=moment_data.get('frame_extraction', {}),
                confidence=moment_data.get('confidence', 0.5)
            )

            # Categorize by mode
            if mode == "precise":
                mode1_moments.append(moment)
            elif mode == "micro_temporal":
                mode2_moments.append(moment)
            elif mode == "inference_window":
                mode3_moments.append(moment)
            elif mode == "cluster":
                mode4_moments.append(moment)

        logger.info("=" * 60)
        logger.info(f"Pass 2A Complete:")
        logger.info(f"  Mode 1 (precise): {len(mode1_moments)}")
        logger.info(f"  Mode 2 (micro_temporal): {len(mode2_moments)}")
        logger.info(f"  Mode 3 (inference_window): {len(mode3_moments)}")
        logger.info(f"  Mode 4 (cluster): {len(mode4_moments)}")
        logger.info(f"  Flagged for Opus 4: {len(response.get('flagged_for_opus4', []))}")
        logger.info(f"  Total moments: {len(response.get('moments', []))}")
        logger.info("=" * 60)

        return {
            'mode1_precise': [m.to_dict() for m in mode1_moments],
            'mode2_micro_temporal': [m.to_dict() for m in mode2_moments],
            'mode3_inference_window': [m.to_dict() for m in mode3_moments],
            'mode4_clusters': [m.to_dict() for m in mode4_moments],
            'flagged_for_opus4': response.get('flagged_for_opus4', []),
            'coverage': response.get('coverage', {}),
            'cost': response.get('cost', 0)
        }


def run_pass2a_selection(
    selected_frames: List[Dict],
    audio_analysis: Dict,
    clip_analysis: Dict,
    frames_dir: str,
    output_path: str
) -> Dict:
    """
    Run Pass 2A and save results

    Args:
        selected_frames: Frames from Pass 1
        audio_analysis: Audio analysis
        clip_analysis: CLIP analysis
        frames_dir: Directory with frame images
        output_path: Path to save results

    Returns:
        Pass 2A results
    """
    selector = Pass2ASonnetSelector()

    results = selector.process_frames(
        selected_frames,
        audio_analysis,
        clip_analysis,
        Path(frames_dir)
    )

    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Pass 2A results saved to {output_path}")

    return results
