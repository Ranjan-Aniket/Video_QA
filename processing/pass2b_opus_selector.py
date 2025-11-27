"""
Pass 2B: Opus 4 - Hard Ontology Types + Spurious Detection

Processes ~30 frames (flagged by Pass 2A + CLIP spurious) for 4 hard types:
1. Inference (unstated cause/purpose/intent)
2. Holistic (full-video patterns, recurring elements)
3. AVStitching (editing intent, why cuts were made)
4. Spurious (counter-intuitive pairings, audio-visual mismatch)

Uses Opus 4 for superior deep multimodal reasoning.

Cost: ~$1.00 (Opus 4)
Output: 4-mode structured moments (same format as Pass 2A)
"""

import json
import base64
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
from loguru import logger
import anthropic
import os
from processing.ontology_types import (
    get_sub_task_type,
    normalize_type,
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


class Pass2BOpusSelector:
    """
    Opus 4 selector for 4 hard ontology types requiring deep reasoning
    """

    def __init__(self):
        """Initialize Pass 2B selector"""
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = "claude-opus-4-20250514"

        # ✅ FIXED: Hard ontology definitions (using official type names from ontology_types.py)
        self.ontology_definitions = {
            "Inference": """
Unstated cause, purpose, or intent that must be inferred from visual evidence.
- WHY did someone do X?
- What is the PURPOSE of this action?
- What CAUSED this outcome?
- NOT explicitly stated in audio, must be reasoned from visuals
""",
            "General Holistic Reasoning": """
Full-video patterns and recurring elements across the entire video.
- Elements that appear multiple times throughout video
- Structural patterns (e.g., recurring format, repeated transitions)
- Themes or motifs that span the full video
- Requires synthesizing information from distant parts of video
""",
            "Audio-Visual Stitching": """
Editing intent and how spliced clips relate to each other.
- WHY was this cut made here?
- How do scenes before/after cut relate semantically?
- What is the editor trying to convey with this transition?
- Contrast between consecutive scenes
""",
            "Tackling Spurious Correlations": """
Counter-intuitive pairings where obvious interpretation is WRONG.
- Audio suggests one thing, visual shows something completely different
- Obvious reading leads to wrong conclusion
- Misleading coincidences or false associations
- Hidden meanings that contradict surface interpretation
"""
        }

        # ✅ FIXED: Import from ontology_types.py (single source of truth)
        # ✅ P1/P2 Validation: Mode duration ranges (Guidelines PDF compliance)
        self.mode_duration_ranges = MODE_DURATION_RANGES

        # ✅ P2 Validation: Protected window radii by mode
        self.protected_radii = MODE_PROTECTED_RADII

        # ✅ P1 Validation: Hedging language patterns (Guidelines: avoid hedging)
        self.hedging_patterns = [
            r'\bappears?\s+to\b', r'\bseems?\s+to\b', r'\blooks?\s+like\b',
            r'\bcould\s+be\b', r'\bmay\s+be\b', r'\bmight\s+be\b',
            r'\bsuggests?\b', r'\blikely\b', r'\bprobably\b', r'\bpossibly\b'
        ]

        # ✅ P1 Validation: Pronoun patterns (Guidelines: no pronouns without context)
        self.pronoun_patterns = [
            r'\bhe\b', r'\bshe\b', r'\bhim\b', r'\bher\b', r'\bhis\b', r'\bhers\b',
            r'\bthey\b', r'\bthem\b', r'\btheir\b', r'\btheirs\b'
        ]

        # Minimum temporal windows by question type (P0 Critical Bug Fix #3)
        # ✅ FIXED: Now importing from centralized ontology_types.py instead of defining locally

        # ✅ FIXED: Frame requirements by ontology type (using official names)
        self.frame_requirements = {
            "Inference": {
                "check": lambda f: len(f.get('objects', [])) >= 1,
                "error": "Need visible objects for Inference question"
            },
            "General Holistic Reasoning": {
                "check": lambda f: True,  # General Holistic Reasoning can work with any frame
                "error": ""
            },
            "Audio-Visual Stitching": {
                "check": lambda f: True,  # Audio-Visual Stitching focuses on audio-visual relationship
                "error": ""
            },
            "Tackling Spurious Correlations": {
                "check": lambda f: len(f.get('objects', [])) >= 1,
                "error": "Need visible objects for Tackling Spurious Correlations question"
            },
        }

    def encode_frame(self, frame_path: str) -> str:
        """Encode frame to base64"""
        with open(frame_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def validate_frame_for_ontology(self, frame: dict, ontology_type: str) -> tuple[bool, str]:
        """
        ✅ P0 #3: Validate frame has required content for ontology type

        Per Guidelines PDF, frames must contain relevant visual elements for the question type.
        """
        requirement = self.frame_requirements.get(ontology_type)
        if requirement is None:
            return True, ""

        try:
            if not requirement['check'](frame):
                return False, requirement['error']
        except Exception as e:
            logger.warning(f"Frame validation error for {ontology_type}: {e}")
            return True, ""  # Graceful degradation

        return True, ""

    def validate_temporal_window_for_type(self, moment: dict) -> tuple[bool, str]:
        """
        Validate temporal window meets minimum for question type (P0 Critical Bug Fix #3).

        Args:
            moment: Moment dict with protected_window and primary_ontology

        Returns:
            (is_valid, error_message)
        """
        # ✅ FIXED: Use shared validation module (eliminates duplication with Pass 2A)
        return validate_temporal_window_for_type(moment)

    def validate_moment_duration(self, moment: dict) -> tuple[bool, str]:
        """
        ✅ P1 #4: Validate moment duration matches mode requirements

        Per Guidelines PDF:
        - precise: 1-5s
        - micro_temporal: 3-8s
        - inference_window: 8-15s
        - cluster: 15-60s
        """
        # ✅ FIXED: Use shared validation module (eliminates duplication with Pass 2A)
        return validate_moment_duration(moment, self.mode_duration_ranges)

    def validate_cue_quality(self, moment: dict) -> tuple[bool, list[str]]:
        """
        ✅ P1 #5: Validate visual_cues and audio_cues for quality issues

        Per Guidelines PDF:
        - No hedging language (appears, seems, likely, probably)
        - No pronouns without clear context (he, she, they)
        """
        # ✅ FIXED: Use shared validation module (eliminates duplication with Pass 2A)
        return validate_cue_quality(moment, self.hedging_patterns, self.pronoun_patterns)

    def validate_protected_radius(self, moment: dict) -> tuple[bool, str]:
        """
        ✅ P2 #7: Validate protected window radius matches mode

        Per Guidelines PDF:
        - precise: ±5s
        - micro_temporal: ±3s
        - inference_window: ±2s
        - cluster: none
        """
        # ✅ FIXED: Use shared validation module (eliminates duplication with Pass 2A)
        return validate_protected_radius(moment, self.protected_radii)

    def _retry_api_call(self, api_func, max_retries: int = 3, initial_delay: float = 1.0):
        """
        ✅ FIX: Add retry logic with exponential backoff for API calls

        Args:
            api_func: Function that makes the API call (should return response)
            max_retries: Maximum number of retry attempts (default: 3)
            initial_delay: Initial delay in seconds before first retry (default: 1.0)

        Returns:
            API response

        Raises:
            Exception: If all retries fail
        """
        last_exception = None
        delay = initial_delay

        for attempt in range(max_retries):
            try:
                return api_func()
            except anthropic.RateLimitError as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries}), retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Rate limit exceeded after {max_retries} attempts")
            except anthropic.APIError as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(f"API error (attempt {attempt + 1}/{max_retries}): {e}, retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.error(f"API error after {max_retries} attempts: {e}")
            except Exception as e:
                # For other exceptions, don't retry
                logger.error(f"Non-retryable error: {e}")
                raise

        # If we get here, all retries failed
        raise last_exception

    def _extract_spurious_candidate_data(self, sc) -> dict:
        """
        ✅ FIXED: Safely extract SpuriousCandidate data with validation.
        Handles both dataclass and dict formats with proper field validation.

        Args:
            sc: SpuriousCandidate (dataclass or dict)

        Returns:
            Dict with validated fields: {frame_id, timestamp, mismatch_score, text_segment, reason}

        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Try dataclass first (expected format)
        if hasattr(sc, 'frame_id'):
            # Validate required fields exist and are not None
            if not hasattr(sc, 'timestamp') or sc.timestamp is None:
                raise ValueError(f"SpuriousCandidate missing required field 'timestamp'")
            if not hasattr(sc, 'reason') or sc.reason is None:
                raise ValueError(f"SpuriousCandidate missing required field 'reason'")

            return {
                'frame_id': sc.frame_id,
                'timestamp': sc.timestamp,
                'mismatch_score': getattr(sc, 'mismatch_score', 0.0),
                'text_segment': getattr(sc, 'text_segment', ''),
                'reason': sc.reason
            }

        # Fallback to dict format (for backward compatibility)
        elif isinstance(sc, dict):
            frame_id = sc.get('frame_id')
            if frame_id is None:
                # Try alternate field name
                frame_ids = sc.get('frame_ids', [])
                frame_id = frame_ids[0] if frame_ids else None

            if frame_id is None:
                raise ValueError("SpuriousCandidate dict missing 'frame_id' field")

            timestamp = sc.get('timestamp')
            if timestamp is None:
                raise ValueError("SpuriousCandidate dict missing 'timestamp' field")

            reason = sc.get('reason')
            if not reason:
                raise ValueError("SpuriousCandidate dict missing 'reason' field")

            return {
                'frame_id': frame_id,
                'timestamp': timestamp,
                'mismatch_score': sc.get('mismatch_score', 0.0),
                'text_segment': sc.get('text_segment', ''),
                'reason': reason
            }

        else:
            raise ValueError(f"Invalid SpuriousCandidate type: {type(sc)}")

    def prepare_frames(
        self,
        flagged_frames: List[Dict],
        spurious_candidates: List[Dict],
        frames_dir: Path,
        all_frames_metadata: List[Dict]
    ) -> List[Dict]:
        """
        Prepare frames for Opus 4 analysis

        Args:
            flagged_frames: Frames flagged by Pass 2A
            spurious_candidates: CLIP spurious candidates
            frames_dir: Directory with frame images
            all_frames_metadata: All frame metadata for context

        Returns:
            Prepared frames with images
        """
        # Combine flagged + spurious
        frame_ids_to_process = set()

        for f in flagged_frames:
            frame_ids_to_process.update(f.get('frame_ids', []))

        for sc in spurious_candidates:
            # ✅ FIXED: Use validated extraction method
            try:
                sc_data = self._extract_spurious_candidate_data(sc)
                frame_ids_to_process.add(sc_data['frame_id'])
            except ValueError as e:
                logger.warning(f"Skipping invalid SpuriousCandidate: {e}")

        # Get metadata for these frames
        frames_to_process = [
            f for f in all_frames_metadata
            if f['frame_id'] in frame_ids_to_process
        ]

        # Encode frames
        prepared = []
        for frame in frames_to_process:
            frame_id = frame['frame_id']
            frame_file = frames_dir / f"frame_{frame_id:06d}.jpg"

            if frame_file.exists():
                frame_b64 = self.encode_frame(str(frame_file))
                prepared.append({
                    **frame,
                    'image_base64': frame_b64
                })
            else:
                logger.warning(f"Frame file not found: {frame_file}")

        logger.info(f"Prepared {len(prepared)} frames for Opus 4 analysis")

        return prepared

    def build_prompt(
        self,
        frames: List[Dict],
        flagged_frames: List[Dict],
        spurious_candidates: List[Dict],
        audio_analysis: Dict,
        full_video_context: Dict
    ) -> str:
        """
        Build comprehensive prompt for Opus 4

        Args:
            frames: Prepared frames
            flagged_frames: Frames flagged by Pass 2A
            spurious_candidates: CLIP spurious candidates
            audio_analysis: Audio analysis
            full_video_context: Context from earlier phases (scenes, themes, etc.)

        Returns:
            Prompt string
        """
        prompt = f"""You are performing deep multimodal reasoning to identify challenging adversarial moments.

TASK: Analyze frames for 4 HARD ontology types requiring sophisticated reasoning:

{json.dumps(self.ontology_definitions, indent=2)}

=== FRAMES FLAGGED BY PASS 2A ===

Sonnet 4.5 flagged these for deeper analysis:

"""
        for ff in flagged_frames:
            prompt += f"- Frame IDs {ff.get('frame_ids')}: {ff.get('reason')}\n"
            prompt += f"  Suggested ontology: {ff.get('suggested_ontology')}\n\n"

        prompt += f"""

=== SPURIOUS CANDIDATES (CLIP FLAGGED) ===

CLIP detected potential semantic mismatches:

"""
        for sc in spurious_candidates[:15]:
            # ✅ FIXED: Use validated extraction method
            try:
                sc_data = self._extract_spurious_candidate_data(sc)
                prompt += f"- Frame {sc_data['frame_id']} @ {sc_data['timestamp']}s\n"
                prompt += f"  Mismatch score: {sc_data['mismatch_score']:.2f}\n"
                prompt += f"  Text: \"{sc_data['text_segment']}\"\n"
                prompt += f"  Reason: {sc_data['reason']}\n\n"
            except ValueError as e:
                logger.warning(f"Skipping invalid SpuriousCandidate in prompt: {e}")

        prompt += f"""

=== FULL VIDEO TRANSCRIPT ===

"""
        if 'segments' in audio_analysis:
            for seg in audio_analysis['segments'][:150]:
                prompt += f"[{seg['start']:.1f}s] {seg.get('speaker', 'SPEAKER')}: {seg['text']}\n"

        prompt += f"""

=== FRAME DETAILS ===

"""
        for i, frame in enumerate(frames):
            frame_id = frame['frame_id']
            ts = frame['timestamp']
            scene = frame.get('scene_type', 'unknown')
            ocr = frame.get('text_detected', '')
            obj_count = len(frame.get('objects', []))

            prompt += f"Frame {i+1}: ID={frame_id}, time={ts:.1f}s, scene={scene}, objects={obj_count}"
            if ocr:
                prompt += f", OCR='{ocr[:50]}'"
            prompt += "\n"

        prompt += f"""

=== DEEP REASONING INSTRUCTIONS ===

For INFERENCE moments:
- Look for actions where the WHY is not stated in audio
- Identify visual cues that reveal purpose/intent/causality
- Consider: Would a naive viewer understand why this happened?
- The ANSWER should be inferable from visuals but not obvious

For HOLISTIC moments:
- Identify patterns that span the full video
- Look for recurring visual elements or themes
- Consider structural patterns (e.g., format, transitions)
- Requires synthesizing information from distant parts

For AVSTITCHING moments:
- Analyze scene transitions and cuts
- Ask: Why did the editor cut HERE?
- How do scenes before/after relate semantically?
- What does this edit convey beyond individual scenes?

═══════════════════════════════════════════════════════════════
SPURIOUS CORRELATION: CONCRETE DETECTION CRITERIA
═══════════════════════════════════════════════════════════════

Flag as SPURIOUS if moment meets ANY of these criteria:

1. SEMANTIC MISMATCH:
   • Audio says "happy" but visual shows sad expression
   • Audio describes "fast" but visual shows slow movement
   • Audio references object X but visual shows object Y
   Example: "Audio mentions 'red ball' but only blue ball visible"

2. MISLEADING COINCIDENCE:
   • Audio event aligns with unrelated visual event by chance
   • Timing suggests causation but there's none
   • Similar sounds from different sources confuse relationship
   Example: "Door slam audio coincides with unrelated hand gesture"

3. EXPECTATION VIOLATION:
   • Normal interpretation leads to wrong conclusion
   • Context suggests one thing, reality is another
   • Common assumption would be incorrect
   Example: "Audio says 'turn left' but person turns right"

CONFIDENCE THRESHOLD:
• < 0.75: Too uncertain, DO NOT flag as spurious
• 0.75-0.85: Flag with detailed reasoning
• > 0.85: Strong spurious candidate, prioritize

SPURIOUS REQUIRES:
- Clear evidence of mismatch or misleading relationship
- High confidence (≥ 0.75) that it will trick Gemini
- Specific explanation of WHY it's spurious

═══════════════════════════════════════════════════════════════
AUDIO-VISUAL STITCHING: 4-STEP DECISION TREE
═══════════════════════════════════════════════════════════════

Ask these questions IN ORDER. Stop at first "NO":

┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Does audio provide SEMANTIC anchor (not just timing)? │
├─────────────────────────────────────────────────────────────┤
│ ✓ SEMANTIC: "narrator says 'watch the ball'" (describes)  │
│ ✗ TEMPORAL: "audio peak at 2:30" (timing only)            │
│                                                             │
│ If NO → NOT AV Stitching (stop here)                       │
└─────────────────────────────────────────────────────────────┘
                           ↓ YES
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Does visual show what audio REFERENCES?            │
├─────────────────────────────────────────────────────────────┤
│ ✓ MATCH: Audio "red car" + Visual shows red car            │
│ ✗ MISMATCH: Audio "red car" + Visual shows blue car        │
│                                                             │
│ If NO → NOT AV Stitching (stop here)                       │
└─────────────────────────────────────────────────────────────┘
                           ↓ YES
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Would answering require BOTH modalities?           │
├─────────────────────────────────────────────────────────────┤
│ • Can answer with visual alone? → NOT AV Stitching         │
│ • Can answer with audio alone? → NOT AV Stitching          │
│                                                             │
│ If either YES → NOT AV Stitching (stop here)               │
└─────────────────────────────────────────────────────────────┘
                           ↓ NO (both needed)
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: Is connection MEANING-BASED (not coincidental)?    │
├─────────────────────────────────────────────────────────────┤
│ ✓ MEANING: Speech describes visible action                 │
│ ✗ COINCIDENCE: Music happens during unrelated action       │
│                                                             │
│ If NO → NOT AV Stitching (stop here)                       │
└─────────────────────────────────────────────────────────────┘
                           ↓ YES
                    ✅ ASSIGN AV STITCHING

=== CRITICAL CUE QUALITY GUIDELINES ===

❌ NO HEDGING LANGUAGE in visual_cues or audio_cues:
- FORBIDDEN: "appears to", "seems to", "looks like", "could be", "may be", "might be"
- FORBIDDEN: "suggests", "likely", "probably", "possibly"
- USE DEFINITIVE LANGUAGE: "shows", "displays", "contains", "indicates", "demonstrates"

Examples:
✗ BAD: "The action appears to cause an effect"
✓ GOOD: "Action causes effect"
✗ BAD: "Speaker seems uncertain"
✓ GOOD: "Speaker pauses before answering"

❌ NO PRONOUNS in visual_cues or audio_cues:
- FORBIDDEN: "he", "she", "him", "her", "his", "hers", "they", "them", "their"
- USE DESCRIPTORS: "person", "speaker", "individual", "figure", "woman", "man"

❌ NO PROPER NAMES in visual_cues or audio_cues:
- FORBIDDEN: Person names, brand names, team names, character names
- USE GENERIC DESCRIPTORS: "person", "speaker", "brand logo", "team", "character"

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

Example for Audio-Visual Stitching:
✓ VALID: {{"start": 5.0, "end": 28.0}}  // 23s span ≥ 20s minimum
✗ INVALID: {{"start": 5.0, "end": 18.0}}  // 13s span < 20s minimum → REJECT

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
✗ NO inferring emotions/intentions without visual evidence

ONLY describe what is DIRECTLY OBSERVABLE in:
- The provided frames (visual)
- The provided transcript segments (audio)

For INFERENCE type specifically:
- The QUESTION should ask about unstated WHY/PURPOSE
- The ANSWER must be inferable from visible evidence
- But DON'T hallucinate details not present in frames

=== OUTPUT FORMAT ===

For each detected moment, provide:

{{
  "frame_ids": [123, 124],
  "timestamps": [45.2, 45.7],
  "mode": "inference_window",  // or cluster
  "duration": 10.0,
  "visual_cues": ["Person picks up hammer", "Examines nail", "Sets hammer down without using it"],
  "audio_cues": ["Speaker says 'I realized something was wrong'"],
  "correspondence": "Audio provides context but doesn't explain WHY hammer wasn't used - must infer from visual sequence",
  "primary_ontology": "Inference",
  "secondary_ontologies": ["Temporal"],
  "adversarial_features": [
    "Requires inferring unstated reason (nail already hammered)",
    "Audio provides misdirection (focuses on realization, not the action)",
    "Easy to miss visual detail (nail state)"
  ],
  "priority": 0.90,
  "protected_window": {{
    "start": 43.2,
    "end": 47.2,
    "radius": 2.0
  }},
  "frame_extraction": {{
    "method": "keyframe_sample",
    "frames": [44.0, 45.2, 46.5],
    "anchor": 45.2
  }},
  "confidence": 0.85,
  "reasoning": "Detailed explanation of why this is adversarial and what makes it challenging"
}}

=== MODE GUIDELINES FOR HARD TYPES ===

- Inference: Usually "inference_window" (8-12s) - need setup + action
- Holistic: Always "cluster" (20-30s) - spans multiple scenes
- AVStitching: "inference_window" or "cluster" depending on cut span
- Spurious: Usually "inference_window" (8-12s) - need context for mismatch

=== QUALITY OVER QUANTITY ===

- Generate 8-15 moments total (focus on truly challenging cases)
- Each moment should genuinely require deep reasoning
- Don't force moments if evidence is weak
- Spurious is the hardest - only flag if you're confident in the mismatch

Return JSON:
{{
  "moments": [array of moment objects],
  "reasoning_notes": {{
    "inference_rationale": "Why these moments require inference...",
    "holistic_patterns": "What patterns were found...",
    "avstitching_insights": "Key editing choices...",
    "spurious_analysis": "Mismatches detected..."
  }}
}}

Begin deep analysis. Use your full reasoning capabilities - these are the hardest moments in the video.
"""

        return prompt

    def call_opus_4(
        self,
        frames: List[Dict],
        flagged_frames: List[Dict],
        spurious_candidates: List[Dict],
        audio_analysis: Dict,
        full_video_context: Dict
    ) -> Dict:
        """
        Call Opus 4 for deep analysis

        Args:
            frames: Prepared frames with images
            flagged_frames: Frames flagged by Pass 2A
            spurious_candidates: CLIP spurious candidates
            audio_analysis: Audio analysis
            full_video_context: Full video context

        Returns:
            Parsed response with moments
        """
        logger.info(f"Calling Opus 4 with {len(frames)} frames for deep reasoning...")

        # Build prompt
        prompt = self.build_prompt(
            frames,
            flagged_frames,
            spurious_candidates,
            audio_analysis,
            full_video_context
        )

        # Build content (text + images)
        content = [{"type": "text", "text": prompt}]

        # Add images
        for frame in frames[:30]:  # Limit to 30 images for Opus 4
            if 'image_base64' in frame:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": frame['image_base64']
                    }
                })

        try:
            # ✅ FIX: Wrap API call with retry logic
            def make_api_call():
                return self.client.messages.create(
                    model=self.model,
                    max_tokens=8000,
                    messages=[
                        {"role": "user", "content": content}
                    ]
                )

            response = self._retry_api_call(make_api_call)

            # Parse response
            response_text = response.content[0].text

            # Extract JSON
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                logger.error("No JSON found in Opus 4 response")
                return {"moments": [], "reasoning_notes": {}}

            result = json.loads(json_match.group(0))

            # Track cost
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = (input_tokens * 15 / 1_000_000) + (output_tokens * 75 / 1_000_000)
            logger.info(f"Opus 4 cost: ${cost:.4f}")

            result['cost'] = cost
            result['tokens'] = {
                'input': input_tokens,
                'output': output_tokens
            }

            return result

        except Exception as e:
            logger.error(f"Opus 4 call failed after retries: {e}")
            return {"moments": [], "reasoning_notes": {}, "error": str(e)}

    def _validate_and_filter_moments(self, all_moments: List[Dict], all_frames_metadata: List[Dict]) -> tuple[List[Dict], List[Dict]]:
        """
        Validate moments and filter out those with violations.

        Checks:
        - Mode duration matching requirements
        - Cue quality (hedging/pronouns)
        - Protected radius matching mode
        - Frame content matching ontology type

        Args:
            all_moments: List of all moment dicts from LLM
            all_frames_metadata: Original frames with metadata for content validation

        Returns:
            (valid_moments, rejected_moments)
        """
        valid_moments = []
        rejected_moments = []

        # Create frame lookup for content validation
        frame_lookup = {f['frame_id']: f for f in all_frames_metadata}

        validation_stats = {
            'mode_duration': 0,
            'cue_quality': 0,
            'protected_radius': 0,
            'frame_content': 0,
            'temporal_window': 0,  # P0 Bug Fix #3
        }

        for moment in all_moments:
            rejection_reason = None

            # ✅ OPTIMIZATION: Fail-fast validation order (cheapest/most-likely-to-fail first)

            # ✅ Validation #1: Frame content for ontology type (CHEAPEST - just dict lookup)
            # Most likely to fail, so check first to avoid wasting time on other validations
            ontology_type = moment.get('primary_ontology', '')
            frame_ids = moment.get('frame_ids', [])

            frame_content_valid = False
            for fid in frame_ids:
                frame = frame_lookup.get(fid)
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

            # Accept or reject based on all validations
            if rejection_reason is None:
                # ✅ P1 #6: Assign sub-task type based on ontology
                ontology_type = moment.get('primary_ontology', '')
                if ontology_type:
                    moment['sub_task_type'] = get_sub_task_type(ontology_type)
                valid_moments.append(moment)
            else:
                timestamps = moment.get('timestamps', [])
                timestamp_str = f"{timestamps[0]:.1f}s" if timestamps else "N/A"
                logger.warning(f"Rejected moment (ontology={moment.get('primary_ontology')}, "
                              f"timestamp={timestamp_str}): {rejection_reason}")
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
                    logger.warning(f"⚠️  AUDIO DIVERSITY: Insufficient audio modality diversity in Pass 2B moments")
                    for warning in diversity_warnings:
                        logger.warning(f"  {warning}")
                else:
                    logger.info(f"✅ Audio modality diversity check passed ({len(moments_for_audio_check)} moments checked)")

        # Log validation statistics
        logger.info(f"Moment validation: {len(valid_moments)} valid, {len(rejected_moments)} rejected")
        logger.info(f"  Rejection breakdown: "
                   f"duration={validation_stats['mode_duration']}, "
                   f"cue_quality={validation_stats['cue_quality']}, "
                   f"radius={validation_stats['protected_radius']}, "
                   f"frame_content={validation_stats['frame_content']}")

        return valid_moments, rejected_moments

    def process_frames(
        self,
        flagged_frames: List[Dict],
        spurious_candidates: List[Dict],
        audio_analysis: Dict,
        frames_dir: Path,
        all_frames_metadata: List[Dict],
        full_video_context: Dict
    ) -> Dict:
        """
        Run full Pass 2B processing

        Args:
            flagged_frames: Frames flagged by Pass 2A
            spurious_candidates: CLIP spurious candidates
            audio_analysis: Audio analysis
            frames_dir: Directory with frame images
            all_frames_metadata: All frame metadata
            full_video_context: Full video context

        Returns:
            {
                'mode1_precise': [...],
                'mode2_micro_temporal': [...],
                'mode3_inference_window': [...],
                'mode4_clusters': [...],
                'reasoning_notes': {...},
                'cost': float
            }
        """
        logger.info("=" * 60)
        logger.info("PASS 2B: Opus 4 Hard Ontology Types + Spurious")
        logger.info("=" * 60)

        # Prepare frames
        prepared_frames = self.prepare_frames(
            flagged_frames,
            spurious_candidates,
            frames_dir,
            all_frames_metadata
        )

        if not prepared_frames:
            logger.warning("No frames to process in Pass 2B")
            return {
                'mode1_precise': [],
                'mode2_micro_temporal': [],
                'mode3_inference_window': [],
                'mode4_clusters': [],
                'reasoning_notes': {},
                'cost': 0
            }

        # Call Opus 4
        response = self.call_opus_4(
            prepared_frames,
            flagged_frames,
            spurious_candidates,
            audio_analysis,
            full_video_context
        )

        # Validate and filter moments
        all_moments = response.get('moments', [])

        # ✅ CRITICAL GAP FIX: Normalize types and enforce Sequential+Temporal rule BEFORE validation
        # This must happen before validation to ensure moments aren't rejected due to missing Temporal Understanding
        from processing.ontology_types import normalize_type
        for moment in all_moments:
            # Normalize ontology type names to official PDF names
            if 'primary_ontology' in moment:
                moment['primary_ontology'] = normalize_type(moment['primary_ontology'])
            if 'secondary_ontologies' in moment:
                moment['secondary_ontologies'] = [normalize_type(t) for t in moment['secondary_ontologies']]

            # Enforce Sequential+Temporal Understanding co-occurrence rule
            # Guidelines: "Sequential always go hand in hand with Temporal Understanding"
            primary = moment.get('primary_ontology', '')
            secondaries = moment.get('secondary_ontologies', [])
            all_types = [primary] + secondaries
            if "Sequential" in all_types and "Temporal Understanding" not in all_types:
                # Add Temporal Understanding to secondary ontologies
                if 'secondary_ontologies' not in moment:
                    moment['secondary_ontologies'] = []
                moment['secondary_ontologies'].append("Temporal Understanding")
                logger.debug(f"Enforcing Sequential+Temporal rule: Added Temporal Understanding to moment at {moment.get('timestamps', [0])[0]:.1f}s")

        valid_moments, rejected_moments = self._validate_and_filter_moments(all_moments, all_frames_metadata)

        # Organize validated moments by mode
        mode1_moments = []
        mode2_moments = []
        mode3_moments = []
        mode4_moments = []

        for moment_data in valid_moments:
            mode = moment_data.get('mode', 'inference_window')

            if mode == "precise":
                mode1_moments.append(moment_data)
            elif mode == "micro_temporal":
                mode2_moments.append(moment_data)
            elif mode == "inference_window":
                mode3_moments.append(moment_data)
            elif mode == "cluster":
                mode4_moments.append(moment_data)

        logger.info("=" * 60)
        logger.info(f"Pass 2B Complete:")
        logger.info(f"  Mode 1 (precise): {len(mode1_moments)}")
        logger.info(f"  Mode 2 (micro_temporal): {len(mode2_moments)}")
        logger.info(f"  Mode 3 (inference_window): {len(mode3_moments)}")
        logger.info(f"  Mode 4 (cluster): {len(mode4_moments)}")
        logger.info(f"  Valid moments: {len(valid_moments)}")
        logger.info(f"  Rejected moments: {len(rejected_moments)}")
        logger.info("=" * 60)

        return {
            'mode1_precise': mode1_moments,
            'mode2_micro_temporal': mode2_moments,
            'mode3_inference_window': mode3_moments,
            'mode4_clusters': mode4_moments,
            'reasoning_notes': response.get('reasoning_notes', {}),
            'cost': response.get('cost', 0)
        }


def run_pass2b_selection(
    flagged_frames: List[Dict],
    spurious_candidates: List[Dict],
    audio_analysis: Dict,
    frames_dir: str,
    all_frames_metadata: List[Dict],
    full_video_context: Dict,
    output_path: str
) -> Dict:
    """
    Run Pass 2B and save results

    Args:
        flagged_frames: Frames flagged by Pass 2A
        spurious_candidates: CLIP spurious candidates
        audio_analysis: Audio analysis
        frames_dir: Directory with frame images
        all_frames_metadata: All frame metadata
        full_video_context: Full video context
        output_path: Path to save results

    Returns:
        Pass 2B results
    """
    selector = Pass2BOpusSelector()

    results = selector.process_frames(
        flagged_frames,
        spurious_candidates,
        audio_analysis,
        Path(frames_dir),
        all_frames_metadata,
        full_video_context
    )

    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Pass 2B results saved to {output_path}")

    return results
