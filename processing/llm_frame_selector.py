"""
LLM Frame Selector - Phase 5: Claude-powered intelligent frame selection

Uses Claude Sonnet 4.5 with FULL VISUAL CONTEXT to select frames:
1. Visual context from quick_visual_sampler (BLIP-2, objects, scenes, etc.)
2. Ranked highlights from universal_highlight_detector
3. Quality map from quality_mapper
4. Dynamic frame budget
5. Coverage of all 13 question types
6. ✅ NEW: Spatial coverage guarantee (6 segments, min 3 frames each)

KEY: Claude sees visual context BEFORE selecting frames
"""

import anthropic
import logging
from typing import List, Dict, Optional, Tuple
import os
import json

logger = logging.getLogger(__name__)


class LLMFrameSelector:
    """Claude-powered intelligent frame selection with visual context"""

    # 13 NVIDIA question types
    QUESTION_TYPES = [
        "Temporal Understanding",
        "Sequential",
        "Subscene",
        "General Holistic Reasoning",
        "Inference",
        "Context",
        "Needle",
        "Referential Grounding",
        "Counting",
        "Comparative",
        "Object Interaction Reasoning",
        "Audio-Visual Stitching",
        "Tackling Spurious Correlations"
    ]

    def __init__(self, anthropic_api_key: Optional[str] = None):
        """Initialize Claude client"""
        api_key = anthropic_api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY required")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-5-20250929"

    def select_frames(
        self,
        visual_samples: List[Dict],
        highlights: List[Dict],
        quality_map: Dict[float, float],
        frame_budget: int,
        video_duration: float
    ) -> Dict:
        """
        Intelligently select frames using Claude with visual context.

        Args:
            visual_samples: From quick_visual_sampler (BLIP-2 captions, objects, etc.)
            highlights: Ranked highlights from universal_highlight_detector
            quality_map: Quality scores by timestamp
            frame_budget: Number of frames to select (47-150)
            video_duration: Total video duration in seconds

        Returns:
            {
                'selection_plan': [
                    {
                        'timestamp': float,
                        'reason': str,
                        'question_types': [str],
                        'priority': float
                    }
                ],
                'dense_clusters': [
                    {
                        'start': float,
                        'end': float,
                        'frame_count': int,
                        'reason': str,
                        'scene_types': [str],
                        'scene_type_consistent': bool,
                        'validation': {
                            'same_scene_type': bool,
                            'same_location': bool,
                            'continuous_action': bool,
                            'is_scene_cut': bool
                        }
                    }
                ],
                'coverage': dict,
                'cost_summary': {
                    'claude_api_call': {
                        'input_tokens': int,
                        'output_tokens': int,
                        'total_tokens': int,
                        'input_cost': float,
                        'output_cost': float,
                        'total_cost': float
                    }
                }
            }
        """
        logger.info(f"Selecting {frame_budget} frames with Claude + visual context...")

        # Build visual context summary
        visual_context = self._build_visual_context(visual_samples)

        # Build highlight summary
        highlight_summary = self._build_highlight_summary(highlights[:30])  # Top 30

        # Build quality summary
        quality_summary = self._build_quality_summary(quality_map)

        # Call Claude for frame selection
        selection_plan = self._call_claude_for_selection(
            visual_context,
            highlight_summary,
            quality_summary,
            frame_budget,
            video_duration,
            highlights
        )

        # Multi-layer cluster validation
        dense_clusters = selection_plan.get('dense_clusters', [])
        validated_clusters = []

        logger.info(f"Validating {len(dense_clusters)} clusters from Claude...")

        for i, cluster in enumerate(dense_clusters, 1):
            start = cluster['start']
            end = cluster['end']

            # Layer 4: Python coherence validation (structured fields)
            is_valid_coherence, coherence_reason = self._validate_cluster_coherence(
                cluster,
                visual_samples
            )

            if not is_valid_coherence:
                logger.warning(f"  ✗ Cluster {i}: {start:.1f}s-{end:.1f}s REJECTED (Coherence)")
                logger.warning(f"    Reason: {coherence_reason}")
                logger.warning(f"    Scene types: {cluster.get('scene_types', [])}")
                continue  # Skip this cluster

            # Passed coherence validation!
            validated_clusters.append(cluster)
            logger.info(f"  ✓ Cluster {i}: {start:.1f}s-{end:.1f}s VALID (Coherence check)")

        logger.info(f"Cluster validation: {len(dense_clusters)} → {len(validated_clusters)} valid")
        if len(validated_clusters) < len(dense_clusters):
            logger.info(f"  Rejected {len(dense_clusters) - len(validated_clusters)} clusters (scene cuts or invalid)")

        # Extract single frames
        single_frames = selection_plan.get('single_frames', [])

        # ✅ NEW: Ensure spatial coverage across full video timeline
        logger.info(f"\n🗺️  Ensuring spatial coverage across video timeline...")
        single_frames = self.ensure_spatial_coverage(
            selected_frames=single_frames,
            highlights=highlights,  # ✅ IMPROVED: Use Phase 3 highlights instead of Phase 2 visual samples
            video_duration=video_duration,
            num_segments=6,
            min_frames_per_segment=3
        )

        # Validate coverage of question types (CRITICAL - cannot be neglected)
        coverage = self._validate_type_coverage(single_frames)

        logger.info(f"Selected {len(single_frames)} single frames + {len(validated_clusters)} validated clusters")
        logger.info(f"Type coverage: {coverage['covered_types']}/{coverage['total_types']} ({coverage['coverage_ratio']:.1%})")
        if coverage['missing_types']:
            logger.warning(f"  Missing types: {', '.join(coverage['missing_types'])}")

        # Preserve cost_summary from selection_plan
        cost_summary = selection_plan.get('cost_summary', {})

        return {
            'selection_plan': single_frames,
            'dense_clusters': validated_clusters,  # Return validated clusters only
            'coverage': coverage,  # Preserve coverage statistics
            'cost_summary': cost_summary  # Preserve cost tracking!
        }

    def _validate_cluster_coherence(
        self,
        cluster: Dict,
        visual_samples: List[Dict]
    ) -> Tuple[bool, str]:
        """
        Layer 4: Validate cluster coherence using structured validation fields.

        Args:
            cluster: Dense cluster dict from Claude with validation fields
            visual_samples: Visual samples to cross-check scene_types

        Returns:
            (is_valid: bool, rejection_reason: str)
        """
        # Check 1: Validation fields present
        validation = cluster.get('validation', {})
        if not validation:
            logger.debug("Cluster missing validation fields, allowing through (legacy format)")
            return True, ""  # Allow legacy format without validation fields

        # Check 2: Must NOT be a scene cut
        if validation.get('is_scene_cut', False):
            return False, "Marked as scene cut by Claude"

        # Check 3: Must have same scene_type
        if not validation.get('same_scene_type', False):
            return False, "Different scene_types detected"

        # Check 4: Must be same location
        if not validation.get('same_location', False):
            return False, "Different locations detected"

        # Check 5: Must have continuous action
        if not validation.get('continuous_action', False):
            return False, "No continuous action detected"

        # Check 6: Verify scene_type_consistent flag
        if not cluster.get('scene_type_consistent', True):  # Default True for legacy
            return False, "scene_type_consistent is false"

        # Check 7: Cross-validate with actual scene_types array
        scene_types = cluster.get('scene_types', [])
        if len(scene_types) >= 2:
            # All scene_types must be identical
            first_scene_type = scene_types[0]
            if not all(st == first_scene_type for st in scene_types):
                actual_types = " → ".join(scene_types)
                return False, f"Scene type mismatch: {actual_types} (this is a scene cut!)"

        # Check 8: Cross-validate with visual_samples
        start = cluster['start']
        end = cluster['end']

        cluster_samples = [
            s for s in visual_samples
            if start <= s['timestamp'] <= end
        ]

        if cluster_samples:
            sample_scene_types = [s.get('scene_type', 'unknown') for s in cluster_samples]
            unique_scene_types = set(sample_scene_types)

            if len(unique_scene_types) > 1:
                return False, f"Visual samples show scene type changes: {unique_scene_types}"

        # Passed all checks
        return True, ""

    # ✅ NEW FUNCTION: Ensure spatial coverage
    def ensure_spatial_coverage(
        self,
        selected_frames: List[Dict],
        highlights: List[Dict],
        video_duration: float,
        num_segments: int = 6,
        min_frames_per_segment: int = 3
    ) -> List[Dict]:
        """
        Ensure every time segment has minimum frames for Phase 8 distribution.

        This prevents Phase 8 from having segments with 0 available frames,
        which would break the spatial distribution goal.

        Architecture:
        - Phase 5 selects based on quality (highlights)
        - This ensures based on coverage (spatial distribution)
        - Together = quality + coverage ✅

        Args:
            selected_frames: Frames already selected by Claude
            highlights: Phase 3 highlights (multi-signal fusion - better than Phase 2)
            video_duration: Total video duration in seconds
            num_segments: Number of time segments (default: 6)
            min_frames_per_segment: Minimum frames needed per segment (default: 3)

        Returns:
            selected_frames with gap-filling frames added
        """
        logger.info(f"   Ensuring spatial coverage ({num_segments} segments)...")

        # Create time segments
        segment_duration = video_duration / num_segments
        segments = []
        for i in range(num_segments):
            start = i * segment_duration
            end = (i + 1) * segment_duration if i < num_segments - 1 else video_duration
            segments.append({
                'id': i,
                'start': start,
                'end': end,
                'frames': []
            })

        # Assign selected frames to segments
        for frame in selected_frames:
            timestamp = frame['timestamp']
            for seg in segments:
                if seg['start'] <= timestamp < seg['end']:
                    seg['frames'].append(frame)
                    break

        # Identify gaps and fill them
        gap_filled_frames = list(selected_frames)  # Copy existing
        total_gaps_filled = 0

        for seg in segments:
            current_count = len(seg['frames'])

            if current_count < min_frames_per_segment:
                gap = min_frames_per_segment - current_count
                logger.info(f"   Segment {seg['id']} ({seg['start']:.0f}-{seg['end']:.0f}s): "
                           f"{current_count} frames → filling {gap} gaps")

                # ✅ IMPROVED: Find Phase 3 highlights in this segment NOT already selected
                selected_timestamps = {f['timestamp'] for f in selected_frames}

                segment_highlights = [
                    h for h in highlights
                    if seg['start'] <= h['timestamp'] < seg['end']
                    and h['timestamp'] not in selected_timestamps
                ]

                if not segment_highlights:
                    logger.warning(f"      ⚠️  No Phase 3 highlights available in segment {seg['id']}, skipping")
                    continue

                # ✅ IMPROVED: Sort by combined_score (multi-signal fusion)
                # This is better than BRISQUE quality alone as it includes audio + visual + semantic
                segment_highlights.sort(
                    key=lambda h: h.get('combined_score', 0),
                    reverse=True
                )

                # Take top highlights to fill gap
                selected_highlights = segment_highlights[:gap]
                logger.info(f"      Using {len(selected_highlights)} Phase 3 highlights (combined_score based)")

                # Add selected highlights to fill gap
                for highlight in selected_highlights:
                    combined_score = highlight.get('combined_score', 0.5)
                    audio_score = highlight.get('audio_score', 0)
                    visual_score = highlight.get('visual_score', 0)
                    semantic_score = highlight.get('semantic_score', 0)

                    # ✅ IMPROVED: Convert Phase 3 highlight to Phase 5 frame format
                    # Include all multi-signal scores for better transparency
                    gap_frame = {
                        'timestamp': highlight['timestamp'],
                        'reason': f"Gap-fill (highlight score: {combined_score:.3f})",
                        'question_types': ['Context', 'General Holistic Reasoning'],  # Generic types
                        'priority': combined_score * 0.9,  # ✅ REDUCED PENALTY: 10% (was 20%)
                        'audio_score': audio_score,
                        'visual_score': visual_score,
                        'semantic_score': semantic_score,
                        'is_gap_fill': True  # Mark as gap-fill
                    }
                    gap_filled_frames.append(gap_frame)
                    total_gaps_filled += 1
                    logger.info(f"      ✓ Added gap-fill frame at {highlight['timestamp']:.1f}s "
                               f"(combined_score={combined_score:.3f}, priority={gap_frame['priority']:.2f})")
            else:
                logger.info(f"   Segment {seg['id']} ({seg['start']:.0f}-{seg['end']:.0f}s): "
                           f"{current_count} frames ✅")

        logger.info(f"   Total frames: {len(selected_frames)} → {len(gap_filled_frames)} "
                   f"(+{total_gaps_filled} gap-fills)")

        return gap_filled_frames

    def _build_visual_context(self, visual_samples: List[Dict]) -> str:
        """Build concise visual context summary from samples"""
        context_lines = []

        for i, sample in enumerate(visual_samples[:50]):  # Max 50 samples
            timestamp = sample['timestamp']
            caption = sample.get('blip2_caption', 'N/A')
            scene_type = sample.get('scene_type', 'N/A')
            objects = sample.get('objects', [])
            text = sample.get('text_detected', '')
            poses = sample.get('poses', [])
            quality = sample.get('quality', 0.0)

            # Concise summary
            obj_str = f"{len(objects)} objects" if objects else "no objects"
            text_str = f"text: '{text[:30]}'" if text else "no text"
            pose_str = f"{len(poses)} people" if poses else "no people"

            line = f"[{timestamp:.1f}s] {caption} | Scene: {scene_type} | {obj_str}, {pose_str}, {text_str} | Q: {quality:.1f}"
            context_lines.append(line)

        return "\n".join(context_lines)

    def _build_highlight_summary(self, highlights: List[Dict]) -> str:
        """Build highlight summary"""
        lines = []

        for h in highlights:
            timestamp = h['timestamp']
            score = h['combined_score']
            audio = h['audio_score']
            visual = h['visual_score']
            semantic = h['semantic_score']

            signals = []
            if audio > 0:
                signals.append(f"audio:{audio:.2f}")
            if visual > 0:
                signals.append(f"visual:{visual:.2f}")
            if semantic > 0:
                signals.append(f"semantic:{semantic:.2f}")

            signal_str = ", ".join(signals)
            line = f"[{timestamp:.1f}s] score={score:.3f} ({signal_str})"
            lines.append(line)

        return "\n".join(lines)

    def _build_quality_summary(self, quality_map: Dict[float, float]) -> Dict:
        """Build quality summary statistics"""
        if not quality_map:
            return {'average': 0.0, 'high_quality_ratio': 0.0}

        scores = list(quality_map.values())
        avg_quality = sum(scores) / len(scores)
        high_quality_count = sum(1 for s in scores if s >= 0.8)
        high_quality_ratio = high_quality_count / len(scores)

        return {
            'average': avg_quality,
            'high_quality_ratio': high_quality_ratio,
            'total_samples': len(scores)
        }

    def _call_claude_for_selection(
        self,
        visual_context: str,
        highlight_summary: str,
        quality_summary: Dict,
        frame_budget: int,
        video_duration: float,
        highlights: List[Dict]
    ) -> Dict:
        """Call Claude for intelligent frame selection"""

        # Target 120-150 frames for ~60 questions (after Phase 8 filters duplicates)
        # Previous issue: Claude was too conservative, selecting only ~40 frames
        # Fix: Clear instruction that 120-150 is the FINAL count, no post-filtering
        min_target = max(47, int(frame_budget * 0.80))  # 120 frames for 150 budget
        max_target = int(frame_budget * 1.00)  # 150 frames for 150 budget

        prompt = f"""You are selecting frames from a {video_duration:.1f}s video for adversarial multimodal question generation.

VISUAL CONTEXT (from FREE models - BLIP-2, YOLO, etc.):
{visual_context}

HIGHLIGHTS (multi-signal fusion):
{highlight_summary}

QUALITY SUMMARY:
- Average quality: {quality_summary.get('average', 0.0):.2f}
- High quality ratio: {quality_summary.get('high_quality_ratio', 0.0):.2%}

═══════════════════════════════════════════════════════════════════════════════
OBJECTIVE CRITERIA FOR 13 QUESTION TYPES
═══════════════════════════════════════════════════════════════════════════════

1. TEMPORAL UNDERSTANDING
   ✓ Evidence: Multiple highlights across video showing changes/progression

2. SEQUENTIAL
   ✓ Evidence: Ordered events/steps visible in visual context

3. SUBSCENE
   ✓ Evidence: Continuous action sequence within same scene (4+ highlights within 10s)

4. GENERAL HOLISTIC REASONING
   ✓ Evidence: Rich visual context (objects, people, actions)

5. INFERENCE
   ✓ Evidence: Visual cues suggesting implicit information

6. CONTEXT
   ✓ Evidence: Environmental details providing situational context

7. NEEDLE
   ✓ Evidence: Readable text in text_detected field

8. REFERENTIAL GROUNDING
   ✓ Evidence: 2+ objects detected in same frame

9. COUNTING
   ✓ Evidence: Multiple instances of same object type

10. COMPARATIVE
    ✓ Evidence: Contrasting visual elements across timestamps

11. OBJECT INTERACTION REASONING
    ✓ Evidence: Person-object interactions visible

12. AUDIO-VISUAL STITCHING ⚠️  STRICT VALIDATION REQUIRED
    ✓ Evidence: High audio_score + rich visual context

    ⚠️  CRITICAL: Audio MUST describe what's visible in frame!

    Before assigning "Audio-Visual Stitching", verify ONE of these:

    ✅ VALID Audio-Visual Stitching (assign this type):

    1. OBJECT REFERENCE: Speech mentions object in detected_objects list
       Example: "this model" + detected_objects: ["person", "toy"] ✅
       Example: "that container" + detected_objects: ["bottle", "cup"] ✅
       Example: "the device" + detected_objects: ["cell phone", "remote"] ✅

    2. ACTION NARRATION: Speech describes visible action/motion
       Example: "object moving left" + motion detected + direction ✅
       Example: "reaching for the item" + pose_count > 0 + motion ✅
       Example: "combining materials" + object_interaction visible ✅

    3. COUNTING CUE: Speech mentions quantity matching visual count
       Example: "multiple items" + 5+ detected_objects ✅
       Example: "three instances" + 3 similar objects detected ✅
       Example: "six repetitions" + repetitive action 6 times ✅

    4. SCENE DESCRIPTION: Speech names the scene/environment
       Example: "indoors" + scene_type: "indoor" ✅
       Example: "in this workspace" + scene_type: "conference_room" ✅
       Example: "outside" + scene_type contains "outdoor" ✅

    ❌ INVALID Audio-Visual Stitching (DO NOT assign):

    1. Generic speech + unrelated visual
       Example: "this is important" + random objects ❌
       Example: "great example" + unrelated scene ❌

    2. Audio peak + motion peak (coincidence, not description)
       Example: loud music + camera pan (audio doesn't describe pan) ❌
       Example: volume spike + motion (no semantic link) ❌

    3. Topic shift + scene change (editing, not alignment)
       Example: "now let's discuss X" + new scene (just a cut) ❌
       Example: "next topic" + different location (transition) ❌

    4. Emphasis without visual reference
       Example: "this is extremely important!" + no specific object mentioned ❌
       Example: "pay attention!" + generic scene ❌

    VALIDATION CHECKLIST for "Audio-Visual Stitching":
    ☐ Does audio provide a temporal/contextual anchor for the visual?
    ☐ Would answering this require BOTH audio AND visual information?
    ☐ Can the answer be determined from visual alone? (If YES → reject)
    ☐ Can the answer be determined from audio alone? (If YES → reject)

    Audio-Visual Stitching is VALID if:
    - Audio and visual are temporally synchronized (happen together)
    - Question requires integrating BOTH modalities
    - Answer cannot be determined from single modality

    Accept these patterns:
    ✓ Speech mentions object → visual shows object
    ✓ Sound effect → visual shows source of sound
    ✓ Music change → visual shows corresponding action/transition
    ✓ Silence → visual shows dramatic moment

    If audio and visual are unrelated → DO NOT assign "Audio-Visual Stitching"
    → Assign "General Holistic Reasoning" or "Context" instead

13. TACKLING SPURIOUS CORRELATIONS
    ✓ Evidence: Unusual/unexpected context

═══════════════════════════════════════════════════════════════════════════════
CLUSTER REQUIREMENTS (CRITICAL - READ CAREFULLY)
═══════════════════════════════════════════════════════════════════════════════

CLUSTER VALIDATION (Flexible but Effective):

PRIORITY 1 - MUST HAVE (Hard Requirements):
✓ No drastic visual changes (ImageHash diff < adaptive_threshold)
✓ Same physical location (spatial consistency across frames)
✓ Temporal continuity (no gaps > 3 seconds between frames)
✓ Optimal duration: 5-15 seconds (min: 4s, max: 20s)

PRIORITY 2 - SHOULD HAVE (Soft Requirements):
✓ Scene_type similarity (allow minor BLIP-2 labeling variations)
  → "indoor_basketball_court" vs "basketball_game" = SAME SCENE ✓
  → "indoor_medium" vs "conference_room" = DIFFERENT SCENES ✗
✓ Related actions (causal/temporal link between frames)
✓ Object consistency (similar objects across frames)

PRIORITY 3 - NICE TO HAVE (Bonus):
✓ 4+ frames within 10 seconds (high density)
✓ Clear progression (beginning → middle → end)

CLUSTER TEMPORAL WINDOW GUIDELINES:
Optimal cluster duration: 5-15 seconds
- Minimum: 4 seconds (need enough context for multi-step reasoning)
- Maximum: 20 seconds (avoid scene cuts, maintain coherence)

DURATION BY CLUSTER TYPE:
- Fast action (sports, combat, games): 5-10 seconds
- Process/sequence (cooking, assembly, crafts): 8-15 seconds
- Transformation (construction, timelapse): 10-20 seconds

FORBIDDEN DURATIONS:
❌ < 4 seconds (too short for temporal reasoning)
❌ > 25 seconds (high risk of scene cuts, context shift)

CLUSTER VALIDATION CHECKLIST - Verify for EACH proposed cluster:
☐ No scene cuts detected by ImageHash? (diff < threshold)
   → If scene cut detected: REJECT cluster
☐ Same physical location across frames?
   → If location changes: REJECT cluster
☐ No temporal gaps > 3 seconds?
   → If large gap: REJECT cluster
☐ Scene_type labels reasonably similar? (allow BLIP-2 noise)
   → If completely different contexts (office→street): REJECT
☐ Duration between 5-20 seconds?
   → If < 4s or > 25s: REJECT cluster

FORBIDDEN CLUSTER TYPES (AUTO-REJECT):
✗ "Scene transition sequence" = SCENE CUT, not cluster
   → Example: indoor_medium (111s) → conference_room (113s)
   → This is TWO DIFFERENT SCENES, REJECT!
✗ "Location change sequence" = SCENE CUT, not cluster
   → Example: office → hallway → conference room
   → Multiple locations = scene cuts, REJECT!
✗ "Topic shift sequence" = CONTENT CUT, not cluster
   → Example: interview → demonstration → interview
   → Content discontinuity = cut, REJECT!

VALID CLUSTER EXAMPLES:
✓ "Continuous action - 4 frames showing repeated motion in same location (scene_type: indoor)"
   → Same location, continuous action = VALID
✓ "Process sequence - step 1 → step 2 → step 3 in same workspace (scene_type: indoor)"
   → Same location, related actions = VALID
✓ "Presentation gestures - 3 consecutive gestures at whiteboard (scene_type: conference_room)"
   → Same location, continuous action = VALID

INVALID CLUSTER EXAMPLES (NEVER CREATE THESE):
✗ "Scene transition: indoor_medium → conference_room" = SCENE CUT
✗ "Location change: office → hallway" = SCENE CUT
✗ Any cluster with "→" indicating scene_type change = SCENE CUT

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT WITH VALIDATION FIELDS
═══════════════════════════════════════════════════════════════════════════════

Return ONLY valid JSON with this EXACT structure:

{{
  "execution_summary": {{
    "pass1_count": <integer>,
    "pass2_executed": <boolean>,
    "pass2_count": <integer>,
    "frames_added_pass2": <integer>,
    "gap_analysis": {{
      "total_gaps_identified": <integer>,
      "qualifying_gaps_over_40s": <integer>,
      "gaps_filled": <integer>
    }}
  }},
  "single_frames": [
    {{
      "timestamp": 10.5,
      "reason": "Audio peak (0.88) + topic shift (semantic: 0.95)",
      "question_types": ["Audio-Visual Stitching", "Temporal Understanding"],
      "priority": 0.95
    }}
  ],
  "dense_clusters": [
    {{
      "start": 45.0,
      "end": 52.0,
      "frame_count": 7,
      "reason": "Basketball dribbling sequence in same indoor court (scene_type: basketball_court_indoor)",
      
      "scene_types": ["basketball_court_indoor", "basketball_court_indoor", "basketball_court_indoor"],
      "scene_type_consistent": true,
      "validation": {{
        "same_scene_type": true,
        "same_location": true,
        "continuous_action": true,
        "is_scene_cut": false
      }}
    }}
  ],
  "coverage": {{
    "covered_types": <integer>,
    "total_types": 13,
    "missing_types": [<list of strings>],
    "coverage_ratio": <float>
  }}
}}

CRITICAL VALIDATION FIELD REQUIREMENTS:
✓ scene_types: Array of scene_type from EACH frame in cluster (check visual context)
✓ scene_type_consistent: true if all scene_types identical, false otherwise
✓ validation.same_scene_type: MUST be true for valid cluster
✓ validation.same_location: MUST be true for valid cluster
✓ validation.continuous_action: MUST be true for valid cluster
✓ validation.is_scene_cut: MUST be false for valid cluster

CLUSTER REASON FORMAT:
✓ GOOD: "Continuous action in same scene_type" (e.g., "Dribbling in basketball_court_indoor")
✗ BAD: "Scene transition" or "X → Y" (indicates scene cut)

═══════════════════════════════════════════════════════════════════════════════
FINAL CHECKLIST (Verify Before Submitting)
═══════════════════════════════════════════════════════════════════════════════

CLUSTER VALIDATION (CRITICAL):
✓ Every cluster has ALL frames with SAME scene_type (verify in visual context)
✓ Cluster scene_types array shows identical values
✓ Cluster scene_type_consistent = true
✓ Cluster validation.is_scene_cut = false
✓ Cluster validation.same_scene_type = true
✓ Cluster validation.same_location = true
✓ Cluster validation.continuous_action = true
✓ Cluster reasons describe continuous action (NOT "scene transition")
✓ NO clusters with "→" or scene_type changes in reason

COVERAGE (MUST BE MAINTAINED):
✓ ALL frames with 2+ objects include "Referential Grounding" type
✓ Coverage ≥11/13 types (85% minimum)
✓ Missing types documented in coverage.missing_types

FRAME COUNT:
✓ Total frames between 120-150 (prioritize BOTH quality and comprehensive coverage)
✓ Pass 2 executed if Pass 1 < 120 frames
✓ This is the final selection - no post-filtering will occur

JSON:"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=10000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )

            # Capture token usage from response
            usage = response.usage
            input_tokens = usage.input_tokens
            output_tokens = usage.output_tokens
            total_tokens = input_tokens + output_tokens

            # Calculate cost (Claude Sonnet 4.5 pricing)
            # Input: $3.00 per 1M tokens, Output: $15.00 per 1M tokens
            input_cost = (input_tokens / 1_000_000) * 3.00
            output_cost = (output_tokens / 1_000_000) * 15.00
            total_cost = input_cost + output_cost

            logger.info(f"Claude API call completed:")
            logger.info(f"  Input tokens:  {input_tokens:,}")
            logger.info(f"  Output tokens: {output_tokens:,}")
            logger.info(f"  Total tokens:  {total_tokens:,}")
            logger.info(f"  Cost: ${total_cost:.4f} (input: ${input_cost:.4f}, output: ${output_cost:.4f})")

            response_text = response.content[0].text.strip()

            # Strip markdown code fences if present
            if response_text.startswith('```'):
                lines = response_text.split('\n')
                if lines[0].startswith('```'):
                    lines = lines[1:]
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                response_text = '\n'.join(lines).strip()

            # Parse JSON
            selection_plan = json.loads(response_text)

            # Log execution summary
            exec_summary = selection_plan.get('execution_summary', {})
            pass1_count = exec_summary.get('pass1_count', 0)
            pass2_executed = exec_summary.get('pass2_executed', False)
            pass2_count = exec_summary.get('pass2_count', 0)
            frames_added = exec_summary.get('frames_added_pass2', 0)

            logger.info(f"Pass 1: {pass1_count} frames selected")
            logger.info(f"Pass 2 executed: {pass2_executed}")
            if pass2_executed:
                logger.info(f"Pass 2: Added {frames_added} frames → Total: {pass2_count}")

            # Validate frame count
            single_count = len(selection_plan.get('single_frames', []))
            cluster_count = sum(c['frame_count'] for c in selection_plan.get('dense_clusters', []))
            total_frames = single_count + cluster_count

            logger.info(f"Claude selected {total_frames} frames ({single_count} single + {cluster_count} in clusters)")

            # Add cost summary to selection_plan
            selection_plan['cost_summary'] = {
                'claude_api_call': {
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': total_tokens,
                    'input_cost': round(input_cost, 4),
                    'output_cost': round(output_cost, 4),
                    'total_cost': round(total_cost, 4)
                }
            }

            return selection_plan

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude response: {e}")
            return self._fallback_selection(highlights, frame_budget, video_duration)
        except Exception as e:
            logger.error(f"Error in Claude frame selection: {e}")
            return self._fallback_selection(highlights, frame_budget, video_duration)

    def _fallback_selection(
        self,
        highlights: List[Dict],
        frame_budget: int,
        video_duration: float
    ) -> Dict:
        """Fallback selection if Claude fails"""
        logger.warning("Using fallback frame selection")

        single_frames = []
        max_target = int(frame_budget * 0.80)

        for i, h in enumerate(highlights[:max_target]):
            single_frames.append({
                'timestamp': h['timestamp'],
                'reason': f"Highlight rank {i+1}, score {h['combined_score']:.3f}",
                'question_types': ["General Holistic Reasoning"],
                'priority': h['combined_score']
            })

        return {
            'single_frames': single_frames,
            'dense_clusters': []
        }

    def _validate_type_coverage(self, single_frames: List[Dict]) -> Dict:
        """
        Validate coverage of question types.

        CRITICAL: This ensures we have diverse question types across frames.
        Cannot be neglected as it drives question generation quality.
        """
        covered_types = set()

        for frame in single_frames:
            question_types = frame.get('question_types', [])
            covered_types.update(question_types)

        missing_types = set(self.QUESTION_TYPES) - covered_types

        return {
            'covered_types': len(covered_types),
            'total_types': len(self.QUESTION_TYPES),
            'missing_types': list(missing_types),
            'coverage_ratio': len(covered_types) / len(self.QUESTION_TYPES)
        }

    @staticmethod
    def get_top_highlights(
        highlights: List[Dict],
        top_n: int = 50,
        min_score: float = 0.3
    ) -> List[Dict]:
        """Get top N highlights above minimum score"""
        filtered = [h for h in highlights if h.get('combined_score', 0) >= min_score]
        sorted_highlights = sorted(filtered, key=lambda x: x.get('combined_score', 0), reverse=True)
        return sorted_highlights[:top_n]