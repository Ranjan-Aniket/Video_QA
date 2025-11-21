"""
LLM Frame Selector - Phase 5: Claude-powered intelligent frame selection

Uses Claude Sonnet 4.5 with FULL VISUAL CONTEXT to select frames:
1. Visual context from quick_visual_sampler (BLIP-2, objects, scenes, etc.)
2. Ranked highlights from universal_highlight_detector
3. Quality map from quality_mapper
4. Dynamic frame budget
5. Coverage of all 13 question types

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
                        'reason': str
                    }
                ],
                'coverage': dict
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

        # Extract dense clusters and single frames
        dense_clusters = selection_plan.get('dense_clusters', [])
        single_frames = selection_plan.get('single_frames', [])

        # Validate coverage of question types
        coverage = self._validate_type_coverage(single_frames)

        logger.info(f"Selected {len(single_frames)} single frames + {len(dense_clusters)} dense clusters")
        logger.info(f"Type coverage: {coverage['covered_types']}/{coverage['total_types']}")

        return {
            'selection_plan': single_frames,
            'dense_clusters': dense_clusters,
            'coverage': coverage
        }

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

        # Calculate target range (80-120 frames based on quality)
        min_target = max(47, int(frame_budget * 0.53))  # 80 frames for 150 budget
        max_target = int(frame_budget * 0.80)  # 120 frames for 150 budget

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
   ✓ Example: "3 topic shifts at 10s, 45s, 80s showing presentation flow"

2. SEQUENTIAL
   ✓ Evidence: Ordered events/steps visible in visual context
   ✓ Example: "Step-by-step demo: setup (15s) → execution (30s) → results (50s)"

3. SUBSCENE
   ✓ Evidence: Continuous action sequence within same scene (4+ highlights within 10s)
   ✓ Evidence: Multi-step process with intermediate stages (setup → execution → result)
   ✓ Example: "Door opening sequence: hand on handle (5s) → turning (6s) → pulling (7s) → entering (9s)"
   ✓ Example: "Cooking sequence: chopping (15s) → mixing (18s) → pouring (22s) in same kitchen scene"

4. GENERAL HOLISTIC REASONING
   ✓ Evidence: Rich visual context (objects, people, actions) spanning multiple timestamps
   ✓ Example: "Complex scene: 5 people, whiteboard with diagrams, laptop screens visible"

5. INFERENCE
   ✓ Evidence: Visual cues suggesting implicit information (reactions, gestures, expressions)
   ✓ Example: "Audience leaning forward, note-taking → implies engagement/important moment"

6. CONTEXT
   ✓ Evidence: Environmental details (scene_type, objects) providing situational context
   ✓ Example: "Conference room setting, professional attire, presentation screen → business context"

7. NEEDLE
   ✓ Evidence: Readable text in text_detected field (OCR)
   ✓ Example: "Text visible: 'Q3 Revenue: $2.5M' at 25.3s"

8. REFERENTIAL GROUNDING
   ✓ Evidence: 2+ objects detected in same frame (ALWAYS triggers this type)
   ✓ Evidence: Objects with spatial relationships or specific attributes
   ✓ Example: "Laptop (left), coffee cup (center), notebook (right) - spatial positions"
   ✓ Example: "Man holding blue ring + 2 objects detected - object interaction with attributes"

   CRITICAL RULE: If frame has 2+ objects detected (from YOLO), you MUST include
   "Referential Grounding" in question_types array. This is non-negotiable.

9. COUNTING
   ✓ Evidence: Multiple instances of same object type (people, chairs, screens)
   ✓ Example: "8 people detected via poses at 15.2s"

10. COMPARATIVE
    ✓ Evidence: Contrasting visual elements across timestamps (before/after, A vs B)
    ✓ Example: "Empty room (5s) vs full audience (20s)"

11. OBJECT INTERACTION REASONING
    ✓ Evidence: Person-object interactions visible (poses + objects in same frame)
    ✓ Example: "Person holding microphone (pose + object: microphone detected)"

12. AUDIO-VISUAL STITCHING
    ✓ Evidence: High audio_score + rich visual context at same timestamp
    ✓ Example: "Audio peak (0.85) + speaker gesturing + slide change at 32.1s"

13. TACKLING SPURIOUS CORRELATIONS
    ✓ Evidence: Unusual/unexpected context (surprising scene_type, atypical objects for setting)
    ✓ Example: "Outdoor scene with server racks (atypical equipment for outdoor setting)"

═══════════════════════════════════════════════════════════════════════════════
TWO-PASS SELECTION ALGORITHM (FOLLOW EXACTLY IN ORDER)
═══════════════════════════════════════════════════════════════════════════════

PASS 1: Must-Have Frames
Execute these steps in order:
  1. Select ALL frames with priority ≥0.85 (high-quality highlights)
  2. Select ALL frames with text detected (for Needle questions)
     - Use actual highlight score if available
     - Otherwise assign priority = 0.75
  3. Select ALL frames with 5+ objects OR 5+ people (for Counting/Interaction questions)
     - Use actual highlight score if available
     - Otherwise assign priority = 0.75
  4. Identify dense clusters (4+ highlights within 10s continuous action)
     - Sample 5-8 frames per cluster, evenly spaced
     - Assign priority = 0.75 for cluster frames
  5. Count total frames selected → Store as PASS1_COUNT

PASS 2: Gap-Filling (MANDATORY IF PASS1_COUNT < 80)
This pass is MANDATORY. You cannot skip it. Execute these steps:

  Step 1: Check if Pass 2 is required
    IF PASS1_COUNT ≥ 80 → Skip to Coverage Check
    IF PASS1_COUNT < 80 → CONTINUE with Pass 2 (you MUST do this)

  Step 2: Calculate frames needed
    FRAMES_NEEDED = 80 - PASS1_COUNT
    Target: Add approximately FRAMES_NEEDED frames

  Step 3: Identify temporal gaps
    • List all selected frames in order by timestamp
    • Calculate gaps between consecutive frames
    • Mark gaps >40 seconds as "qualifying gaps"
    • Example: Frames at [10s, 55s, 100s] → Gaps of [45s, 45s] → Both qualify

  Step 4: Fill gaps (repeat until FRAMES_NEEDED satisfied OR no frames available)
    FOR each qualifying gap (>40s):
      • Find frames in that gap with priority ≥0.75
      • Select 1-2 frames from gap (prefer higher priority)
      • Add frames to selection
      • Update count
      IF total frames ≥80 → STOP
      IF no more priority ≥0.75 frames available → STOP

  Step 5: Store final count → PASS2_COUNT

  Step 6: Document execution
    Record: pass2_executed = true
    Record: frames_added = PASS2_COUNT - PASS1_COUNT

COVERAGE CHECK (After Pass 2 Completes)
  1. Count question types covered
  2. IF coverage <11/13 types:
     - Identify missing types
     - For each missing type, add 2-3 targeted frames (priority ≥0.70)
     - Special attention to Referential Grounding:
       * If missing, select frames with 2+ objects detected
       * Ensure objects have spatial relationships or attributes

CLUSTER DETECTION (After All Frames Selected)
  1. Scan all selected frames for consecutive sequences (<5s gaps)
  2. IF 3+ consecutive frames show related actions → Dense cluster
  3. Add to dense_clusters array
  4. Tag all cluster frames with "Subscene" type

STOP CONDITIONS (Check in Order)
  1. IF frames ≥{max_target} → STOP (maximum reached)
  2. IF frames ≥80 AND coverage ≥12/13 → STOP (mission accomplished)
  3. IF frames <80 but no priority ≥0.75 frames remain → STOP (accept count)

═══════════════════════════════════════════════════════════════════════════════
EXECUTION CHECKLIST (Verify Before Outputting JSON)
═══════════════════════════════════════════════════════════════════════════════

Before returning JSON, verify you completed these steps:
  ✓ Pass 1 executed, counted frames → PASS1_COUNT = ?
  ✓ IF PASS1_COUNT < 80 → Pass 2 MUST have executed
  ✓ Pass 2 identified gaps >40s → How many gaps found?
  ✓ Pass 2 added frames from gaps → How many frames added?
  ✓ Final count → PASS2_COUNT = ?
  ✓ Coverage check completed → Missing types addressed?
  ✓ Cluster detection completed → Sequences identified?
  ✓ All frames with 2+ objects include "Referential Grounding" type

═══════════════════════════════════════════════════════════════════════════════
PRIORITY ASSIGNMENT RULES
═══════════════════════════════════════════════════════════════════════════════

✓ High-quality highlights (≥0.85): Use actual highlight score
✓ Text detection frames: Use highlight score OR 0.75 minimum
✓ Object/people-rich frames (5+): Use highlight score OR 0.75 minimum
✓ Dense cluster frames: Assign 0.75
✓ Gap-filling frames (Pass 2): Must be ≥0.75 from highlights
✓ Coverage gap frames: Can use 0.70-0.75 if solving missing type
✓ Spurious Correlations: Can use 0.50-0.74 for true anomalies
✗ Never use priority <0.50 (indicates poor quality)

SPECIAL CASE: Video end frames
  • If unusual context (e.g., "black cat flying") for Spurious Correlations
  • Priority 0.50-0.60 acceptable IF clearly anomalous
  • BUT prefer priority ≥0.70 if available

═══════════════════════════════════════════════════════════════════════════════
EXAMPLES OF GOOD VS BAD REASONING
═══════════════════════════════════════════════════════════════════════════════

✓ GOOD: "Audio peak (0.92) + speaker pointing at slide with 'Budget' text → Audio-Visual Stitching + Needle"
✓ GOOD: "Scene change: conference_room → outdoor (scene_type) + 3 highlights within 5s → Subscene + dense cluster candidate"
✓ GOOD: "7 people detected (poses) at single timestamp → Counting"
✓ GOOD: "Empty stage (10s) vs full audience (45s) → Comparative"
✓ GOOD: "Gap 92s→136s (44 seconds) filled with priority 0.78 frame showing 5 people + high semantic (0.90)"

✗ AVOID: "Mid-section coverage" (vague, not evidence-based)
✗ AVOID: "Temporal distribution" (gap-filling without justification)
✗ AVOID: "Between highlights" (filler reasoning)
✗ AVOID: "Sequential question opportunity" (cycling through types without evidence)
✗ AVOID: "quality:0.0 indicates anomaly, priority 0.0" (too low, no value)

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT
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
      "reason": "Audio peak (0.88) + topic shift (semantic: 0.95) + clear BLIP-2 caption",
      "question_types": ["Audio-Visual Stitching", "Temporal Understanding"],
      "priority": 0.95
    }}
  ],
  "dense_clusters": [
    {{
      "start": 45.0,
      "end": 52.0,
      "frame_count": 7,
      "reason": "5 highlights within 7s, continuous action sequence, scene change midpoint"
    }}
  ],
  "coverage": {{
    "covered_types": <integer>,
    "total_types": 13,
    "missing_types": [<list of strings>],
    "coverage_ratio": <float>
  }}
}}

═══════════════════════════════════════════════════════════════════════════════
FINAL CHECKLIST (Verify Before Submitting)
═══════════════════════════════════════════════════════════════════════════════

Before returning JSON, verify ALL of these:

FRAME COUNT:
✓ execution_summary.pass1_count documented (how many after Pass 1)
✓ IF pass1_count < 80 → execution_summary.pass2_executed MUST be true
✓ execution_summary.pass2_count documented (how many after Pass 2)
✓ Total frames in single_frames = pass2_count (numbers match)
✓ Total frames between 80-120 (unless no qualifying frames available)

GAP ANALYSIS:
✓ execution_summary.gap_analysis shows gaps identified
✓ IF pass1_count < 80 → gap_analysis.qualifying_gaps_over_40s > 0
✓ IF qualifying gaps found → gap_analysis.gaps_filled > 0

QUALITY:
✓ Every frame has specific evidence-based reasoning (no generic phrases)
✓ Priority ≥0.75 for special case frames (text, objects, clusters)
✓ Priority ≥0.50 for all frames (no frames below this threshold)
✓ Question types assigned ONLY when objective criteria met

COVERAGE:
✓ ALL frames with 2+ objects include "Referential Grounding" type
✓ Dense clusters populated if sequences found (3+ frames <5s apart)
✓ All cluster frames tagged with "Subscene" type
✓ Coverage ≥11/13 types (85% minimum)

CRITICAL: If pass2_executed = false AND pass2_count < 80, you made an error.
Go back and execute Pass 2 properly before submitting JSON.

JSON:"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=10000,  # Increased to handle 150 frames (~100 tokens per frame)
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text.strip()

            # Strip markdown code fences if present
            if response_text.startswith('```'):
                # Remove opening fence (```json or ```)
                lines = response_text.split('\n')
                if lines[0].startswith('```'):
                    lines = lines[1:]
                # Remove closing fence
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                response_text = '\n'.join(lines).strip()

            # Parse JSON
            selection_plan = json.loads(response_text)

            # DEBUG: Check what keys Claude actually returned
            logger.info(f"DEBUG - Keys in Claude response: {list(selection_plan.keys())}")
            logger.info(f"DEBUG - Frames in 'single_frames': {len(selection_plan.get('single_frames', []))}")

            # Log execution summary
            exec_summary = selection_plan.get('execution_summary', {})
            pass1_count = exec_summary.get('pass1_count', 0)
            pass2_executed = exec_summary.get('pass2_executed', False)
            pass2_count = exec_summary.get('pass2_count', 0)
            frames_added = exec_summary.get('frames_added_pass2', 0)
            gap_analysis = exec_summary.get('gap_analysis', {})

            logger.info(f"Pass 1: {pass1_count} frames selected")
            logger.info(f"Pass 2 executed: {pass2_executed}")
            if pass2_executed:
                logger.info(f"Pass 2: Added {frames_added} frames → Total: {pass2_count}")
                logger.info(f"Gap analysis: {gap_analysis.get('qualifying_gaps_over_40s', 0)} qualifying gaps, {gap_analysis.get('gaps_filled', 0)} gaps filled")

            # Validate frame count
            single_count = len(selection_plan.get('single_frames', []))
            cluster_count = sum(c['frame_count'] for c in selection_plan.get('dense_clusters', []))
            total_frames = single_count + cluster_count

            min_target = max(47, int(frame_budget * 0.53))  # 80 frames for 150 budget
            max_target = int(frame_budget * 0.80)  # 120 frames for 150 budget

            # Verify execution_summary matches actual selection
            if pass2_count != total_frames:
                logger.warning(f"Mismatch: execution_summary.pass2_count={pass2_count} but single_frames+clusters has {total_frames} frames")

            # Verify Pass 2 was executed if needed
            if pass1_count < 80 and not pass2_executed:
                logger.error(f"CRITICAL ERROR: Pass 1 had {pass1_count} frames (<80) but Pass 2 was not executed!")
                logger.error("Claude did not follow the mandatory two-pass algorithm.")
                logger.error("Triggering fallback selection...")
                return self._fallback_selection(highlights, frame_budget, video_duration)

            if pass2_executed and pass2_count < 80:
                logger.warning(f"Pass 2 executed but only reached {pass2_count} frames (target: 80)")
                logger.warning("This may indicate insufficient qualifying frames in the video.")

            if total_frames < min_target:
                logger.warning(f"Claude selected {total_frames} frames (below target {min_target}-{max_target}). Accepting - may indicate low video quality or no qualifying frames.")
            elif total_frames > max_target:
                logger.warning(f"Claude selected {total_frames} frames (above target {min_target}-{max_target}). Accepting - prompt should handle this, but monitor for issues.")
            else:
                logger.info(f"Claude selected {total_frames} frames (within target range {min_target}-{max_target})")

            return selection_plan

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude response: {e}")
            logger.error(f"Response: {response_text}")
            # Fallback to simple selection
            return self._fallback_selection(highlights, frame_budget, video_duration)
        except Exception as e:
            logger.error(f"Error in Claude frame selection: {e}")
            return self._fallback_selection(highlights, frame_budget, video_duration)

    def _adjust_frame_count(self, selection_plan: Dict, target: int) -> Dict:
        """Adjust frame count to match target"""
        single_frames = selection_plan.get('single_frames', [])
        dense_clusters = selection_plan.get('dense_clusters', [])

        current_count = len(single_frames) + sum(c['frame_count'] for c in dense_clusters)

        if current_count < target:
            # Add more single frames from top of list
            diff = target - current_count
            logger.info(f"Adding {diff} frames to reach target")
            # Just duplicate some high-priority frames with small time offsets
            for i in range(diff):
                if single_frames:
                    base_frame = single_frames[i % len(single_frames)]
                    new_frame = base_frame.copy()
                    new_frame['timestamp'] += 0.5 * (i + 1)
                    single_frames.append(new_frame)

        elif current_count > target:
            # Remove excess single frames
            diff = current_count - target
            logger.info(f"Removing {diff} frames to reach target")
            single_frames = single_frames[:-diff]

        return {
            'single_frames': single_frames,
            'dense_clusters': dense_clusters
        }

    def _fallback_selection(
        self,
        highlights: List[Dict],
        frame_budget: int,
        video_duration: float
    ) -> Dict:
        """Fallback selection if Claude fails"""
        logger.warning("Using fallback frame selection")

        single_frames = []
        max_target = int(frame_budget * 0.80)  # 120 frames for 150 budget

        # Take top highlights (up to max_target)
        for i, h in enumerate(highlights[:max_target]):
            single_frames.append({
                'timestamp': h['timestamp'],
                'reason': f"Highlight rank {i+1}, score {h['combined_score']:.3f}",
                'question_types': ["General Holistic Reasoning"],
                'priority': h['combined_score']
            })

        # Only add evenly spaced frames if we have very few highlights
        min_acceptable = max(47, int(frame_budget * 0.53))  # 80 frames for 150 budget
        if len(single_frames) < min_acceptable:
            remaining = min_acceptable - len(single_frames)
            interval = video_duration / (remaining + 1)
            existing_timestamps = {f['timestamp'] for f in single_frames}

            for i in range(remaining):
                timestamp = interval * (i + 1)
                # Avoid duplicates (within 1 second)
                if not any(abs(timestamp - t) < 1.0 for t in existing_timestamps):
                    single_frames.append({
                        'timestamp': timestamp,
                        'reason': f"Temporal coverage: evenly spaced sample {i+1}/{remaining}",
                        'question_types': ["General Holistic Reasoning"],
                        'priority': 0.3  # Lower priority than highlights
                    })

            logger.info(f"Filled {remaining} additional frames to reach minimum (total: {len(single_frames)})")

        return {
            'single_frames': single_frames,
            'dense_clusters': []
        }

    def _validate_type_coverage(self, single_frames: List[Dict]) -> Dict:
        """Validate coverage of question types"""
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
        """
        Get top N highlights above minimum score (static method for pipeline)

        Args:
            highlights: List of highlight dicts from universal_highlight_detector
            top_n: Number of top highlights to return
            min_score: Minimum combined_score threshold

        Returns:
            List of top N highlights sorted by score (descending)
        """
        # Filter by minimum score
        filtered = [h for h in highlights if h.get('combined_score', 0) >= min_score]

        # Sort by combined_score descending
        sorted_highlights = sorted(
            filtered,
            key=lambda x: x.get('combined_score', 0),
            reverse=True
        )

        # Return top N
        return sorted_highlights[:top_n]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with mock data
    selector = LLMFrameSelector()

    mock_samples = [
        {
            'timestamp': 10.0,
            'blip2_caption': 'A person presenting in a conference room',
            'scene_type': 'conference_room',
            'objects': ['person', 'laptop', 'screen'],
            'text_detected': 'Welcome',
            'poses': [{'keypoints': []}],
            'quality': 1.0
        },
        {
            'timestamp': 30.0,
            'blip2_caption': 'Audience listening and taking notes',
            'scene_type': 'auditorium',
            'objects': ['person', 'chair', 'notebook'],
            'text_detected': '',
            'poses': [{'keypoints': []}, {'keypoints': []}],
            'quality': 0.8
        }
    ]

    mock_highlights = [
        {'timestamp': 10.0, 'combined_score': 0.9, 'audio_score': 0.8, 'visual_score': 0.9, 'semantic_score': 0.95},
        {'timestamp': 30.0, 'combined_score': 0.75, 'audio_score': 0.7, 'visual_score': 0.6, 'semantic_score': 0.8}
    ]

    mock_quality = {10.0: 1.0, 20.0: 0.8, 30.0: 0.8, 40.0: 0.5}

    result = selector.select_frames(
        mock_samples,
        mock_highlights,
        mock_quality,
        frame_budget=50,
        video_duration=60.0
    )

    print(f"\nSelection plan:")
    print(f"  Single frames: {len(result['selection_plan'])}")
    print(f"  Dense clusters: {len(result['dense_clusters'])}")
    print(f"  Type coverage: {result['coverage']['covered_types']}/{result['coverage']['total_types']}")