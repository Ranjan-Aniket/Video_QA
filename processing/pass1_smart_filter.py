"""
Pass 1: Smart Pre-Filter using 3-Tier Selection

Filters ~1850 frames down to ~260 frames for Pass 2A using:
- Tier 1: Rule-based auto-keep (~90 frames)
- Tier 2: Wildcard sampling (90 frames)
- Tier 3: Sonnet 3.5 LLM selection (up to 80 frames)

Cost: ~$0.35 (Sonnet 3.5)
"""

import json
from typing import Dict, List, Set
from pathlib import Path
from loguru import logger
import anthropic
import os


class Pass1SmartFilter:
    """
    Three-tier frame pre-filtering system
    """

    def __init__(self):
        """Initialize Pass 1 filter"""
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def tier1_auto_keep(
        self,
        frames: List[Dict],
        audio_analysis: Dict,
        scenes: List[Dict],
        clip_analysis: Dict = None
    ) -> Set[int]:
        """
        Tier 1: Rule-based automatic keep

        Criteria:
        - OCR detected frames
        - Scene boundary frames ±2
        - Speaker change frames
        - Speech + high visual change frames
        - Audio energy spikes (non-speech)
        - Visual anomaly frames (from CLIP analysis)

        Args:
            frames: List of frame metadata dicts
            audio_analysis: Audio analysis results
            scenes: Scene detection results
            clip_analysis: Optional CLIP analysis results (for visual anomalies)

        Returns:
            Set of frame IDs to keep
        """
        logger.info("Tier 1: Rule-based auto-keep...")

        keep_frames = set()

        # 1. OCR detected frames
        # ✅ FIXED: Validate frame_id exists before accessing
        ocr_frames = [
            f['frame_id'] for f in frames
            if f.get('frame_id') is not None and f.get('text_detected')
        ]
        keep_frames.update(ocr_frames)
        logger.info(f"  OCR frames: {len(ocr_frames)}")

        # 2. Scene boundary frames ±2
        # ✅ FIXED: Validate timestamp and frame_id exist before accessing
        scene_boundary_frames = set()
        for scene in scenes:
            start_ts = scene['start']
            end_ts = scene['end']

            # Find frames near boundaries
            for f in frames:
                if f.get('timestamp') is None or f.get('frame_id') is None:
                    continue
                ts = f['timestamp']
                # Within 2 seconds of boundary
                if abs(ts - start_ts) < 2.0 or abs(ts - end_ts) < 2.0:
                    scene_boundary_frames.add(f['frame_id'])

        keep_frames.update(scene_boundary_frames)
        logger.info(f"  Scene boundary frames: {len(scene_boundary_frames)}")

        # 3. Speaker change frames
        speaker_change_frames = set()
        if 'segments' in audio_analysis:
            segments = audio_analysis['segments']
            for i in range(1, len(segments)):
                prev_speaker = segments[i-1].get('speaker', 'UNKNOWN')
                curr_speaker = segments[i].get('speaker', 'UNKNOWN')

                if prev_speaker != curr_speaker:
                    change_time = segments[i]['start']

                    # Find closest frame
                    # ✅ FIXED: Validate timestamp exists before accessing
                    valid_frames = [f for f in frames if f.get('timestamp') is not None]
                    if not valid_frames:
                        continue

                    closest_frame = min(
                        valid_frames,
                        key=lambda f: abs(f['timestamp'] - change_time)
                    )

                    # ✅ FIXED: Validate frame_id exists before accessing
                    if closest_frame.get('frame_id') is not None:
                        speaker_change_frames.add(closest_frame['frame_id'])

        keep_frames.update(speaker_change_frames)
        logger.info(f"  Speaker change frames: {len(speaker_change_frames)}")

        # 4. Speech + high visual change frames
        # TODO: Implement when frame delta scores are available
        # For now, use frames with high object count as proxy
        # ✅ FIXED: Validate frame_id exists before accessing
        high_activity_frames = [
            f['frame_id'] for f in frames
            if f.get('frame_id') is not None and len(f.get('objects', [])) >= 3
        ]
        keep_frames.update(high_activity_frames[:20])  # Cap at 20
        logger.info(f"  High activity frames: {min(len(high_activity_frames), 20)}")

        # 5. Audio energy spikes (non-speech)
        # TODO: Implement when audio energy data is available

        # ✅ ISSUE #2 FIX: Visual anomaly frames (from CLIP analysis)
        # Visual anomalies = frames that look different from their neighbors
        # These often indicate scene changes, unique moments, or interesting content
        if clip_analysis:
            visual_anomalies = set(clip_analysis.get('visual_anomalies', []))
            # Validate that anomaly frame IDs exist in frames list
            valid_anomalies = [
                fid for fid in visual_anomalies
                if any(f.get('frame_id') == fid for f in frames)
            ]
            keep_frames.update(valid_anomalies)
            logger.info(f"  Visual anomaly frames: {len(valid_anomalies)}")

        logger.info(f"Tier 1 total: {len(keep_frames)} frames")

        return keep_frames

    def tier2_wildcard_sampling(
        self,
        frames: List[Dict],
        video_duration: float,
        interval: float = 10.0
    ) -> Set[int]:
        """
        Tier 2: Wildcard sampling to catch subtle moments

        Args:
            frames: List of frame metadata dicts
            video_duration: Video duration in seconds
            interval: Sample 1 frame every N seconds

        Returns:
            Set of frame IDs to keep
        """
        logger.info("Tier 2: Wildcard sampling (1 frame per 10s)...")

        keep_frames = set()

        # Generate target timestamps
        target_timestamps = []
        current_time = 0.0
        while current_time < video_duration:
            target_timestamps.append(current_time)
            current_time += interval

        # Find closest frame to each target timestamp
        # ✅ FIXED: Validate timestamp and frame_id exist before accessing
        for target_ts in target_timestamps:
            valid_frames = [f for f in frames if f.get('timestamp') is not None]
            if not valid_frames:
                continue

            closest_frame = min(
                valid_frames,
                key=lambda f: abs(f['timestamp'] - target_ts)
            )

            if closest_frame.get('frame_id') is not None:
                keep_frames.add(closest_frame['frame_id'])

        logger.info(f"Tier 2 total: {len(keep_frames)} frames")

        return keep_frames

    def tier3_llm_selection(
        self,
        frames: List[Dict],
        audio_analysis: Dict,
        clip_analysis: Dict,
        already_selected: Set[int],
        budget: int = 80
    ) -> Set[int]:
        """
        Tier 3: Sonnet 3.5 LLM selection for high-potential moments

        Looks for:
        - Inference candidates (transcript shows causality)
        - Comparison candidates (before/after language)
        - High-density moments (multiple speakers, rapid dialogue)
        - CLIP anomalies

        Args:
            frames: List of frame metadata dicts
            audio_analysis: Audio analysis results
            clip_analysis: CLIP analysis results
            already_selected: Frame IDs already selected by Tier 1+2
            budget: Max frames to select

        Returns:
            Set of frame IDs to keep
        """
        logger.info(f"Tier 3: Sonnet 3.5 LLM selection (up to {budget} frames)...")

        # Build metadata summary for Sonnet 3.5
        # Only include frames NOT already selected
        # ✅ FIXED: Validate frame_id exists before accessing
        candidate_frames = [
            f for f in frames
            if f.get('frame_id') is not None and f['frame_id'] not in already_selected
        ]

        if not candidate_frames:
            logger.warning("No candidate frames for Tier 3")
            return set()

        # Prepare metadata
        metadata_summary = self._build_metadata_summary(
            candidate_frames,
            audio_analysis,
            clip_analysis
        )

        # Call Sonnet 3.5
        prompt = self._build_tier3_prompt(metadata_summary, budget)

        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Parse response
            response_text = response.content[0].text
            selected_frames = self._parse_llm_response(response_text, candidate_frames)

            logger.info(f"Tier 3 total: {len(selected_frames)} frames")

            # Track cost
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = (input_tokens * 3 / 1_000_000) + (output_tokens * 15 / 1_000_000)
            logger.info(f"Tier 3 cost: ${cost:.4f}")

            return selected_frames

        except Exception as e:
            logger.error(f"Tier 3 LLM selection failed: {e}")
            # Fallback: select top N frames by CLIP score
            return self._fallback_selection(candidate_frames, clip_analysis, budget)

    def _build_metadata_summary(
        self,
        frames: List[Dict],
        audio_analysis: Dict,
        clip_analysis: Dict
    ) -> str:
        """
        Build concise metadata summary for Sonnet 3.5

        Args:
            frames: Candidate frames
            audio_analysis: Audio analysis
            clip_analysis: CLIP analysis

        Returns:
            Formatted metadata string
        """
        summary_parts = []

        # Video overview
        summary_parts.append("VIDEO METADATA:")
        summary_parts.append(f"Duration: {audio_analysis.get('duration', 0):.1f}s")
        summary_parts.append(f"Candidate frames: {len(frames)}")
        summary_parts.append("")

        # Transcript segments with timestamps
        summary_parts.append("TRANSCRIPT (with timestamps):")
        if 'segments' in audio_analysis:
            for i, seg in enumerate(audio_analysis['segments'][:50]):  # Limit to 50
                text = seg['text'].strip()
                start = seg['start']
                speaker = seg.get('speaker', 'SPEAKER')
                summary_parts.append(f"[{start:.1f}s] {speaker}: {text}")
        summary_parts.append("")

        # Frame metadata (concise)
        # ✅ FIXED: Validate frame_id and timestamp exist before accessing
        summary_parts.append("FRAME METADATA:")
        for f in frames[:200]:  # Limit to 200 frames
            # Skip frames missing required fields
            if f.get('frame_id') is None or f.get('timestamp') is None:
                continue

            frame_id = f['frame_id']
            ts = f['timestamp']
            scene = f.get('scene_type', 'unknown')
            ocr = 'OCR' if f.get('text_detected') else ''
            obj_count = len(f.get('objects', []))

            # CLIP ontology scores
            ontology_scores = clip_analysis.get('ontology_scores', {}).get(frame_id, {})
            top_ontology = max(ontology_scores, key=ontology_scores.get) if ontology_scores else 'none'
            top_score = ontology_scores.get(top_ontology, 0) if ontology_scores else 0

            summary_parts.append(
                f"Frame {frame_id} @ {ts:.1f}s | {scene} | {obj_count} objs | {ocr} | "
                f"Top: {top_ontology} ({top_score:.2f})"
            )

        return "\n".join(summary_parts)

    def _build_tier3_prompt(self, metadata_summary: str, budget: int) -> str:
        """
        Build prompt for Sonnet 3.5 Tier 3 selection

        Args:
            metadata_summary: Formatted metadata
            budget: Max frames to select

        Returns:
            Prompt string
        """
        prompt = f"""You are analyzing video metadata to select high-potential frames for adversarial Q&A generation.

{metadata_summary}

TASK: Select UP TO {budget} frames that have the highest potential for creating adversarial questions.

PRIORITIZE frames that show:

1. INFERENCE CANDIDATES (25 frames):
   - Transcript mentions causality: "because", "why", "the reason", "due to"
   - Transcript mentions intent: "to", "in order to", "want to"
   - Frames where cause/effect might be visible

2. COMPARISON CANDIDATES (25 frames):
   - Transcript mentions change: "before", "after", "changed", "became", "now"
   - Transcript mentions differences: "but", "however", "instead", "different"
   - Frames where state changes might be visible

3. HIGH-DENSITY MOMENTS (20 frames):
   - Multiple speakers in quick succession
   - Rapid dialogue (many short utterances)
   - Complex scenes (high object count)
   - Multiple simultaneous events

4. CLIP ANOMALIES (10 frames):
   - High CLIP ontology scores (especially for inference, comparative, needle)
   - Frames that stand out visually

CONSTRAINTS:
- Only select frames from the provided candidate list
- Stay within budget of {budget} frames
- Avoid consecutive frames (spread selections across video)
- Prioritize diversity of ontology types

OUTPUT FORMAT:
Return a JSON array of selected frame IDs with reasons:

[
  {{
    "frame_id": 123,
    "timestamp": 45.2,
    "category": "inference",
    "reason": "Transcript says 'because' at 45.1s, frame shows action result"
  }},
  ...
]

Select UP TO {budget} frames. Focus on quality over quantity - it's okay to select fewer than {budget} if candidates are weak.
"""

        return prompt

    def _parse_llm_response(
        self,
        response_text: str,
        candidate_frames: List[Dict]
    ) -> Set[int]:
        """
        Parse Sonnet 3.5 response to extract selected frame IDs

        Args:
            response_text: LLM response
            candidate_frames: Candidate frames

        Returns:
            Set of selected frame IDs
        """
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if not json_match:
                logger.warning("No JSON array found in LLM response")
                return set()

            selections = json.loads(json_match.group(0))

            # Extract frame IDs
            selected_ids = set()
            for sel in selections:
                if isinstance(sel, dict) and 'frame_id' in sel:
                    selected_ids.add(sel['frame_id'])
                elif isinstance(sel, int):
                    selected_ids.add(sel)

            # Validate frame IDs exist in candidates
            # ✅ FIXED: Validate frame_id exists before accessing
            valid_ids = {f['frame_id'] for f in candidate_frames if f.get('frame_id') is not None}
            selected_ids = selected_ids & valid_ids

            return selected_ids

        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return set()

    def _fallback_selection(
        self,
        candidate_frames: List[Dict],
        clip_analysis: Dict,
        budget: int
    ) -> Set[int]:
        """
        Fallback selection using CLIP scores if LLM fails

        Args:
            candidate_frames: Candidate frames
            clip_analysis: CLIP analysis results
            budget: Max frames to select

        Returns:
            Set of frame IDs
        """
        logger.info("Using fallback selection based on CLIP scores...")

        ontology_scores = clip_analysis.get('ontology_scores', {})

        # Score each frame by max ontology score
        # ✅ FIXED: Validate frame_id exists before accessing
        scored_frames = []
        for f in candidate_frames:
            if f.get('frame_id') is None:
                continue
            frame_id = f['frame_id']
            scores = ontology_scores.get(frame_id, {})
            max_score = max(scores.values()) if scores else 0
            scored_frames.append((frame_id, max_score))

        # Sort by score and take top N
        scored_frames.sort(key=lambda x: x[1], reverse=True)
        selected = set(fid for fid, _ in scored_frames[:budget])

        return selected

    def filter_frames(
        self,
        frames: List[Dict],
        audio_analysis: Dict,
        clip_analysis: Dict,
        scenes: List[Dict],
        video_duration: float
    ) -> Dict:
        """
        Run full 3-tier filtering pipeline

        Args:
            frames: All frame metadata (~1850 frames)
            audio_analysis: Audio analysis results
            clip_analysis: CLIP analysis results
            scenes: Scene detection results
            video_duration: Video duration in seconds

        Returns:
            {
                'selected_frames': List of selected frame dicts,
                'selection_breakdown': {
                    'tier1': int,
                    'tier2': int,
                    'tier3': int,
                    'total': int
                },
                'cost': float
            }
        """
        logger.info("=" * 60)
        logger.info("PASS 1: Smart Pre-Filter (3-Tier Selection)")
        logger.info("=" * 60)

        # Tier 1: Rule-based auto-keep
        tier1_frames = self.tier1_auto_keep(frames, audio_analysis, scenes, clip_analysis)

        # Tier 2: Wildcard sampling
        tier2_frames = self.tier2_wildcard_sampling(frames, video_duration)

        # Combine Tier 1 + Tier 2
        already_selected = tier1_frames | tier2_frames

        # Tier 3: LLM selection
        tier3_frames = self.tier3_llm_selection(
            frames,
            audio_analysis,
            clip_analysis,
            already_selected,
            budget=80
        )

        # Combine all tiers
        all_selected = tier1_frames | tier2_frames | tier3_frames

        # Get full frame metadata for selected frames
        # ✅ FIXED: Validate frame_id exists before accessing
        selected_frame_data = [
            f for f in frames
            if f.get('frame_id') is not None and f['frame_id'] in all_selected
        ]

        # Sort by timestamp
        # ✅ FIXED: Validate timestamp exists before sorting
        selected_frame_data = [f for f in selected_frame_data if f.get('timestamp') is not None]
        selected_frame_data.sort(key=lambda x: x['timestamp'])

        logger.info("=" * 60)
        logger.info(f"Pass 1 Complete: Selected {len(all_selected)} / {len(frames)} frames")
        logger.info(f"  Tier 1 (rules): {len(tier1_frames)}")
        logger.info(f"  Tier 2 (wildcard): {len(tier2_frames)}")
        logger.info(f"  Tier 3 (LLM): {len(tier3_frames)}")
        logger.info("=" * 60)

        return {
            'selected_frames': selected_frame_data,
            'selection_breakdown': {
                'tier1': len(tier1_frames),
                'tier2': len(tier2_frames),
                'tier3': len(tier3_frames),
                'total': len(all_selected)
            },
            'cost': 0.35  # Approximate cost for Sonnet 3.5
        }


def run_pass1_filter(
    frames: List[Dict],
    audio_analysis: Dict,
    clip_analysis: Dict,
    scenes: List[Dict],
    video_duration: float,
    output_path: str
) -> Dict:
    """
    Run Pass 1 smart filter and save results

    Args:
        frames: All frame metadata
        audio_analysis: Audio analysis results
        clip_analysis: CLIP analysis results
        scenes: Scene detection results
        video_duration: Video duration in seconds
        output_path: Path to save results

    Returns:
        Filter results dict
    """
    filter_engine = Pass1SmartFilter()

    results = filter_engine.filter_frames(
        frames,
        audio_analysis,
        clip_analysis,
        scenes,
        video_duration
    )

    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Pass 1 results saved to {output_path}")

    return results
