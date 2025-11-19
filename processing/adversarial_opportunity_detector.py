"""
Adversarial Opportunity Detector - Phase 2 of Smart Pipeline

Analyzes transcript to identify adversarial opportunities that expose Gemini's weaknesses:
1. Temporal markers ("before", "after", "when" phrases)
2. Ambiguous references ("he", "that", "it" with multiple visual candidates)
3. Counting opportunities (repeated events with boundaries)
4. Sequential events (multiple actions in sequence)
5. Context-rich moments (background/foreground descriptions)
6. Complex audio-visual sync points (7 premium keyframes for GPT-4V/Claude)

Uses GPT-4 for intelligent opportunity mining based on transcript analysis.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import os
import openai

logger = logging.getLogger(__name__)


@dataclass
class TemporalMarker:
    """Temporal marker opportunity (before/after/when phrases)"""
    quote: str
    timestamp: float
    type: str  # "before", "after", "when", "right before", "immediately after"
    complexity: str  # "low", "medium", "high"
    context: str  # Surrounding context


@dataclass
class AmbiguousReference:
    """Ambiguous reference opportunity (he/that/it with multiple candidates)"""
    quote: str
    timestamp: float
    ambiguity_score: float  # 0.0 to 1.0
    possible_referents: List[str]
    context: str


@dataclass
class CountingOpportunity:
    """Counting opportunity with temporal boundaries"""
    event_type: str
    boundary_quote: str
    boundary_timestamp: float
    complexity: str  # "low", "medium", "high"
    estimated_count: Optional[int] = None


@dataclass
class SequentialEvent:
    """Sequential events opportunity"""
    events: List[str]
    start: float
    end: float
    complexity: str


@dataclass
class ContextRichFrame:
    """Context-rich frame for background/foreground analysis"""
    timestamp: float
    audio_cue: str
    background_importance: str  # "low", "medium", "high"
    description: str


@dataclass
class AdversarialOpportunities:
    """Complete adversarial opportunities detected in transcript"""
    video_id: str
    transcript_duration: float

    # Opportunities by type
    temporal_markers: List[TemporalMarker] = field(default_factory=list)
    ambiguous_references: List[AmbiguousReference] = field(default_factory=list)
    counting_opportunities: List[CountingOpportunity] = field(default_factory=list)
    sequential_events: List[SequentialEvent] = field(default_factory=list)
    context_rich_frames: List[ContextRichFrame] = field(default_factory=list)

    # Top 7 keyframes for premium model analysis (GPT-4V/Claude)
    premium_analysis_keyframes: List[float] = field(default_factory=list)

    # Cost tracking
    detection_cost: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "video_id": self.video_id,
            "transcript_duration": self.transcript_duration,
            "temporal_markers": [
                {
                    "quote": tm.quote,
                    "timestamp": tm.timestamp,
                    "type": tm.type,
                    "complexity": tm.complexity,
                    "context": tm.context
                } for tm in self.temporal_markers
            ],
            "ambiguous_references": [
                {
                    "quote": ar.quote,
                    "timestamp": ar.timestamp,
                    "ambiguity_score": ar.ambiguity_score,
                    "possible_referents": ar.possible_referents,
                    "context": ar.context
                } for ar in self.ambiguous_references
            ],
            "counting_opportunities": [
                {
                    "event_type": co.event_type,
                    "boundary_quote": co.boundary_quote,
                    "boundary_timestamp": co.boundary_timestamp,
                    "complexity": co.complexity,
                    "estimated_count": co.estimated_count
                } for co in self.counting_opportunities
            ],
            "sequential_events": [
                {
                    "events": se.events,
                    "start": se.start,
                    "end": se.end,
                    "complexity": se.complexity
                } for se in self.sequential_events
            ],
            "context_rich_frames": [
                {
                    "timestamp": crf.timestamp,
                    "audio_cue": crf.audio_cue,
                    "background_importance": crf.background_importance,
                    "description": crf.description
                } for crf in self.context_rich_frames
            ],
            "premium_analysis_keyframes": self.premium_analysis_keyframes,
            "detection_cost": self.detection_cost,
            "summary": {
                "temporal_markers_count": len(self.temporal_markers),
                "ambiguous_references_count": len(self.ambiguous_references),
                "counting_opportunities_count": len(self.counting_opportunities),
                "sequential_events_count": len(self.sequential_events),
                "context_rich_frames_count": len(self.context_rich_frames),
                "premium_keyframes_count": len(self.premium_analysis_keyframes)
            }
        }


class AdversarialOpportunityDetector:
    """
    Detect adversarial opportunities in video transcript using GPT-4.

    This is Phase 2 of the smart pipeline - analyzes transcript to find
    moments that will expose Gemini's weaknesses in audio-visual understanding.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4-turbo-preview"
    ):
        """
        Initialize adversarial opportunity detector.

        Args:
            openai_api_key: OpenAI API key (or use OPENAI_API_KEY env var)
            model: GPT-4 model to use
        """
        # Set API key
        if openai_api_key:
            openai.api_key = openai_api_key
        elif os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass openai_api_key")

        self.model = model
        self.total_cost = 0.0

        logger.info(f"AdversarialOpportunityDetector initialized (model: {model})")

    def detect_opportunities(
        self,
        audio_analysis: Dict,
        video_id: str = "unknown"
    ) -> AdversarialOpportunities:
        """
        Analyze transcript to detect all adversarial opportunities.

        Args:
            audio_analysis: Output from audio_analysis.py
            video_id: Video identifier

        Returns:
            AdversarialOpportunities object with all detected opportunities
        """
        logger.info("=" * 80)
        logger.info("ADVERSARIAL OPPORTUNITY DETECTION - PHASE 2")
        logger.info("=" * 80)

        # Prepare transcript for GPT-4 analysis
        transcript = self._prepare_transcript(audio_analysis)
        duration = audio_analysis.get("duration", 0.0)

        logger.info(f"Video ID: {video_id}")
        logger.info(f"Duration: {duration:.1f}s")
        logger.info(f"Transcript length: {len(transcript)} chars")

        # Use GPT-4 to analyze transcript
        logger.info("\n🤖 Analyzing transcript with GPT-4...")
        opportunities_data = self._analyze_with_gpt4(transcript, duration, audio_analysis)

        # Parse GPT-4 response into structured opportunities
        opportunities = self._parse_gpt4_response(
            opportunities_data,
            video_id,
            duration
        )

        # Calculate cost (GPT-4 Turbo: ~$0.01 per 1K tokens input, ~$0.03 per 1K tokens output)
        # Estimate: ~1000 input tokens, ~500 output tokens = ~$0.025
        opportunities.detection_cost = 0.02
        self.total_cost += opportunities.detection_cost

        logger.info("=" * 80)
        logger.info("✅ ADVERSARIAL OPPORTUNITY DETECTION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Temporal markers: {len(opportunities.temporal_markers)}")
        logger.info(f"Ambiguous references: {len(opportunities.ambiguous_references)}")
        logger.info(f"Counting opportunities: {len(opportunities.counting_opportunities)}")
        logger.info(f"Sequential events: {len(opportunities.sequential_events)}")
        logger.info(f"Context-rich frames: {len(opportunities.context_rich_frames)}")
        logger.info(f"Premium keyframes: {len(opportunities.premium_analysis_keyframes)}")
        logger.info(f"Detection cost: ${opportunities.detection_cost:.4f}")
        logger.info("=" * 80)

        return opportunities

    def _prepare_transcript(self, audio_analysis: Dict) -> str:
        """
        Prepare transcript with timestamps for GPT-4 analysis.

        Args:
            audio_analysis: Audio analysis results

        Returns:
            Formatted transcript string
        """
        segments = audio_analysis.get("segments", [])

        # Format as: [HH:MM:SS] SPEAKER: text
        formatted_lines = []
        for segment in segments:
            timestamp = self._format_timestamp(segment["start"])
            speaker = segment.get("speaker", "SPEAKER_00")
            text = segment["text"]
            formatted_lines.append(f"[{timestamp}] {speaker}: {text}")

        return "\n".join(formatted_lines)

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as MM:SS"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def _analyze_with_gpt4(
        self,
        transcript: str,
        duration: float,
        audio_analysis: Dict
    ) -> Dict:
        """
        Analyze transcript using GPT-4 to find adversarial opportunities.

        Args:
            transcript: Formatted transcript
            duration: Video duration in seconds
            audio_analysis: Complete audio analysis

        Returns:
            GPT-4 response as dict
        """
        prompt = f"""Analyze this video transcript for adversarial question opportunities that would expose Gemini 2.0 Flash's weaknesses in audio-visual understanding.

VIDEO DURATION: {duration:.1f} seconds

TRANSCRIPT:
{transcript}

Your task is to identify specific opportunities for adversarial questions in these categories:

1. **TEMPORAL MARKERS**: Find quotes with "before", "after", "when", "right before", "immediately after" that reference visual events. Return 8-10 examples.

2. **AMBIGUOUS REFERENCES**: Find pronouns or demonstratives ("he", "that", "it", "this") where multiple visual candidates could exist, making it ambiguous what the speaker refers to. Return 5-7 examples.

3. **COUNTING OPPORTUNITIES**: Find repeated events with clear temporal boundaries (audio cues that mark start/end). Example: "How many times did X happen before the announcer says Y?" Return 4-5 examples.

4. **SEQUENTIAL EVENTS**: Find descriptions of multiple actions happening in sequence that can be tested. Return 4-5 examples.

5. **CONTEXT-RICH MOMENTS**: Find moments where audio describes background/foreground elements that would be visible on screen. Return 4-5 examples.

6. **PREMIUM KEYFRAMES**: Identify exactly 7 timestamps where audio-visual analysis would be MOST challenging for Gemini. These should be moments of:
   - Complex ambiguous references
   - Spurious correlations (unexpected connections)
   - Needle-in-haystack details
   - Multi-modal inference requirements
   - Holistic reasoning needs

Return your analysis as valid JSON with this structure:
{{
  "temporal_markers": [
    {{
      "quote": "exact quote from transcript",
      "timestamp": 45.2,
      "type": "before|after|when",
      "complexity": "low|medium|high",
      "context": "brief description of what makes this adversarial"
    }}
  ],
  "ambiguous_references": [
    {{
      "quote": "exact quote",
      "timestamp": 67.3,
      "ambiguity_score": 0.9,
      "possible_referents": ["person", "object1", "object2"],
      "context": "why this is ambiguous"
    }}
  ],
  "counting_opportunities": [
    {{
      "event_type": "scoring|action|event",
      "boundary_quote": "exact quote that marks boundary",
      "boundary_timestamp": 301.0,
      "complexity": "low|medium|high",
      "estimated_count": null
    }}
  ],
  "sequential_events": [
    {{
      "events": ["action1", "action2", "action3"],
      "start": 120.0,
      "end": 135.0,
      "complexity": "medium"
    }}
  ],
  "context_rich_frames": [
    {{
      "timestamp": 89.5,
      "audio_cue": "when narrator explains X",
      "background_importance": "high",
      "description": "background elements likely visible"
    }}
  ],
  "premium_analysis_keyframes": [67.3, 120.5, 197.0, 245.8, 301.2, 389.4, 456.7]
}}

CRITICAL: Return ONLY valid JSON, no explanatory text before or after."""

        try:
            logger.info("Sending request to GPT-4...")
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing video transcripts to find adversarial opportunities that expose AI model weaknesses. You return only valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=2000
            )

            # Extract response
            content = response.choices[0].message.content.strip()

            # Remove markdown code blocks if present
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            logger.info("✓ GPT-4 analysis complete")

            # Parse JSON
            result = json.loads(content)
            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse GPT-4 response as JSON: {e}")
            logger.error(f"Response was: {content[:500]}")
            # Return empty structure
            return {
                "temporal_markers": [],
                "ambiguous_references": [],
                "counting_opportunities": [],
                "sequential_events": [],
                "context_rich_frames": [],
                "premium_analysis_keyframes": []
            }
        except Exception as e:
            logger.error(f"GPT-4 analysis failed: {e}")
            raise

    def _parse_gpt4_response(
        self,
        data: Dict,
        video_id: str,
        duration: float
    ) -> AdversarialOpportunities:
        """
        Parse GPT-4 JSON response into AdversarialOpportunities object.

        Args:
            data: GPT-4 response dict
            video_id: Video ID
            duration: Video duration

        Returns:
            AdversarialOpportunities object
        """
        opportunities = AdversarialOpportunities(
            video_id=video_id,
            transcript_duration=duration
        )

        # Parse temporal markers
        for tm in data.get("temporal_markers", []):
            opportunities.temporal_markers.append(TemporalMarker(
                quote=tm["quote"],
                timestamp=tm["timestamp"],
                type=tm["type"],
                complexity=tm.get("complexity", "medium"),
                context=tm.get("context", "")
            ))

        # Parse ambiguous references
        for ar in data.get("ambiguous_references", []):
            opportunities.ambiguous_references.append(AmbiguousReference(
                quote=ar["quote"],
                timestamp=ar["timestamp"],
                ambiguity_score=ar.get("ambiguity_score", 0.8),
                possible_referents=ar.get("possible_referents", []),
                context=ar.get("context", "")
            ))

        # Parse counting opportunities
        for co in data.get("counting_opportunities", []):
            opportunities.counting_opportunities.append(CountingOpportunity(
                event_type=co["event_type"],
                boundary_quote=co["boundary_quote"],
                boundary_timestamp=co["boundary_timestamp"],
                complexity=co.get("complexity", "medium"),
                estimated_count=co.get("estimated_count")
            ))

        # Parse sequential events
        for se in data.get("sequential_events", []):
            opportunities.sequential_events.append(SequentialEvent(
                events=se["events"],
                start=se["start"],
                end=se["end"],
                complexity=se.get("complexity", "medium")
            ))

        # Parse context-rich frames
        for crf in data.get("context_rich_frames", []):
            opportunities.context_rich_frames.append(ContextRichFrame(
                timestamp=crf["timestamp"],
                audio_cue=crf["audio_cue"],
                background_importance=crf.get("background_importance", "medium"),
                description=crf.get("description", "")
            ))

        # Parse premium keyframes (ensure exactly 7)
        keyframes = data.get("premium_analysis_keyframes", [])
        opportunities.premium_analysis_keyframes = keyframes[:7]

        # If less than 7, distribute evenly across video duration
        if len(opportunities.premium_analysis_keyframes) < 7:
            logger.warning(f"Only {len(opportunities.premium_analysis_keyframes)} keyframes found, generating {7 - len(opportunities.premium_analysis_keyframes)} more")
            step = duration / 8
            for i in range(7 - len(opportunities.premium_analysis_keyframes)):
                opportunities.premium_analysis_keyframes.append(step * (i + 1))

        return opportunities

    def save_opportunities(
        self,
        opportunities: AdversarialOpportunities,
        output_path: Path
    ):
        """
        Save opportunities to JSON file.

        Args:
            opportunities: AdversarialOpportunities object
            output_path: Path to save JSON
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(opportunities.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Saved opportunities to: {output_path}")


# Test function
def test_opportunity_detector(video_path: str):
    """
    Test adversarial opportunity detector.

    Args:
        video_path: Path to video file
    """
    from processing.audio_analysis import AudioAnalyzer

    # Step 1: Audio analysis
    print("Step 1: Running audio analysis...")
    analyzer = AudioAnalyzer(video_path)
    audio_analysis = analyzer.analyze(save_json=True)

    # Step 2: Opportunity detection
    print("\nStep 2: Detecting adversarial opportunities...")
    detector = AdversarialOpportunityDetector()
    opportunities = detector.detect_opportunities(
        audio_analysis,
        video_id=Path(video_path).stem
    )

    # Step 3: Save results
    output_path = Path(video_path).parent / f"{Path(video_path).stem}_opportunities.json"
    detector.save_opportunities(opportunities, output_path)

    # Print summary
    print("\n" + "=" * 80)
    print("ADVERSARIAL OPPORTUNITIES DETECTED")
    print("=" * 80)
    print(f"Temporal markers: {len(opportunities.temporal_markers)}")
    print(f"Ambiguous references: {len(opportunities.ambiguous_references)}")
    print(f"Counting opportunities: {len(opportunities.counting_opportunities)}")
    print(f"Sequential events: {len(opportunities.sequential_events)}")
    print(f"Context-rich frames: {len(opportunities.context_rich_frames)}")
    print(f"Premium keyframes: {opportunities.premium_analysis_keyframes}")
    print(f"Cost: ${opportunities.detection_cost:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_opportunity_detector(sys.argv[1])
    else:
        print("Usage: python adversarial_opportunity_detector.py <video_path>")
