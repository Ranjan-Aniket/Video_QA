"""
LLM Semantic Detector - Phase 3: Detect semantic highlights using Claude

Uses Claude Sonnet 4.5 to detect:
- Topic shifts
- Emphasis markers (tone, energy)
- Important information
- Key moments

NO HARDCODED KEYWORDS - fully domain-agnostic LLM analysis
"""

import anthropic
import logging
from typing import List, Dict, Optional
import os

logger = logging.getLogger(__name__)


class LLMSemanticDetector:
    """Detect semantic highlights using Claude (NO hardcoded keywords)"""

    def __init__(self, anthropic_api_key: Optional[str] = None):
        """Initialize Claude client"""
        api_key = anthropic_api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY required")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-5-20250929"  # Fixed: dashes not periods

    def detect_semantic_highlights(
        self,
        transcript_segments: List[Dict]
    ) -> List[Dict]:
        """
        Detect semantic highlights from transcript.

        Args:
            transcript_segments: List of {'start': float, 'end': float, 'text': str}

        Returns:
            List of highlight dicts with timestamps and confidence scores
        """
        if not transcript_segments:
            logger.warning("No transcript segments provided")
            return []

        highlights = []

        # Analyze in chunks (10 segments at a time for context)
        chunk_size = 10
        for i in range(0, len(transcript_segments), chunk_size // 2):  # 50% overlap
            chunk = transcript_segments[i:i + chunk_size]
            chunk_highlights = self._analyze_chunk(chunk)
            highlights.extend(chunk_highlights)

        # Deduplicate highlights within 5 seconds
        highlights = self._deduplicate_highlights(highlights, window=5.0)

        logger.info(f"Detected {len(highlights)} semantic highlights")

        return highlights

    def _analyze_chunk(self, segments: List[Dict]) -> List[Dict]:
        """Analyze chunk of transcript for semantic highlights"""

        # Build transcript text with timestamps
        transcript_text = "\n".join([
            f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}"
            for seg in segments
        ])

        prompt = f"""Analyze this transcript segment and identify SEMANTIC HIGHLIGHTS.

Transcript:
{transcript_text}

Identify moments that are semantically important, such as:
1. Topic shifts or transitions to new subjects
2. Emphasis markers (speaker expressing strong emotion, urgency, or importance)
3. Key information being revealed or explained
4. Pivotal moments in the narrative or discussion

For each highlight, provide:
- timestamp (in seconds, use the START time of the segment)
- type (topic_shift, emphasis, key_info, or pivotal_moment)
- confidence (0.0-1.0)
- reason (brief explanation)

Return ONLY valid JSON array format:
[
  {{"timestamp": 12.5, "type": "topic_shift", "confidence": 0.9, "reason": "Transition from intro to main topic"}},
  {{"timestamp": 45.2, "type": "emphasis", "confidence": 0.85, "reason": "Strong emphasis on key point"}}
]

If NO highlights found, return empty array: []

JSON array:"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse Claude's response
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

            # Extract JSON array
            import json
            highlights = json.loads(response_text)

            # Add signal type
            for h in highlights:
                h['signal'] = 'semantic'

            return highlights

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude response: {e}")
            logger.error(f"Response: {response_text}")
            return []
        except Exception as e:
            logger.error(f"Error in LLM semantic detection: {e}")
            return []

    def _deduplicate_highlights(
        self,
        highlights: List[Dict],
        window: float = 5.0
    ) -> List[Dict]:
        """Remove duplicate highlights within time window"""
        if not highlights:
            return []

        # Sort by timestamp
        sorted_highlights = sorted(highlights, key=lambda x: x['timestamp'])

        deduplicated = []
        last_timestamp = -float('inf')

        for highlight in sorted_highlights:
            timestamp = highlight['timestamp']

            # If more than window seconds since last, add it
            if timestamp - last_timestamp >= window:
                deduplicated.append(highlight)
                last_timestamp = timestamp
            else:
                # Within window - keep higher confidence
                if deduplicated and highlight['confidence'] > deduplicated[-1]['confidence']:
                    deduplicated[-1] = highlight

        return deduplicated


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with mock transcript
    detector = LLMSemanticDetector()

    mock_segments = [
        {'start': 0.0, 'end': 5.0, 'text': "Welcome to today's presentation."},
        {'start': 5.0, 'end': 10.0, 'text': "We'll be discussing our new product launch."},
        {'start': 10.0, 'end': 15.0, 'text': "This is extremely important for our company's future."},
        {'start': 15.0, 'end': 20.0, 'text': "Let me show you the key features."},
    ]

    highlights = detector.detect_semantic_highlights(mock_segments)

    print(f"\nDetected {len(highlights)} semantic highlights:")
    for h in highlights:
        print(f"  {h['timestamp']:.1f}s - {h['type']} (conf: {h['confidence']:.2f})")
        print(f"    Reason: {h['reason']}")
