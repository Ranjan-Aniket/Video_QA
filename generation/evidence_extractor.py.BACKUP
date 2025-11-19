"""
Evidence Extractor - FIXED IMPLEMENTATION

Extracts structured evidence from RawVideoContext for question generation.

CRITICAL: This is the ONLY source of truth for questions.
NO hardcoding allowed - everything comes from actual video content.
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    """Single transcript segment with timing"""
    text: str
    start: float
    end: float
    confidence: float

@dataclass
class RawVideoContext:
    """Raw multimodal context from video processing"""
    video_id: str
    duration: float
    transcript: List[Dict]
    frames: List[Dict]
    audio_features: Dict
    metadata: Dict = field(default_factory=dict)
    words: List[Dict] = field(default_factory=list)


@dataclass
class PersonDetection:
    """Person detected in video"""
    timestamp: float
    bbox: tuple  # (x1, y1, x2, y2)
    confidence: float
    attributes: Dict = field(default_factory=dict)


@dataclass
class ObjectDetection:
    """Object detected in video"""
    object_class: str
    timestamp: float
    bbox: tuple
    confidence: float


@dataclass
class OCRDetection:
    """Text detected via OCR"""
    text: str
    timestamp: float
    bbox: List
    confidence: float


@dataclass
class SceneDetection:
    """Scene information"""
    scene_id: int
    start_time: float
    end_time: float
    scene_type: str


@dataclass
class EvidenceDatabase:
    """
    Structured evidence database for question generation
    
    This is what the question generation expects!
    """
    video_id: str
    duration: float
    
    # Transcript
    transcript_segments: List[TranscriptSegment] = field(default_factory=list)
    
    # Audio
    music_segments: List[Dict] = field(default_factory=list)
    sound_effects: List[Dict] = field(default_factory=list)
    ambient_sounds: List[Dict] = field(default_factory=list)
    tone_changes: List[Dict] = field(default_factory=list)
    
    # Visual
    person_detections: List[PersonDetection] = field(default_factory=list)
    object_detections: List[ObjectDetection] = field(default_factory=list)
    scene_detections: List[SceneDetection] = field(default_factory=list)
    ocr_detections: List[OCRDetection] = field(default_factory=list)
    action_detections: List[Dict] = field(default_factory=list)
    scene_changes: List[float] = field(default_factory=list)
    
    # Timeline
    event_timeline: List[Dict] = field(default_factory=list)
    
    # Names to block (per guidelines)
    character_names: List[str] = field(default_factory=list)
    team_names: List[str] = field(default_factory=list)
    media_names: List[str] = field(default_factory=list)
    brand_names: List[str] = field(default_factory=list)


class EvidenceExtractor:
    """
    FIXED: Extract structured evidence from RawVideoContext
    """
    
    def __init__(self):
        self.name_patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # John Smith
            r'\b[A-Z][a-z]+\b(?= says| said| does)',  # "John says"
        ]
        logger.info("EvidenceExtractor initialized")
    
    def extract(self, raw_context) -> EvidenceDatabase:
        """
        FIXED: Extract evidence from RawVideoContext
        
        Args:
            raw_context: RawVideoContext from VideoProcessor
            
        Returns:
            Structured EvidenceDatabase
        """
        logger.info(f"Extracting evidence from video {raw_context.video_id}")
        
        # Create evidence database
        evidence = EvidenceDatabase(
            video_id=raw_context.video_id,
            duration=raw_context.duration
        )
        
        # Extract transcript
        evidence.transcript_segments = self._extract_transcript(raw_context.transcript)
        logger.info(f"✓ Extracted {len(evidence.transcript_segments)} transcript segments")
        
        # Extract visual detections from frames
        person_count = 0
        object_count = 0
        ocr_count = 0
        
        for frame_dict in raw_context.frames:
            timestamp = frame_dict['timestamp']
            
            # Extract persons (filter person class from objects)
            for detection in frame_dict.get('detections', []):
                if detection['class'] == 'person':
                    evidence.person_detections.append(PersonDetection(
                        timestamp=timestamp,
                        bbox=tuple(detection['bbox']),
                        confidence=detection['confidence']
                    ))
                    person_count += 1
                else:
                    # Other objects
                    evidence.object_detections.append(ObjectDetection(
                        object_class=detection['class'],
                        timestamp=timestamp,
                        bbox=tuple(detection['bbox']),
                        confidence=detection['confidence']
                    ))
                    object_count += 1
            
            # Extract OCR results
            for ocr_result in frame_dict.get('ocr_results', []):
                evidence.ocr_detections.append(OCRDetection(
                    text=ocr_result['text'],
                    timestamp=timestamp,
                    bbox=ocr_result['bbox'],
                    confidence=ocr_result['confidence']
                ))
                ocr_count += 1
        
        logger.info(f"✓ Extracted {person_count} person detections")
        logger.info(f"✓ Extracted {object_count} object detections")
        logger.info(f"✓ Extracted {ocr_count} OCR detections")
        
        # Extract scenes from metadata
        if 'scenes' in raw_context.metadata:
            for i, (start, end) in enumerate(raw_context.metadata['scenes']):
                evidence.scene_detections.append(SceneDetection(
                    scene_id=i,
                    start_time=start,
                    end_time=end,
                    scene_type='main_content'
                ))
                evidence.scene_changes.append(start)
        logger.info(f"✓ Extracted {len(evidence.scene_detections)} scenes")
        
        # Extract audio features
        evidence.music_segments = raw_context.audio_features.get('music', [])
        evidence.sound_effects = raw_context.audio_features.get('sounds', [])
        evidence.ambient_sounds = raw_context.audio_features.get('ambient', [])
        
        # Build event timeline
        evidence.event_timeline = self._build_event_timeline(evidence)
        logger.info(f"✓ Built timeline with {len(evidence.event_timeline)} events")
        
        # Detect names to block
        evidence.character_names = self._detect_character_names(evidence)
        evidence.team_names = self._detect_team_names(evidence)
        evidence.media_names = self._detect_media_names(evidence)
        evidence.brand_names = self._detect_brand_names(evidence)
        
        blocked_names = (
            len(evidence.character_names) +
            len(evidence.team_names) +
            len(evidence.media_names) +
            len(evidence.brand_names)
        )
        logger.info(f"✓ Detected {blocked_names} names to block")
        
        logger.info(
            f"✓ Evidence extraction complete for {raw_context.video_id}: "
            f"{len(evidence.transcript_segments)} transcript, "
            f"{len(evidence.object_detections)} objects, "
            f"{len(evidence.ocr_detections)} OCR, "
            f"{len(evidence.scene_detections)} scenes"
        )
        
        return evidence
    
    def _extract_transcript(self, transcript: List[Dict]) -> List[TranscriptSegment]:
        """Extract transcript segments"""
        segments = []
        
        for segment in transcript:
            segments.append(TranscriptSegment(
                text=segment.get('text', ''),
                start=segment.get('start', 0.0),
                end=segment.get('end', 0.0),
                confidence=segment.get('confidence', 0.9),
                words=segment.get('words', [])
            ))
        
        return segments
    
    def _build_event_timeline(self, evidence: EvidenceDatabase) -> List[Dict]:
        """Build chronological event timeline"""
        events = []
        
        # Add transcript events
        for segment in evidence.transcript_segments:
            events.append({
                'timestamp': segment.start,
                'type': 'transcript',
                'data': segment.text
            })
        
        # Add object detection events
        for obj in evidence.object_detections:
            events.append({
                'timestamp': obj.timestamp,
                'type': 'object_detection',
                'data': obj.object_class
            })
        
        # Add OCR events
        for ocr in evidence.ocr_detections:
            events.append({
                'timestamp': ocr.timestamp,
                'type': 'ocr',
                'data': ocr.text
            })
        
        # Add scene changes
        for timestamp in evidence.scene_changes:
            events.append({
                'timestamp': timestamp,
                'type': 'scene_change',
                'data': None
            })
        
        # Sort by timestamp
        events.sort(key=lambda x: x['timestamp'])
        
        return events
    
    def _detect_character_names(self, evidence: EvidenceDatabase) -> List[str]:
        """Detect character names from transcript and OCR"""
        names = set()
        
        # Check transcript for name patterns
        for segment in evidence.transcript_segments:
            text = segment.text
            # Look for capitalized names
            matches = re.findall(r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b', text)
            for match in matches:
                # Filter out common words (this is simplified)
                if len(match) > 2 and match not in ['The', 'This', 'That', 'Then']:
                    names.add(match)
        
        # Check OCR for names
        for ocr in evidence.ocr_detections:
            text = ocr.text
            matches = re.findall(r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b', text)
            for match in matches:
                if len(match) > 2:
                    names.add(match)
        
        return list(names)
    
    def _detect_team_names(self, evidence: EvidenceDatabase) -> List[str]:
        """Detect team names from transcript and OCR"""
        team_keywords = ['team', 'club', 'fc', 'united', 'city', 'athletic']
        teams = set()
        
        # Check transcript
        for segment in evidence.transcript_segments:
            text = segment.text.lower()
            if any(keyword in text for keyword in team_keywords):
                # Extract team name
                matches = re.findall(r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b', segment.text)
                teams.update(matches)
        
        # Check OCR
        for ocr in evidence.ocr_detections:
            text = ocr.text.lower()
            if any(keyword in text for keyword in team_keywords):
                matches = re.findall(r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b', ocr.text)
                teams.update(matches)
        
        return list(teams)
    
    def _detect_media_names(self, evidence: EvidenceDatabase) -> List[str]:
        """Detect media names (movies, shows, brands)"""
        media = set()
        
        # Look for quoted titles in transcript
        for segment in evidence.transcript_segments:
            # Find text in quotes
            matches = re.findall(r'"([^"]+)"', segment.text)
            media.update(matches)
        
        return list(media)
    
    def _detect_brand_names(self, evidence: EvidenceDatabase) -> List[str]:
        """Detect brand names from OCR"""
        brands = set()
        
        # Check OCR for all-caps words (often brands)
        for ocr in evidence.ocr_detections:
            matches = re.findall(r'\b[A-Z]{2,}\b', ocr.text)
            brands.update(matches)
        
        return list(brands)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Evidence Extractor - FIXED")
    print("This now properly extracts evidence from RawVideoContext")
    print("Use with the fixed VideoProcessor to get actual data!")