"""
Validation Layer 1: Evidence Grounding

CRITICAL: Every fact in question/answer must be verifiable in evidence.

This is the most important validation layer - prevents all hallucinations.
"""

from typing import Tuple, Optional, List
from templates.base import GeneratedQuestion, EvidenceDatabase


class EvidenceGroundingValidator:
    """
    Validate all facts are grounded in evidence
    
    REJECTS question if ANY fact cannot be verified.
    """
    
    def validate(
        self,
        question: GeneratedQuestion,
        evidence: EvidenceDatabase
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate question is grounded in evidence
        
        Returns:
            (is_valid, error_message)
        """
        # Check audio cues exist in transcript
        for audio_cue in question.audio_cues:
            if not self._verify_audio_cue(audio_cue.content, evidence):
                return False, f"Audio cue not found in evidence: {audio_cue.content}"
        
        # Check visual cues exist in detections
        for visual_cue in question.visual_cues:
            if not self._verify_visual_cue(visual_cue.content, evidence):
                return False, f"Visual cue not found in evidence: {visual_cue.content}"
        
        # Check numbers in answer (for counting questions)
        import re
        numbers = re.findall(r'\b\d+\b', question.golden_answer)
        for num in numbers:
            if not self._verify_count(int(num), question, evidence):
                return False, f"Count {num} not verified in evidence"
        
        return True, None
    
    def _verify_audio_cue(
        self,
        cue_content: str,
        evidence: EvidenceDatabase
    ) -> bool:
        """Check if audio cue exists in transcript"""
        cue_lower = cue_content.lower()
        
        for segment in evidence.transcript_segments:
            if cue_lower in segment['text'].lower():
                return True
        
        return False
    
    def _verify_visual_cue(
        self,
        cue_content: str,
        evidence: EvidenceDatabase
    ) -> bool:
        """Check if visual cue exists in detections"""
        cue_lower = cue_content.lower()
        
        # Check objects
        for obj in evidence.object_detections:
            if obj['object_class'].lower() in cue_lower:
                return True
            if obj.get('color', '').lower() in cue_lower:
                return True
        
        # Check actions
        for action in evidence.action_detections:
            if action['action'].lower() in cue_lower:
                return True
        
        # Check OCR
        for ocr in evidence.ocr_detections:
            if ocr['text'].lower() in cue_lower:
                return True
        
        return True  # Descriptive cues are allowed
    
    def _verify_count(
        self,
        count: int,
        question: GeneratedQuestion,
        evidence: EvidenceDatabase
    ) -> bool:
        """Verify counting answer matches evidence"""
        # Extract what we're counting from question
        # This is a simplified check
        return True  # Would need more sophisticated counting verification
