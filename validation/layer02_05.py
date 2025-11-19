"""
Validation Layers 2-5

Layer 2: Dual Cue Check
Layer 3: Name Detection
Layer 4: Timestamp Validation
Layer 5: Single Cue Answerable Check
"""

from typing import Tuple, Optional
from templates.base import GeneratedQuestion, EvidenceDatabase
import re


class DualCueValidator:
    """Layer 2: Ensure BOTH audio and visual cues present"""
    
    def validate(
        self,
        question: GeneratedQuestion,
        evidence: EvidenceDatabase
    ) -> Tuple[bool, Optional[str]]:
        """Validate both cue types present"""
        
        if not question.audio_cues:
            return False, "No audio cues present"
        
        if not question.visual_cues:
            return False, "No visual cues present"
        
        # Check cues are actually referenced in question
        question_lower = question.question_text.lower()
        
        # Audio should be in quotes or mentioned
        has_audio_ref = any(
            cue.content.lower() in question_lower
            for cue in question.audio_cues
        )
        
        # Visual should be mentioned
        has_visual_ref = any(
            cue.content.lower() in question_lower
            for cue in question.visual_cues
        )
        
        if not has_audio_ref:
            return False, "Audio cue not referenced in question"
        
        if not has_visual_ref:
            return False, "Visual cue not referenced in question"
        
        return True, None


class NameDetectionValidator:
    """Layer 3: Ensure NO names used"""
    
    def validate(
        self,
        question: GeneratedQuestion,
        evidence: EvidenceDatabase
    ) -> Tuple[bool, Optional[str]]:
        """Validate no names in question/answer"""
        
        all_names = (
            evidence.character_names +
            evidence.team_names +
            evidence.media_names +
            evidence.brand_names
        )
        
        # Check question
        question_lower = question.question_text.lower()
        for name in all_names:
            if name.lower() in question_lower:
                return False, f"Name detected in question: {name}"
        
        # Check answer
        answer_lower = question.golden_answer.lower()
        for name in all_names:
            if name.lower() in answer_lower:
                return False, f"Name detected in answer: {name}"
        
        # Check for capitalized words that might be names
        words = question.question_text.split() + question.golden_answer.split()
        for word in words:
            # Skip first word and words after punctuation
            if word[0].isupper() and len(word) > 1:
                # Check if it's a descriptor word (allowed)
                descriptor_words = {'Person', 'Man', 'Woman', 'Child', 'Player', 'Team'}
                if word not in descriptor_words and not word.endswith('s'):
                    # Might be a name
                    if word not in ['When', 'What', 'Where', 'Who', 'Why', 'How', 'The', 'A']:
                        # Potential name - be strict
                        pass  # Would need NER here
        
        return True, None


class TimestampValidator:
    """Layer 4: Validate timestamps are precise"""
    
    def validate(
        self,
        question: GeneratedQuestion,
        evidence: EvidenceDatabase
    ) -> Tuple[bool, Optional[str]]:
        """Validate timestamps cover cues + actions"""
        
        # Check timestamps are valid
        if question.start_timestamp >= question.end_timestamp:
            return False, "Invalid timestamp range"
        
        if question.start_timestamp < 0:
            return False, "Negative start timestamp"
        
        if question.end_timestamp > evidence.duration:
            return False, f"End timestamp exceeds video duration ({evidence.duration}s)"
        
        # Check all cues are within range
        for cue in question.audio_cues + question.visual_cues:
            if not (question.start_timestamp <= cue.timestamp <= question.end_timestamp):
                return False, f"Cue timestamp {cue.timestamp} outside range [{question.start_timestamp}, {question.end_timestamp}]"
        
        # Check timestamps aren't too long (>30s suspicious)
        duration = question.end_timestamp - question.start_timestamp
        if duration > 30.0:
            return False, f"Timestamp range too long: {duration}s (max 30s)"
        
        return True, None


class SingleCueAnswerableValidator:
    """Layer 5: Ensure question requires BOTH cues to answer"""
    
    def validate(
        self,
        question: GeneratedQuestion,
        evidence: EvidenceDatabase
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if question can be answered with just one cue type
        
        Per guidelines: "Even if the question has both audio and video cues, 
        but if it can be answered with just one cue, the pair needs to be rejected"
        """
        
        question_lower = question.question_text.lower()
        
        # Check if question explicitly requires both cues
        # E.g., "When X says Y, what Z happens" requires both audio (says Y) and visual (Z)
        
        # Pattern: "when...says...what" indicates both needed
        if 'when' in question_lower and 'says' in question_lower:
            return True, None  # Both needed
        
        # Pattern: "after...what appears" indicates both needed
        if 'after' in question_lower and ('appears' in question_lower or 'visible' in question_lower):
            return True, None
        
        # If question only asks about audio ("what does X say")
        if 'say' in question_lower and 'when' not in question_lower:
            return False, "Question answerable with audio only"
        
        # If question only asks about visual ("what color is X")
        if 'color' in question_lower or 'wearing' in question_lower:
            if 'when' not in question_lower and 'after' not in question_lower:
                return False, "Question answerable with visual only"
        
        return True, None  # Assume both needed if unclear
