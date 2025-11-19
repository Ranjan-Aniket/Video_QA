"""
Validation Layers 6-10

Layer 6: Intro/Outro Check
Layer 7: Complexity Check
Layer 8: Descriptor Validation
Layer 9: Cue Necessity
Layer 10: Final QC
"""

from typing import Tuple, Optional
from templates.base import GeneratedQuestion, EvidenceDatabase
import re


class IntroOutroValidator:
    """Layer 6: Ensure not in intro/outro segments"""
    
    def validate(
        self,
        question: GeneratedQuestion,
        evidence: EvidenceDatabase
    ) -> Tuple[bool, Optional[str]]:
        """Validate timestamps not in intro/outro"""
        
        # Check if in intro
        if evidence.intro_end:
            if question.start_timestamp < evidence.intro_end:
                return False, f"Question starts in intro segment (< {evidence.intro_end}s)"
        
        # Check if in outro
        if evidence.outro_start:
            if question.end_timestamp > evidence.outro_start:
                return False, f"Question ends in outro segment (> {evidence.outro_start}s)"
        
        return True, None


class ComplexityValidator:
    """Layer 7: Ensure question is sufficiently challenging"""
    
    def validate(
        self,
        question: GeneratedQuestion,
        evidence: EvidenceDatabase
    ) -> Tuple[bool, Optional[str]]:
        """Validate question complexity"""
        
        # Check complexity score
        if question.complexity_score < 0.5:
            return False, f"Complexity score too low: {question.complexity_score}"
        
        # Check question length (very short questions are usually too simple)
        if len(question.question_text.split()) < 8:
            return False, "Question too short (< 8 words)"
        
        # Check answer length (very short answers suspicious)
        if len(question.golden_answer.split()) < 5:
            return False, "Answer too short (< 5 words)"
        
        # Check for yes/no questions (too simple)
        question_lower = question.question_text.lower()
        if question_lower.startswith('is ') or question_lower.startswith('does '):
            answer_lower = question.golden_answer.lower()
            if answer_lower.startswith('yes') or answer_lower.startswith('no'):
                return False, "Yes/no questions too simple"
        
        return True, None


class DescriptorValidator:
    """Layer 8: Validate descriptors are used correctly"""
    
    def validate(
        self,
        question: GeneratedQuestion,
        evidence: EvidenceDatabase
    ) -> Tuple[bool, Optional[str]]:
        """Validate descriptors match evidence"""
        
        # Check for pronouns (should use descriptors instead)
        text = question.question_text + " " + question.golden_answer
        pronouns = ['he', 'she', 'they', 'him', 'her', 'them', 'his', 'hers', 'their']
        
        words = text.lower().split()
        for pronoun in pronouns:
            if pronoun in words:
                # Check if it's in context where descriptor should be used
                return False, f"Pronoun '{pronoun}' used instead of descriptor"
        
        # Check for vague descriptors
        vague = ['someone', 'something', 'person', 'object', 'thing']
        for vague_word in vague:
            if vague_word in text.lower():
                # Count occurrences
                count = text.lower().count(vague_word)
                if count > 2:  # Too many vague references
                    return False, f"Too many vague references: '{vague_word}'"
        
        return True, None


class CueNecessityValidator:
    """Layer 9: Validate both cues are truly necessary"""
    
    def validate(
        self,
        question: GeneratedQuestion,
        evidence: EvidenceDatabase
    ) -> Tuple[bool, Optional[str]]:
        """Validate both cues are necessary to answer"""
        
        # Similar to Layer 5 but stricter
        question_text = question.question_text.lower()
        
        # Count temporal/conditional markers
        markers = ['when', 'after', 'before', 'during', 'while']
        has_temporal = any(marker in question_text for marker in markers)
        
        # Count audio markers
        audio_markers = ['says', 'said', 'speaks', 'shouts', 'whispers', 'hears']
        has_audio_marker = any(marker in question_text for marker in audio_markers)
        
        # Count visual markers
        visual_markers = ['see', 'sees', 'visible', 'appears', 'shown', 'displays']
        has_visual_marker = any(marker in question_text for marker in visual_markers)
        
        # Should have temporal marker + both cue types
        if not has_temporal:
            if not (has_audio_marker and has_visual_marker):
                return False, "Question doesn't clearly require both cues"
        
        return True, None


class FinalQCValidator:
    """Layer 10: Final quality control checks"""
    
    def validate(
        self,
        question: GeneratedQuestion,
        evidence: EvidenceDatabase
    ) -> Tuple[bool, Optional[str]]:
        """Final QC checks"""
        
        # Check grammar basics
        if not question.question_text.endswith('?'):
            return False, "Question must end with ?"
        
        if not question.golden_answer.endswith('.'):
            return False, "Answer must end with ."
        
        # Check first word capitalized
        if not question.question_text[0].isupper():
            return False, "Question must start with capital letter"
        
        if not question.golden_answer[0].isupper():
            return False, "Answer must start with capital letter"
        
        # Check for double spaces
        if '  ' in question.question_text or '  ' in question.golden_answer:
            return False, "Contains double spaces"
        
        # Check quotes are balanced
        quote_count = question.question_text.count('"')
        if quote_count % 2 != 0:
            return False, "Unbalanced quotes in question"
        
        # Check no excessive punctuation
        if '??' in question.question_text or '!!' in question.question_text:
            return False, "Excessive punctuation"
        
        # Check answer doesn't just repeat question
        question_words = set(question.question_text.lower().split())
        answer_words = set(question.golden_answer.lower().split())
        overlap = len(question_words & answer_words)
        
        if overlap > len(answer_words) * 0.7:
            return False, "Answer too similar to question"
        
        return True, None
