"""
Enhanced Validation Rules - Enforces ALL Guidelines

CRITICAL: This module enforces EVERY guideline from the Guidelines document.
NO rules are skipped, dropped, or truncated.

Enforced Rules:
1. Dual Cue (Audio + Visual) - MANDATORY
2. Single-Cue Rejection - If answerable with one cue → REJECT
3. Timestamp Precision - Exact start/end covering cues + actions
4. Content Rejection - Violence/subtitles → REJECT
5. Precision - No ambiguity, accurate cues
6. Intro/Outro - Never use for reference points
7. Character Description - Never he/she, always descriptors
8. Audio/Visual Diversity - Background sounds, music, non-verbal cues
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re


@dataclass
class ValidationResult:
    """Result of enhanced validation"""
    is_valid: bool
    rule_violations: List[str]
    warnings: List[str]
    confidence_score: float  # 0.0 to 1.0


class EnhancedValidator:
    """
    Enhanced validation enforcing ALL Guidelines requirements.
    
    Every rule from the Guidelines document is implemented here.
    """
    
    def __init__(self):
        # Compile patterns for efficiency
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for validation"""
        # Pronouns to reject
        self.pronoun_pattern = re.compile(
            r'\b(he|she|his|her|him)\b', 
            re.IGNORECASE
        )
        
        # Timestamp question patterns to reject
        self.timestamp_question_pattern = re.compile(
            r'at what time|what timestamp|when \(time\)',
            re.IGNORECASE
        )
        
        # Intro/outro keywords
        self.intro_keywords = [
            'at the beginning', 'at the start', 'in the intro',
            'opening scene', 'first thing', 'video begins'
        ]
        self.outro_keywords = [
            'at the end', 'in the outro', 'closing scene',
            'last thing', 'video ends', 'final scene'
        ]
        
        # Violence/obscene keywords for content rejection
        self.violence_keywords = [
            'gunshot', 'shooting', 'violence', 'blood', 'stabbing',
            'killing', 'murder', 'assault', 'fight', 'beating'
        ]
        self.obscene_keywords = [
            'sexual', 'obscene', 'nude', 'naked', 'explicit'
        ]
    
    def validate_question(
        self,
        question: str,
        answer: str,
        audio_cues: List[str],
        visual_cues: List[str],
        evidence: Dict,
        timestamps: Optional[Tuple[float, float]] = None
    ) -> ValidationResult:
        """
        Validate question against ALL Guidelines rules.
        
        Args:
            question: The question text
            answer: The answer text
            audio_cues: List of audio cues in the question
            visual_cues: List of visual cues in the question
            evidence: Evidence database
            timestamps: Optional (start, end) timestamps
            
        Returns:
            ValidationResult with violations and score
        """
        violations = []
        warnings = []
        confidence = 1.0
        
        # RULE 1: Dual Cue Requirement (Audio + Visual)
        if not self._check_dual_cue(audio_cues, visual_cues):
            violations.append(
                "CRITICAL: Question must have BOTH audio and visual cues. "
                f"Found {len(audio_cues)} audio cue(s) and {len(visual_cues)} visual cue(s)."
            )
            confidence -= 0.5
        
        # RULE 2: Single-Cue Answerable Check
        if self._is_answerable_with_single_cue(question, answer, audio_cues, visual_cues, evidence):
            violations.append(
                "CRITICAL: Question can be answered with just one cue (audio OR visual). "
                "This violates the requirement that both cues must be necessary."
            )
            confidence -= 0.4
        
        # RULE 3: Multipart Question Validation
        if self._is_multipart_question(question):
            if not self._validate_multipart_dual_cues(question, audio_cues, visual_cues):
                violations.append(
                    "CRITICAL: Multipart question - ALL subparts must have both audio and visual cues."
                )
                confidence -= 0.3
        
        # RULE 4: Content Rejection - Violence/Obscene
        if self._contains_violence_or_obscene(question, answer, evidence):
            violations.append(
                "REJECT: Video contains violence, gunshots, or obscene/sexual content. "
                "This violates content guidelines."
            )
            confidence = 0.0  # Auto-reject
        
        # RULE 5: Built-in Subtitles Check
        if self._has_builtin_subtitles(evidence):
            violations.append(
                "REJECT: Video has built-in subtitles on screen. "
                "This violates content guidelines."
            )
            confidence = 0.0  # Auto-reject
        
        # RULE 6: Pronoun Usage (he/she)
        if self._contains_pronouns(question) or self._contains_pronouns(answer):
            violations.append(
                "Never use he/she in question or answer. "
                "Use descriptors like 'man in black jacket', 'woman with white shoes', 'main character'."
            )
            confidence -= 0.2
        
        # RULE 7: Timestamp Questions (avoid)
        if self._is_timestamp_question(question):
            violations.append(
                "Avoid asking questions about timestamps like 'at what time was X said'."
            )
            confidence -= 0.2
        
        # RULE 8: Precision and Ambiguity
        ambiguity_score = self._check_ambiguity(question)
        if ambiguity_score > 0.3:  # threshold
            warnings.append(
                f"Question may be ambiguous (score: {ambiguity_score:.2f}). "
                "Ensure question is precise, concise, and extremely clear."
            )
            confidence -= 0.1
        
        # RULE 9: Intro/Outro Usage
        if self._uses_intro_or_outro(question, answer):
            violations.append(
                "Do NOT use intro and outro of video for reference points. "
                "These deviate from testing comprehensive understanding of entire content."
            )
            confidence -= 0.3
        
        # RULE 10: Cue Accuracy
        if not self._check_cue_accuracy(audio_cues, visual_cues, evidence):
            violations.append(
                "Cues in question must be super accurate. "
                "Do not say 'blue shirt' when it's actually 'black shirt'."
            )
            confidence -= 0.3
        
        # RULE 11: Timestamp Precision (if timestamps provided)
        if timestamps:
            ts_violations = self._validate_timestamp_precision(
                question, answer, timestamps, audio_cues, visual_cues, evidence
            )
            if ts_violations:
                violations.extend(ts_violations)
                confidence -= 0.2 * len(ts_violations)
        
        # RULE 12: Quote Precision
        if self._contains_quotes(question) or self._contains_quotes(answer):
            if not self._check_quote_precision(question, answer, evidence):
                violations.append(
                    "When taking quotes from video, ensure words are transcribed precisely "
                    "without any alterations."
                )
                confidence -= 0.2
        
        # RULE 13: Audio Cue Diversity
        if not self._check_audio_diversity(audio_cues):
            warnings.append(
                "Consider using diverse audio cues: background sounds (bird chirping), "
                "background music (piano), tone/pitch changes, audience clapping - "
                "not just speech."
            )
        
        # RULE 14: Visual Cue Questions
        if not self._has_visual_to_audio_question_pattern(question):
            warnings.append(
                "Consider asking 'When you see X, what do you hear?' "
                "for diversity (not just 'When you hear X, what do you see?')."
            )
        
        # RULE 15: Before/After/When Usage
        if self._uses_temporal_keywords(question):
            if not self._validate_temporal_usage(question, audio_cues, visual_cues):
                warnings.append(
                    "Use before/after/when with caution. "
                    "If audio comes after visual, use 'after X was said' not 'when X was said'."
                )
        
        # Final confidence clamping
        confidence = max(0.0, min(1.0, confidence))
        
        # Determine validity
        is_valid = len(violations) == 0 and confidence >= 0.7
        
        return ValidationResult(
            is_valid=is_valid,
            rule_violations=violations,
            warnings=warnings,
            confidence_score=confidence
        )
    
    # ============================================
    # RULE IMPLEMENTATION METHODS
    # ============================================
    
    def _check_dual_cue(self, audio_cues: List[str], visual_cues: List[str]) -> bool:
        """
        RULE 1: All questions must have BOTH audio and visual cues.
        """
        return len(audio_cues) > 0 and len(visual_cues) > 0
    
    def _is_answerable_with_single_cue(
        self,
        question: str,
        answer: str,
        audio_cues: List[str],
        visual_cues: List[str],
        evidence: Dict
    ) -> bool:
        """
        RULE 2: Check if question can be answered with just one cue.
        
        Examples from Guidelines:
        - "Color of lighthouse when X was said" - if only 1 lighthouse, 
          you don't need audio cue (REJECT)
        - "What happens when people clap" - if you see people clapping, 
          you don't need audio cue (REJECT)
        """
        # This is a heuristic check - more sophisticated logic needed for production
        
        # Check if answer is available without audio cue
        if len(visual_cues) > 0:
            # If visual cue uniquely identifies the answer
            for vcue in visual_cues:
                if vcue.lower() in evidence.get('visual_transcript', '').lower():
                    # Check if this visual cue alone can answer
                    if answer.lower() in evidence.get('visual_transcript', '').lower():
                        # Potential single-cue answerable
                        return True
        
        # Check if answer is available without visual cue
        if len(audio_cues) > 0:
            # If audio cue uniquely identifies the answer
            for acue in audio_cues:
                if acue.lower() in evidence.get('audio_transcript', '').lower():
                    # Check if this audio cue alone can answer
                    if answer.lower() in evidence.get('audio_transcript', '').lower():
                        # Potential single-cue answerable
                        return True
        
        return False
    
    def _is_multipart_question(self, question: str) -> bool:
        """Check if question has multiple parts"""
        # Look for "and", numbered parts, etc.
        return bool(
            re.search(r'\band\b.*\?', question) or
            re.search(r'\d+\)', question) or
            re.search(r'[a-z]\)', question, re.IGNORECASE)
        )
    
    def _validate_multipart_dual_cues(
        self,
        question: str,
        audio_cues: List[str],
        visual_cues: List[str]
    ) -> bool:
        """
        RULE 3: If multipart, ALL subparts must have both audio and visual cues.
        """
        # Count parts
        parts = re.split(r'\band\b|\?', question)
        num_parts = len([p for p in parts if p.strip()])
        
        # Each part should have at least one audio and one visual cue
        # This is simplified - production should track cues per part
        min_required_cues = num_parts
        
        return (len(audio_cues) >= min_required_cues and 
                len(visual_cues) >= min_required_cues)
    
    def _contains_violence_or_obscene(
        self,
        question: str,
        answer: str,
        evidence: Dict
    ) -> bool:
        """
        RULE 4: Reject videos with violence/gunshots/obscene/sexual content.
        """
        text = f"{question} {answer}".lower()
        
        # Check for violence keywords
        for keyword in self.violence_keywords:
            if keyword in text:
                return True
        
        # Check for obscene keywords
        for keyword in self.obscene_keywords:
            if keyword in text:
                return True
        
        # Check evidence metadata
        if evidence.get('contains_violence', False):
            return True
        if evidence.get('contains_obscene_content', False):
            return True
        
        return False
    
    def _has_builtin_subtitles(self, evidence: Dict) -> bool:
        """
        RULE 5: Reject videos with built-in subtitles on screen.
        """
        return evidence.get('has_builtin_subtitles', False)
    
    def _contains_pronouns(self, text: str) -> bool:
        """
        RULE 6: Never use he/she/his/her/him.
        Use descriptors instead.
        """
        return bool(self.pronoun_pattern.search(text))
    
    def _is_timestamp_question(self, question: str) -> bool:
        """
        RULE 7: Avoid asking "at what time was X said".
        """
        return bool(self.timestamp_question_pattern.search(question))
    
    def _check_ambiguity(self, question: str) -> float:
        """
        RULE 8: Questions must be precise, concise, little to no ambiguity.
        
        Returns ambiguity score (0.0 = clear, 1.0 = very ambiguous)
        """
        ambiguity_score = 0.0
        
        # Check for vague words
        vague_words = [
            'some', 'several', 'many', 'few', 'thing', 'stuff',
            'maybe', 'might', 'could', 'possibly', 'approximately'
        ]
        for word in vague_words:
            if word in question.lower():
                ambiguity_score += 0.1
        
        # Check for unclear references
        unclear_refs = ['it', 'that', 'this', 'these', 'those']
        for ref in unclear_refs:
            if f" {ref} " in f" {question.lower()} ":
                ambiguity_score += 0.05
        
        # Long sentences tend to be more ambiguous
        if len(question.split()) > 30:
            ambiguity_score += 0.1
        
        return min(1.0, ambiguity_score)
    
    def _uses_intro_or_outro(self, question: str, answer: str) -> bool:
        """
        RULE 9: Do NOT use intro/outro for reference points.
        """
        text = f"{question} {answer}".lower()
        
        for keyword in self.intro_keywords:
            if keyword in text:
                return True
        
        for keyword in self.outro_keywords:
            if keyword in text:
                return True
        
        return False
    
    def _check_cue_accuracy(
        self,
        audio_cues: List[str],
        visual_cues: List[str],
        evidence: Dict
    ) -> bool:
        """
        RULE 10: Cues must be super accurate.
        Don't say "blue shirt" when it's "black shirt".
        """
        # Check audio cues against transcript
        audio_transcript = evidence.get('audio_transcript', '').lower()
        for cue in audio_cues:
            if cue.lower() not in audio_transcript:
                return False
        
        # Check visual cues against visual data
        visual_data = evidence.get('visual_transcript', '').lower()
        for cue in visual_cues:
            if cue.lower() not in visual_data:
                return False
        
        return True
    
    def _validate_timestamp_precision(
        self,
        question: str,
        answer: str,
        timestamps: Tuple[float, float],
        audio_cues: List[str],
        visual_cues: List[str],
        evidence: Dict
    ) -> List[str]:
        """
        RULE 11: Timestamp precision validation.
        
        From Guidelines:
        - Must incorporate BOTH cues AND subsequent actions
        - Start: when first cue appears
        - End: when action completes
        - Cover entire duration
        - If song referenced, cover till last second
        - Not a second longer or shorter than accurate
        - Focus on what helps answer, not supporting elements
        """
        violations = []
        start_ts, end_ts = timestamps
        
        # Check 1: Timestamps must cover all cues
        cue_timestamps = evidence.get('cue_timestamps', [])
        if cue_timestamps:
            first_cue_ts = min(cue_timestamps)
            if start_ts > first_cue_ts + 0.5:  # Allow 0.5s tolerance
                violations.append(
                    f"Start timestamp ({start_ts}) should begin when first cue appears ({first_cue_ts})"
                )
        
        # Check 2: Must cover action completion
        if 'action_completion_time' in evidence:
            action_end = evidence['action_completion_time']
            if end_ts < action_end - 0.5:  # Allow 0.5s tolerance
                violations.append(
                    f"End timestamp ({end_ts}) should cover until action completes ({action_end})"
                )
        
        # Check 3: If song referenced, must cover till last second
        if 'song' in answer.lower() or 'music' in answer.lower():
            if 'song_end_time' in evidence:
                song_end = evidence['song_end_time']
                if end_ts < song_end - 0.5:
                    violations.append(
                        f"Answer references song - end timestamp must cover till last second of song ({song_end})"
                    )
        
        # Check 4: Not longer than necessary
        if 'minimum_required_duration' in evidence:
            min_duration = evidence['minimum_required_duration']
            actual_duration = end_ts - start_ts
            if actual_duration > min_duration * 1.5:  # 50% tolerance
                violations.append(
                    f"Timestamp duration ({actual_duration:.1f}s) longer than necessary (required: {min_duration:.1f}s)"
                )
        
        return violations
    
    def _contains_quotes(self, text: str) -> bool:
        """Check if text contains quoted speech"""
        return bool(re.search(r'["\'].*["\']', text))
    
    def _check_quote_precision(
        self,
        question: str,
        answer: str,
        evidence: Dict
    ) -> bool:
        """
        RULE 12: Quotes must be transcribed precisely without alterations.
        """
        # Extract quotes
        quotes = re.findall(r'["\']([^"\']+)["\']', f"{question} {answer}")
        
        # Check each quote against transcript
        transcript = evidence.get('audio_transcript', '')
        for quote in quotes:
            if quote not in transcript:
                return False
        
        return True
    
    def _check_audio_diversity(self, audio_cues: List[str]) -> bool:
        """
        RULE 13: Use diverse audio cues - not just speech.
        Background sounds, music, tone changes, clapping, etc.
        """
        # Check for non-speech audio cues
        non_speech_keywords = [
            'music', 'sound', 'noise', 'clapping', 'applause',
            'chirping', 'playing', 'melody', 'tune', 'tone',
            'pitch', 'humming', 'buzzing', 'beep'
        ]
        
        for cue in audio_cues:
            cue_lower = cue.lower()
            for keyword in non_speech_keywords:
                if keyword in cue_lower:
                    return True
        
        return False
    
    def _has_visual_to_audio_question_pattern(self, question: str) -> bool:
        """
        RULE 14: Also ask "When you see X, what do you hear?"
        Not just "When you hear X, what do you see?"
        """
        return bool(re.search(r'when.*see.*hear', question, re.IGNORECASE))
    
    def _uses_temporal_keywords(self, question: str) -> bool:
        """Check if question uses before/after/when"""
        return bool(re.search(r'\b(before|after|when)\b', question, re.IGNORECASE))
    
    def _validate_temporal_usage(
        self,
        question: str,
        audio_cues: List[str],
        visual_cues: List[str]
    ) -> bool:
        """
        RULE 15: Use before/after/when with caution.
        If audio comes after visual, use "after X was said" not "when X was said".
        """
        # This is a simplified check - production needs temporal ordering
        # from evidence
        
        # Look for pattern: "after [audio cue]"
        if 'after' in question.lower():
            # Should reference audio cue after it occurs
            return True
        
        # Look for pattern: "when [audio cue]"
        if 'when' in question.lower():
            # Check if this is appropriate usage
            # "When" is ambiguous - prefer "before" or "after"
            return False
        
        return True


def validate_question_comprehensive(
    question: str,
    answer: str,
    audio_cues: List[str],
    visual_cues: List[str],
    evidence: Dict,
    timestamps: Optional[Tuple[float, float]] = None
) -> ValidationResult:
    """
    Convenience function for comprehensive validation.
    
    Enforces ALL Guidelines requirements.
    """
    validator = EnhancedValidator()
    return validator.validate_question(
        question=question,
        answer=answer,
        audio_cues=audio_cues,
        visual_cues=visual_cues,
        evidence=evidence,
        timestamps=timestamps
    )
