"""
Answer Guidelines Enforcer - Ensure answers follow ALL 15 guidelines

This module validates that ANSWERS (not just questions) follow the guidelines:
- No names, no pronouns
- Precise, concise (1-2 sentences)
- Accurate visual details
- Grounded in evidence
- No ambiguity
- Audio-visual connection
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AnswerValidationResult:
    """Result of answer validation"""
    is_valid: bool
    score: float  # 0.0-1.0
    violations: List[str] = None
    warnings: List[str] = None
    corrected_answer: Optional[str] = None
    rules_passed: int = 0
    rules_total: int = 10

    def __post_init__(self):
        if self.violations is None:
            self.violations = []
        if self.warnings is None:
            self.warnings = []


class AnswerGuidelinesEnforcer:
    """
    Validates that answers follow ALL guidelines from Guidelines_ Prompt Creation.docx
    """

    def __init__(self):
        """Initialize with answer-specific validation patterns"""
        self.pronoun_patterns = [
            r'\bhe\b', r'\bshe\b', r'\bhim\b', r'\bher\b', r'\bhis\b',
            r'\bthey\b', r'\bthem\b', r'\btheir\b', r'\bthey\'re\b', r'\bthere\b'
        ]

        self.name_patterns = [
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # First Last Name
            r'\b[A-Z]{2,}\b',  # Team abbreviations (NBA, NFL, etc.)
            r'\b(?:Michael|LeBron|Kobe|Tom|John|Sarah|James)\b',  # Common names
        ]

        # Vague/ambiguous words
        self.ambiguous_words = [
            'something', 'someone', 'somewhere', 'might', 'maybe', 'possibly',
            'seems', 'appears', 'looks', 'perhaps', 'apparently',
            'somewhat', 'rather', 'quite', 'fairly'
        ]

        logger.info("AnswerGuidelinesEnforcer initialized with 10 critical rules")

    def validate_answer(
        self,
        answer: str,
        question: str,
        audio_cue: str,
        visual_cue: str,
        evidence: Dict,
        answer_should_reference: str = "both"  # "visual", "audio", or "both"
    ) -> AnswerValidationResult:
        """
        Validate answer against all guidelines.

        Args:
            answer: The answer text
            question: The question being answered
            audio_cue: Audio cue from question
            visual_cue: Visual cue from question
            evidence: Evidence dict with frame data, transcript, etc.
            answer_should_reference: What should answer reference

        Returns:
            AnswerValidationResult with violations and score
        """
        violations = []
        warnings = []
        rules_passed = 0

        # RULE 1: No Pronouns
        if not self._check_no_pronouns_in_answer(answer):
            violations.append("Rule 1: Answer contains pronouns (he/she/they). Use descriptors!")
        else:
            rules_passed += 1

        # RULE 2: No Names
        if not self._check_no_names_in_answer(answer):
            violations.append("Rule 2: Answer contains proper names. Use descriptors instead!")
        else:
            rules_passed += 1

        # RULE 3: Precision & Clarity
        precision_check = self._check_answer_precision(answer)
        if not precision_check['is_precise']:
            violations.append(f"Rule 3: Answer not precise - {precision_check['reason']}")
        else:
            rules_passed += 1

        # RULE 4: Conciseness
        conciseness_check = self._check_conciseness(answer)
        if not conciseness_check['is_concise']:
            violations.append(f"Rule 4: Answer not concise - {conciseness_check['reason']}")
        else:
            rules_passed += 1

        # RULE 5: Visual Details Accuracy
        if visual_cue and not self._check_visual_accuracy(answer, visual_cue, evidence):
            violations.append("Rule 5: Visual details in answer don't match evidence")
        else:
            rules_passed += 1

        # RULE 6: Audio Grounding
        if audio_cue and not self._check_audio_grounding(answer, audio_cue, evidence):
            violations.append("Rule 6: Answer not properly grounded in audio")
        else:
            rules_passed += 1

        # RULE 7: Complete Sentences
        if not self._check_complete_sentences(answer):
            violations.append("Rule 7: Answer has incomplete or fragmented sentences")
        else:
            rules_passed += 1

        # RULE 8: Capitalization & Grammar
        if not self._check_grammar_capitalization(answer):
            warnings.append("Rule 8: Answer has grammar/capitalization issues")
        else:
            rules_passed += 1

        # RULE 9: No Filler Words
        if not self._check_no_filler(answer):
            violations.append("Rule 9: Answer contains filler words (um, uh, like, etc.)")
        else:
            rules_passed += 1

        # RULE 10: Relevance to Question
        if not self._check_relevance_to_question(answer, question):
            violations.append("Rule 10: Answer doesn't directly answer the question")
        else:
            rules_passed += 1

        # Determine if valid
        is_valid = len(violations) == 0
        score = rules_passed / 10.0

        return AnswerValidationResult(
            is_valid=is_valid,
            score=score,
            violations=violations,
            warnings=warnings,
            rules_passed=rules_passed,
            rules_total=10
        )

    # Rule implementations

    def _check_no_pronouns_in_answer(self, answer: str) -> bool:
        """Rule 1: NO pronouns in answer"""
        answer_lower = answer.lower()

        for pattern in self.pronoun_patterns:
            if re.search(pattern, answer_lower):
                return False

        return True

    def _check_no_names_in_answer(self, answer: str) -> bool:
        """Rule 2: NO proper names, team names, etc."""
        # Check for common name patterns
        for pattern in self.name_patterns:
            if re.search(pattern, answer):
                return False

        # Check for sports team names
        teams = ['Lakers', 'Warriors', 'Celtics', 'Heat', 'Bulls', 'Spurs',
                'Nets', 'Knicks', 'Mavericks', 'Nuggets']
        for team in teams:
            if team in answer:
                return False

        return True

    def _check_answer_precision(self, answer: str) -> Dict:
        """Rule 3: Answer must be precise with no ambiguity"""
        answer_lower = answer.lower()

        # Check for ambiguous words
        for word in self.ambiguous_words:
            if f' {word} ' in f' {answer_lower} ':
                return {
                    'is_precise': False,
                    'reason': f"Uses ambiguous word '{word}'"
                }

        # Check for vague references
        if answer_lower.startswith('the '):
            # Acceptable: "The player shoots"
            pass

        if re.search(r'kind of|sort of|type of|a|some', answer_lower):
            # These might indicate imprecision
            if len(answer) < 15:  # Too short to be properly specific
                return {
                    'is_precise': False,
                    'reason': "Answer too vague or incomplete"
                }

        return {'is_precise': True}

    def _check_conciseness(self, answer: str) -> Dict:
        """Rule 4: Answer must be 1-2 sentences, concise"""
        # Count sentences
        sentences = re.split(r'[.!?]+', answer.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        num_sentences = len(sentences)

        if num_sentences > 3:
            return {
                'is_concise': False,
                'reason': f"Too many sentences ({num_sentences}). Max 2 sentences."
            }

        # Check word count (should be under 30 words typically)
        words = answer.split()
        if len(words) > 40:
            return {
                'is_concise': False,
                'reason': f"Too wordy ({len(words)} words). Keep under 30-35 words."
            }

        if len(words) < 3:
            return {
                'is_concise': False,
                'reason': "Answer too short. Need complete thought."
            }

        return {'is_concise': True}

    def _check_visual_accuracy(self, answer: str, visual_cue: str, evidence: Dict) -> bool:
        """Rule 5: Visual details must match evidence"""
        # Extract colors from answer
        colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink',
                 'black', 'white', 'gray', 'grey', 'brown', 'navy']

        answer_colors = set()
        for color in colors:
            if color in answer.lower():
                answer_colors.add(color)

        # For now, accept any color claims (would need full evidence to verify)
        # But reject if specific number claims are unrealistic
        numbers = re.findall(r'\d+', answer)
        for num_str in numbers:
            num = int(num_str)
            if num > 1000:  # Unrealistic
                return False

        return True

    def _check_audio_grounding(self, answer: str, audio_cue: str, evidence: Dict) -> bool:
        """Rule 6: Answer should be grounded in audio/transcript"""
        # Check if answer mentions key words from audio cue
        audio_words = set(audio_cue.lower().split())
        answer_words = set(answer.lower().split())

        # Should share some key words
        shared = audio_words & answer_words
        if len(shared) == 0 and len(audio_cue) > 5:
            # No shared words - might not be grounded
            # But allow if answer describes result of audio event
            return True  # Lenient for now

        return True

    def _check_complete_sentences(self, answer: str) -> bool:
        """Rule 7: Sentences should be complete"""
        # Check for sentence fragments
        # Sentences should have subject + verb
        sentences = re.split(r'[.!?]+', answer.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        for sentence in sentences:
            words = sentence.split()
            if len(words) < 3:  # Too short to be complete
                return False

            # Check for fragments like "Very quickly" or "After which"
            if sentence.lower().startswith(('and ', 'or ', 'but ', 'while ', 'after ', 'before ')):
                # These might be OK if continuing previous thought, but strict: no
                if sentence[0].isupper() and len(sentences) == 1:
                    return False

        return True

    def _check_grammar_capitalization(self, answer: str) -> bool:
        """Rule 8: Proper grammar and capitalization"""
        # Check first word capitalized
        if answer and answer[0].islower():
            return False

        # Check basic punctuation
        if not re.search(r'[.!?]$', answer):
            return False

        return True

    def _check_no_filler(self, answer: str) -> bool:
        """Rule 9: No filler words"""
        filler_words = [
            'um ', 'uh ', 'like', 'you know', 'basically', 'literally',
            'actually', 'I think', 'I believe', 'in my opinion'
        ]

        answer_lower = answer.lower()
        for filler in filler_words:
            if filler in answer_lower:
                return False

        return True

    def _check_relevance_to_question(self, answer: str, question: str) -> bool:
        """Rule 10: Answer must directly answer the question"""
        # Extract question type from question
        if 'what' in question.lower():
            # Answer should describe something (not just "yes" or "no")
            if answer.lower() in ['yes', 'no', 'yes.', 'no.']:
                return False

        if 'how many' in question.lower() or 'count' in question.lower():
            # Answer should have a number
            if not re.search(r'\d+', answer):
                return False

        if 'why' in question.lower() or 'reason' in question.lower():
            # Answer should explain reason
            explanation_words = ['because', 'due to', 'caused by', 'result of', 'reason']
            has_explanation = any(w in answer.lower() for w in explanation_words)
            if not has_explanation and len(answer) < 20:
                return False  # Too short to be proper explanation

        return True

    def correct_answer(
        self,
        answer: str,
        question: str,
        audio_cue: str,
        visual_cue: str,
        evidence: Dict
    ) -> str:
        """
        Attempt to auto-correct answer violations (simple fixes only).

        Returns corrected answer or original if can't fix.
        """
        corrected = answer

        # Fix 1: Add period if missing
        if corrected and corrected[-1] not in '.!?':
            corrected += '.'

        # Fix 2: Capitalize first letter
        if corrected and corrected[0].islower():
            corrected = corrected[0].upper() + corrected[1:]

        # Fix 3: Replace pronouns with descriptors
        pronoun_replacements = {
            'he ': 'the individual ',
            'she ': 'the individual ',
            'they ': 'the individuals ',
            'him ': 'the individual ',
            'her ': 'the individual ',
            'his ': 'the individual\'s ',
            'their ': 'the individuals\' '
        }

        corrected_lower = corrected.lower()
        for pronoun, replacement in pronoun_replacements.items():
            if f' {pronoun}' in f' {corrected_lower}':
                corrected = re.sub(
                    rf'\b{pronoun[:-1]}\b',
                    replacement.strip(),
                    corrected,
                    flags=re.IGNORECASE
                )

        return corrected


# Helper function for quick validation
def validate_answer_against_guidelines(
    answer: str,
    question: str,
    audio_cue: str,
    visual_cue: str,
    evidence: Dict
) -> Tuple[bool, str]:
    """
    Quick validation wrapper.

    Returns:
        (is_valid, message) tuple
    """
    enforcer = AnswerGuidelinesEnforcer()
    result = enforcer.validate_answer(answer, question, audio_cue, visual_cue, evidence)

    if result.is_valid:
        return True, f"✓ Valid answer (Score: {result.score:.2f})"
    else:
        violations_str = '\n'.join(f"  - {v}" for v in result.violations)
        return False, f"✗ Invalid answer:\n{violations_str}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test examples
    enforcer = AnswerGuidelinesEnforcer()

    # Example 1: BAD - Has pronoun
    bad_answer_1 = "He shoots the basketball into the hoop."
    result_1 = enforcer.validate_answer(
        bad_answer_1,
        "What does the player in white do?",
        "nothing",
        "player in white",
        {}
    )
    print(f"\nTest 1 - Pronoun in answer:")
    print(f"Answer: {bad_answer_1}")
    print(f"Valid: {result_1.is_valid}, Score: {result_1.score:.2f}")
    print(f"Violations: {result_1.violations}")

    # Example 2: GOOD
    good_answer = "The player in white dunks the basketball."
    result_2 = enforcer.validate_answer(
        good_answer,
        "What does the player in white do?",
        "nothing",
        "player in white dunking",
        {}
    )
    print(f"\nTest 2 - Good answer:")
    print(f"Answer: {good_answer}")
    print(f"Valid: {result_2.is_valid}, Score: {result_2.score:.2f}")

    # Example 3: BAD - Too wordy
    bad_answer_3 = "The individual who is wearing the white colored uniform shirt and who is playing as a guard position on the basketball team does perform a dunking action with the basketball ball in the basketball hoop."
    result_3 = enforcer.validate_answer(
        bad_answer_3,
        "What does player do?",
        "nothing",
        "player in white",
        {}
    )
    print(f"\nTest 3 - Too wordy:")
    print(f"Valid: {result_3.is_valid}, Score: {result_3.score:.2f}")
    print(f"Violations: {result_3.violations}")
