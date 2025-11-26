"""
Template Validator - Ensures Template Questions Follow All 15 Guidelines

This module provides validation utilities specifically for template-generated questions
to ensure they comply with all 15 guidelines before being returned.

CRITICAL FIXES:
1. No "Who" questions → Use "Which person/people"
2. No truncated quotes → Use full quotes or rephrase
3. No pronoun-inducing patterns
4. Validate answers for pronouns/names
5. Ensure conciseness in answers
"""

import re
from typing import Dict, List, Optional, Tuple


class TemplateQuestionValidator:
    """Validates and fixes template-generated questions to follow guidelines"""

    # Pronouns to check for (from guideline 6)
    PRONOUNS = [
        'he', 'she', 'they', 'him', 'her', 'them',
        'his', 'hers', 'their', 'theirs',
        'himself', 'herself', 'themselves'
    ]

    # Question patterns that violate guidelines
    PROBLEMATIC_PATTERNS = [
        r'\bwho\s+is\b',  # "who is" → use "which person is"
        r'\bwho\s+are\b',  # "who are" → use "which people are"
        r'\bwho\b',  # General "who" → encourage specific descriptors
    ]

    # Filler words to avoid
    FILLER_WORDS = [
        'um', 'uh', 'like', 'basically', 'literally',
        'actually', 'honestly', 'obviously', 'clearly',
        'i think', 'i believe', 'you know'
    ]

    def __init__(self):
        pass

    def validate_question_text(self, question: str) -> Tuple[bool, List[str]]:
        """
        Validate question text for guideline compliance

        Args:
            question: Question text

        Returns:
            (is_valid, violations)
        """
        violations = []

        # Check for "who" questions
        if re.search(r'\bwho\s+', question.lower()):
            violations.append("Question uses 'who' - replace with 'which person/people'")

        # Check for pronouns in question
        words = question.lower().split()
        for pronoun in self.PRONOUNS:
            if pronoun in words:
                violations.append(f"Question contains pronoun '{pronoun}' - use descriptors")

        # Check for truncated quotes (...)
        if '..."' in question or '"...' in question:
            violations.append("Question has truncated quote - use full quote or rephrase")

        # Check for vague terms
        vague_terms = ['someone', 'something', 'somewhere']
        for term in vague_terms:
            if re.search(rf'\b{term}\b', question.lower()):
                violations.append(f"Question uses vague term '{term}' - be more specific")

        return len(violations) == 0, violations

    def validate_answer_text(self, answer: str) -> Tuple[bool, List[str]]:
        """
        Validate answer text for guideline compliance

        Args:
            answer: Answer text

        Returns:
            (is_valid, violations)
        """
        violations = []

        # Check for pronouns in answer (CRITICAL)
        words = answer.lower().split()
        for pronoun in self.PRONOUNS:
            if pronoun in words:
                violations.append(f"Answer contains pronoun '{pronoun}' - use descriptors")

        # Check for filler words
        answer_lower = answer.lower()
        for filler in self.FILLER_WORDS:
            if filler in answer_lower:
                violations.append(f"Answer contains filler word '{filler}' - remove it")

        # Check conciseness (1-2 sentences, max 40 words)
        word_count = len(words)
        if word_count > 40:
            violations.append(f"Answer too long ({word_count} words) - keep under 40 words")

        # Check sentence count (1-2 sentences)
        sentence_count = answer.count('.') + answer.count('!') + answer.count('?')
        if sentence_count > 2:
            violations.append(f"Answer has {sentence_count} sentences - keep to 1-2")

        # Check complete sentence (capitalized, ends with punctuation)
        if not answer[0].isupper():
            violations.append("Answer must start with capital letter")

        if not answer.endswith(('.', '!', '?')):
            violations.append("Answer must end with punctuation")

        # Check for sentence fragments
        fragment_patterns = [
            r'^(after|before|during|when)\s',  # Starting with temporal words
            r'^(very|quite|rather)\s',  # Starting with adverbs
        ]
        for pattern in fragment_patterns:
            if re.match(pattern, answer.lower()):
                violations.append(f"Answer appears to be a fragment - use complete sentence")

        return len(violations) == 0, violations

    def fix_who_question(self, question: str) -> str:
        """
        Fix questions that use 'who'

        Args:
            question: Original question

        Returns:
            Fixed question
        """
        # Replace "who is" with "which person is"
        question = re.sub(
            r'\bwho\s+is\b',
            'which person is',
            question,
            flags=re.IGNORECASE
        )

        # Replace "who are" with "which people are"
        question = re.sub(
            r'\bwho\s+are\b',
            'which people are',
            question,
            flags=re.IGNORECASE
        )

        # Replace general "who" with "which person"
        question = re.sub(
            r'\bwho\b',
            'which person',
            question,
            flags=re.IGNORECASE
        )

        return question

    def fix_truncated_quote(self, question: str, full_quote: str) -> str:
        """
        Fix truncated quotes in questions

        Args:
            question: Question with potentially truncated quote
            full_quote: Full quote text

        Returns:
            Fixed question with full quote or rephrased
        """
        # If quote is too long (>80 chars), rephrase the question
        if len(full_quote) > 80:
            # Replace direct quote with paraphrase
            question = re.sub(
                r'someone says ".*?"',
                f'the audio mentions "{full_quote[:40]}..."',
                question
            )
        else:
            # Use full quote
            question = re.sub(
                r'".*?"',
                f'"{full_quote}"',
                question
            )

        return question

    def fix_answer_pronouns(self, answer: str, descriptor: str) -> str:
        """
        Fix pronouns in answers by replacing with descriptor

        Args:
            answer: Answer with potential pronouns
            descriptor: Proper descriptor (e.g., "the player in white")

        Returns:
            Fixed answer
        """
        # Replace common pronoun patterns
        replacements = {
            r'\bhe\b': descriptor,
            r'\bshe\b': descriptor,
            r'\bthey\b': descriptor,
            r'\bhim\b': descriptor,
            r'\bher\b': descriptor,
            r'\bthem\b': descriptor,
            r'\bhis\b': f"{descriptor}'s",
            r'\bhers\b': f"{descriptor}'s",
            r'\btheir\b': f"{descriptor}'s",
        }

        for pattern, replacement in replacements.items():
            answer = re.sub(pattern, replacement, answer, flags=re.IGNORECASE)

        return answer

    def validate_audio_quote_accuracy(
        self,
        quote_in_question: str,
        actual_transcript: str
    ) -> bool:
        """
        Validate that audio quote matches transcript exactly

        Per guideline 12: Must transcribe EXACTLY

        Args:
            quote_in_question: Quote used in question
            actual_transcript: Actual transcript text

        Returns:
            True if exact match, False otherwise
        """
        # Extract quote from question (between quotes)
        quote_match = re.search(r'"([^"]+)"', quote_in_question)
        if not quote_match:
            return False

        quote = quote_match.group(1).strip()

        # Check if quote appears exactly in transcript
        return quote.lower() in actual_transcript.lower()

    def make_answer_concise(self, answer: str, max_words: int = 35) -> str:
        """
        Make answer more concise

        Args:
            answer: Original answer
            max_words: Maximum word count

        Returns:
            Concise answer
        """
        words = answer.split()

        if len(words) <= max_words:
            return answer

        # Remove filler words first
        filtered_words = [
            w for w in words
            if w.lower() not in self.FILLER_WORDS
        ]

        # If still too long, truncate and add period
        if len(filtered_words) > max_words:
            filtered_words = filtered_words[:max_words]
            result = ' '.join(filtered_words)
            if not result.endswith('.'):
                result += '.'
            return result

        return ' '.join(filtered_words)


def validate_generated_question(
    question_text: str,
    answer_text: str,
    audio_quote: Optional[str] = None,
    transcript: Optional[str] = None
) -> Tuple[bool, Dict[str, List[str]]]:
    """
    Validate a generated question/answer pair

    Args:
        question_text: Question text
        answer_text: Answer text
        audio_quote: Audio quote used (optional)
        transcript: Full transcript (optional)

    Returns:
        (is_valid, violations_dict)
    """
    validator = TemplateQuestionValidator()

    violations = {
        'question': [],
        'answer': []
    }

    # Validate question
    q_valid, q_violations = validator.validate_question_text(question_text)
    violations['question'] = q_violations

    # Validate answer
    a_valid, a_violations = validator.validate_answer_text(answer_text)
    violations['answer'] = a_violations

    # Validate audio quote accuracy if provided
    if audio_quote and transcript:
        if not validator.validate_audio_quote_accuracy(audio_quote, transcript):
            violations['question'].append("Audio quote doesn't match transcript exactly")

    is_valid = len(violations['question']) == 0 and len(violations['answer']) == 0

    return is_valid, violations


# Helper function for templates to use
def check_and_fix_question(
    question: str,
    answer: str,
    audio_segment_text: str,
    person_descriptor: Optional[str] = None
) -> Tuple[str, str, bool]:
    """
    Check and auto-fix common template issues

    Args:
        question: Question text
        answer: Answer text
        audio_segment_text: Full audio transcript
        person_descriptor: Descriptor for person (if applicable)

    Returns:
        (fixed_question, fixed_answer, is_valid)
    """
    validator = TemplateQuestionValidator()

    # Fix "who" questions
    if 'who' in question.lower():
        question = validator.fix_who_question(question)

    # Fix pronouns in answer
    if person_descriptor:
        answer = validator.fix_answer_pronouns(answer, person_descriptor)

    # Make answer concise
    answer = validator.make_answer_concise(answer)

    # Final validation
    is_valid, violations = validate_generated_question(
        question, answer, question, audio_segment_text
    )

    return question, answer, is_valid
