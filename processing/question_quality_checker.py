"""
Question Quality Checker - Pre-filter questions before expensive validation

Implements the 15 quality guidelines from validation report.
Catches issues BEFORE questions are written to output.
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass

# ✅ FIXED: Import centralized validation patterns
from .validation_patterns import HEDGING_PATTERNS, VAGUE_REFERENCE_PATTERNS


@dataclass
class QualityViolation:
    """A single quality guideline violation"""
    guideline_num: int
    guideline_name: str
    severity: str  # CRITICAL, MODERATE, MINOR
    message: str
    suggestion: str


class QuestionQualityChecker:
    """
    Pre-filter questions for quality issues.

    Implements 15 guidelines from validation report:
    1. No hedging language
    2. No pronouns without antecedents
    3. No yes/no questions
    4. Single question only
    5. Specific answers (no generic)
    6. Clear question type
    7. Audio-visual integration
    8. No external knowledge
    9. Timestamp accuracy
    10. Visual cue specificity
    11. Audio cue verbatim
    12. Answer groundability
    13. Question clarity
    14. Appropriate difficulty
    15. No multiple-part questions
    """

    # ✅ FIXED: Guideline 1 & 2 patterns imported from validation_patterns.py
    # HEDGING_PATTERNS - imported at module level
    # VAGUE_REFERENCE_PATTERNS - imported at module level

    # Additional vague pronoun patterns specific to question quality checking
    VAGUE_PRONOUNS = [
        r'\bthe scene\b(?! with | depicts )',     # "the scene" without descriptor
        r'\bthe structure\b(?! with | in )',      # "the structure" without descriptor
        r'\bthis\b(?= is | was | suggests | indicates )',  # "this is", "this suggests"
    ]

    # Guideline 5: Generic/weak answer patterns
    GENERIC_ANSWER_PATTERNS = [
        r'^The scene (depicts|shows|displays)',
        r'^The visual (shows|depicts)',
        r'^The context suggests',
        r'suggests a scenario',
        r'indicating a',
        r'typical (in|of)',
    ]

    # Guideline 15: Multiple-part questions
    MULTIPLE_PART_PATTERNS = [
        r'\bor\b.*\?',  # "does X support or contrast Y?"
        r'compared to',  # "what is X compared to Y?"
        r'differences.*before and after',  # "what differences before/after"
    ]

    def check_question(self, question_data: Dict) -> Tuple[bool, List[QualityViolation]]:
        """
        Check a single question for quality issues.

        Args:
            question_data: Dict with keys: question, golden_answer, question_type,
                          audio_cue, visual_cue, confidence

        Returns:
            (is_valid, violations) - True if passes all checks, list of violations found
        """
        violations = []

        question = question_data.get('question', '')
        answer = question_data.get('golden_answer', '')
        audio_cue = question_data.get('audio_cue', '')
        visual_cue = question_data.get('visual_cue', '')

        # Guideline 1: No hedging language
        violations.extend(self._check_hedging(question, answer, visual_cue))

        # Guideline 2: No vague pronouns
        violations.extend(self._check_pronouns(question, answer))

        # Guideline 3: No yes/no questions
        violations.extend(self._check_yesno(question))

        # Guideline 5: No generic answers
        violations.extend(self._check_generic_answer(answer))

        # Guideline 11: Audio cue should be verbatim
        violations.extend(self._check_audio_cue(audio_cue))

        # Guideline 15: No multiple-part questions
        violations.extend(self._check_multiple_parts(question))

        # Question is valid if no CRITICAL violations
        critical_violations = [v for v in violations if v.severity == "CRITICAL"]
        is_valid = len(critical_violations) == 0

        return is_valid, violations

    def _check_hedging(self, question: str, answer: str, visual_cue: str) -> List[QualityViolation]:
        """Guideline 1: Check for hedging language"""
        violations = []

        for pattern in HEDGING_PATTERNS:
            # Check question
            if re.search(pattern, question, re.IGNORECASE):
                match = re.search(pattern, question, re.IGNORECASE)
                violations.append(QualityViolation(
                    guideline_num=1,
                    guideline_name="No hedging language",
                    severity="CRITICAL" if match.group().lower() in ['might', 'seems', 'appears'] else "MODERATE",
                    message=f"Hedging word '{match.group()}' found in question",
                    suggestion=f"Remove '{match.group()}' and use definitive language"
                ))

            # Check answer
            if re.search(pattern, answer, re.IGNORECASE):
                match = re.search(pattern, answer, re.IGNORECASE)
                violations.append(QualityViolation(
                    guideline_num=1,
                    guideline_name="No hedging language",
                    severity="CRITICAL",
                    message=f"Hedging word '{match.group()}' found in answer",
                    suggestion=f"Replace '{match.group()}' with concrete description"
                ))

            # Check visual_cue
            if re.search(pattern, visual_cue, re.IGNORECASE):
                match = re.search(pattern, visual_cue, re.IGNORECASE)
                violations.append(QualityViolation(
                    guideline_num=1,
                    guideline_name="No hedging language",
                    severity="MODERATE",
                    message=f"Hedging word '{match.group()}' found in visual_cue",
                    suggestion=f"Use definitive description: 'X is Y', not 'X {match.group()} Y'"
                ))

        return violations

    def _check_pronouns(self, question: str, answer: str) -> List[QualityViolation]:
        """Guideline 2: Check for vague pronouns"""
        violations = []

        # ✅ FIXED: Combine centralized patterns with class-specific patterns
        all_vague_patterns = VAGUE_REFERENCE_PATTERNS + self.VAGUE_PRONOUNS

        for pattern in all_vague_patterns:
            # Check question
            if re.search(pattern, question, re.IGNORECASE):
                match = re.search(pattern, question, re.IGNORECASE)
                violations.append(QualityViolation(
                    guideline_num=2,
                    guideline_name="No pronouns without antecedents",
                    severity="MODERATE",
                    message=f"Vague reference '{match.group()}' in question",
                    suggestion="Use specific descriptor: 'the yellow LEGO figure', not 'the figure'"
                ))

            # Check answer
            if re.search(pattern, answer, re.IGNORECASE):
                match = re.search(pattern, answer, re.IGNORECASE)
                violations.append(QualityViolation(
                    guideline_num=2,
                    guideline_name="No pronouns without antecedents",
                    severity="MODERATE",
                    message=f"Vague reference '{match.group()}' in answer",
                    suggestion="Replace with specific noun"
                ))

        return violations

    def _check_yesno(self, question: str) -> List[QualityViolation]:
        """Guideline 3: Check for yes/no questions"""
        violations = []

        # ✅ FIXED: Comprehensive yes/no question starters
        # Covers all common auxiliary verbs and to-be forms that create yes/no questions
        yesno_starts = [
            # Present tense to-be
            r'^Is\b', r'^Are\b', r"^Isn't\b", r"^Aren't\b",

            # Present tense auxiliary
            r'^Does\b', r'^Do\b', r"^Doesn't\b", r"^Don't\b",

            # Past tense to-be
            r'^Was\b', r'^Were\b', r"^Wasn't\b", r"^Weren't\b",

            # Past tense auxiliary
            r'^Did\b', r"^Didn't\b",

            # Perfect tense
            r'^Has\b', r'^Have\b', r'^Had\b',
            r"^Hasn't\b", r"^Haven't\b", r"^Hadn't\b",

            # Future tense
            r'^Will\b', r"^Won't\b",

            # Modal verbs
            r'^Can\b', r"^Can't\b", r'^Could\b', r"^Couldn't\b",
            r'^Would\b', r"^Wouldn't\b", r'^Should\b', r"^Shouldn't\b",
            r'^May\b', r'^Might\b', r'^Must\b',
        ]

        for pattern in yesno_starts:
            if re.search(pattern, question, re.IGNORECASE):
                violations.append(QualityViolation(
                    guideline_num=3,
                    guideline_name="No yes/no questions",
                    severity="CRITICAL",
                    message="Question appears to be yes/no format",
                    suggestion="Rephrase to 'What/How/Which' format"
                ))
                break

        return violations

    def _check_generic_answer(self, answer: str) -> List[QualityViolation]:
        """Guideline 5: Check for generic/vague answers"""
        violations = []

        for pattern in self.GENERIC_ANSWER_PATTERNS:
            if re.search(pattern, answer, re.IGNORECASE):
                match = re.search(pattern, answer, re.IGNORECASE)
                violations.append(QualityViolation(
                    guideline_num=5,
                    guideline_name="Specific answers (no generic)",
                    severity="MODERATE",
                    message=f"Generic pattern '{match.group()}' found in answer",
                    suggestion="Use concrete visual descriptions instead of interpretations"
                ))

        return violations

    def _check_audio_cue(self, audio_cue: str) -> List[QualityViolation]:
        """Guideline 11: Check audio cue is verbatim"""
        violations = []

        # Check for added suffixes like "is heard"
        if audio_cue and re.search(r'is heard\.?$', audio_cue, re.IGNORECASE):
            violations.append(QualityViolation(
                guideline_num=11,
                guideline_name="Audio cue verbatim",
                severity="MINOR",
                message="Audio cue has 'is heard' suffix",
                suggestion="Use exact transcript quote without 'is heard'"
            ))

        # Check for "The phrase" or "The audio"
        if re.search(r'^(The phrase|The audio)', audio_cue, re.IGNORECASE):
            violations.append(QualityViolation(
                guideline_num=11,
                guideline_name="Audio cue verbatim",
                severity="MINOR",
                message="Audio cue starts with 'The phrase/audio'",
                suggestion="Use direct quote: 'hello world' instead of 'The phrase hello world'"
            ))

        return violations

    def _check_multiple_parts(self, question: str) -> List[QualityViolation]:
        """Guideline 15: Check for multiple-part questions"""
        violations = []

        for pattern in self.MULTIPLE_PART_PATTERNS:
            if re.search(pattern, question, re.IGNORECASE):
                violations.append(QualityViolation(
                    guideline_num=15,
                    guideline_name="No multiple-part questions",
                    severity="CRITICAL",
                    message="Question asks multiple things (A or B, X vs Y)",
                    suggestion="Split into single focused question"
                ))
                break

        return violations

    def filter_questions(self, questions: List[Dict]) -> Tuple[List[Dict], List[Dict], Dict]:
        """
        Filter a list of questions, separating valid from invalid.

        Args:
            questions: List of question dicts

        Returns:
            (valid_questions, rejected_questions, stats)
        """
        valid = []
        rejected = []
        stats = {
            'total': len(questions),
            'valid': 0,
            'rejected': 0,
            'violations_by_guideline': {},
            'violations_by_severity': {'CRITICAL': 0, 'MODERATE': 0, 'MINOR': 0}
        }

        for q in questions:
            is_valid, violations = self.check_question(q)

            if is_valid:
                valid.append(q)
                stats['valid'] += 1
            else:
                rejected.append({
                    'question': q,
                    'violations': violations
                })
                stats['rejected'] += 1

            # Collect statistics
            for v in violations:
                guideline = f"G{v.guideline_num}: {v.guideline_name}"
                stats['violations_by_guideline'][guideline] = \
                    stats['violations_by_guideline'].get(guideline, 0) + 1
                stats['violations_by_severity'][v.severity] += 1

        return valid, rejected, stats


def print_quality_report(stats: Dict, rejected: List[Dict]):
    """Print a quality check report"""
    print("\n" + "="*80)
    print("QUESTION QUALITY CHECK REPORT")
    print("="*80)

    print(f"\nTotal questions: {stats['total']}")
    print(f"✅ Valid: {stats['valid']} ({stats['valid']/stats['total']*100:.1f}%)")
    print(f"❌ Rejected: {stats['rejected']} ({stats['rejected']/stats['total']*100:.1f}%)")

    print(f"\nViolations by severity:")
    for severity, count in sorted(stats['violations_by_severity'].items()):
        if count > 0:
            emoji = "🔴" if severity == "CRITICAL" else "🟡" if severity == "MODERATE" else "🟢"
            print(f"  {emoji} {severity}: {count}")

    print(f"\nViolations by guideline:")
    for guideline, count in sorted(stats['violations_by_guideline'].items(),
                                   key=lambda x: x[1], reverse=True):
        print(f"  • {guideline}: {count}")

    if rejected:
        print(f"\n{'='*80}")
        print(f"REJECTED QUESTIONS ({len(rejected)} total)")
        print(f"{'='*80}\n")

        for i, item in enumerate(rejected[:5], 1):  # Show first 5
            q = item['question']
            print(f"\n{i}. Question ID: {q.get('question_id', 'N/A')}")
            print(f"   Question: {q['question'][:80]}...")
            print(f"   Violations:")
            for v in item['violations']:
                emoji = "🔴" if v.severity == "CRITICAL" else "🟡" if v.severity == "MODERATE" else "🟢"
                print(f"     {emoji} G{v.guideline_num}: {v.message}")
                print(f"        → {v.suggestion}")

        if len(rejected) > 5:
            print(f"\n   ... and {len(rejected) - 5} more rejected questions")

    print("\n" + "="*80 + "\n")


# Example usage
if __name__ == "__main__":
    checker = QuestionQualityChecker()

    # Test with example questions from validation report
    test_questions = [
        {
            "question_id": "single_013_q03",
            "question": "When the phrase 'was a Lego being built by itself.' is heard, how does the model's appearance support or contrast this statement?",
            "golden_answer": "The model's appearance supports this statement, as it appears modular with robotic features that align with the concept of self-assembly.",
            "question_type": "Audio-Visual Stitching",
            "audio_cue": "'was a Lego being built by itself.' is heard",
            "visual_cue": "The model appears modular",
            "confidence": 0.92
        },
        {
            "question_id": "single_033_q02",
            "question": "What objects are present in the scene when the audio says 'pulling the shark up himself'?",
            "golden_answer": "The objects visible are a LEGO figure with a red mohawk and a gray shark figure on a blue LEGO baseplate.",
            "question_type": "Audio-Visual Stitching",
            "audio_cue": "'pulling the shark up himself' phrase is heard.",
            "visual_cue": "A LEGO figure with a red mohawk is next to a gray shark on a blue LEGO baseplate.",
            "confidence": 0.96
        }
    ]

    valid, rejected, stats = checker.filter_questions(test_questions)
    print_quality_report(stats, rejected)
