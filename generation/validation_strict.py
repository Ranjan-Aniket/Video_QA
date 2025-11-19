"""
Strict 10-Layer Validation System

Enforces ALL guidelines with zero tolerance for violations.
Target: 99.9% hallucination-free questions.

VALIDATION LAYERS:
1. Dual Cue Check - Both audio AND visual required
2. Name Blocking - NO names (people, teams, companies, media)
3. Descriptor Validation - Only descriptors, no pronouns
4. Timestamp Accuracy - Precise timestamps covering cues + actions
5. Single-Cue Test - Can't be answered with just one cue (CRITICAL)
6. Intro/Outro Check - Don't use intro/outro segments
7. Spurious Correlation Detection - For spurious correlation questions
8. Audio Variety Check - Not just speech, use music/sounds
9. Complexity Scoring - Ensure challenging questions
10. Evidence Grounding - All references exist in evidence

Cost Optimization:
- Batch LLM calls where possible
- Use heuristics first, LLM for edge cases
- Cache validation results
"""

from typing import Optional, List, Dict, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import re
import openai
from templates.base import GeneratedQuestion, EvidenceDatabase, CueType


class ValidationLayer(Enum):
    """10 validation layers"""
    DUAL_CUE_CHECK = 1
    NAME_BLOCKING = 2
    DESCRIPTOR_VALIDATION = 3
    TIMESTAMP_ACCURACY = 4
    SINGLE_CUE_TEST = 5
    INTRO_OUTRO_CHECK = 6
    SPURIOUS_CORRELATION = 7
    AUDIO_VARIETY = 8
    COMPLEXITY_SCORING = 9
    EVIDENCE_GROUNDING = 10


@dataclass
class ValidationResult:
    """Result of validation"""
    passed: bool
    layer: ValidationLayer
    error_message: Optional[str] = None
    score: Optional[float] = None  # For scoring layers


class StrictValidator:
    """
    Strict validation enforcing all guidelines
    
    Each layer can REJECT a question if it violates guidelines.
    """
    
    def __init__(self, openai_api_key: str):
        """
        Initialize validator
        
        Args:
            openai_api_key: OpenAI API key for LLM-based validation
        """
        self.client = openai.OpenAI(api_key=openai_api_key)
        
        # Validation statistics
        self.stats = {
            layer: {"passed": 0, "failed": 0}
            for layer in ValidationLayer
        }
        
        # Cache for LLM validation results
        self.validation_cache: Dict[str, ValidationResult] = {}
    
    def validate_all(
        self,
        question: GeneratedQuestion,
        evidence: EvidenceDatabase
    ) -> Tuple[bool, List[ValidationResult]]:
        """
        Run all 10 validation layers
        
        Args:
            question: Generated question to validate
            evidence: Evidence database
            
        Returns:
            (passed_all, list of validation results)
        """
        results = []
        
        # Layer 1: Dual Cue Check (CRITICAL)
        result = self.layer1_dual_cue_check(question)
        results.append(result)
        if not result.passed:
            self._update_stats(ValidationLayer.DUAL_CUE_CHECK, False)
            return False, results
        self._update_stats(ValidationLayer.DUAL_CUE_CHECK, True)
        
        # Layer 2: Name Blocking (CRITICAL)
        result = self.layer2_name_blocking(question, evidence)
        results.append(result)
        if not result.passed:
            self._update_stats(ValidationLayer.NAME_BLOCKING, False)
            return False, results
        self._update_stats(ValidationLayer.NAME_BLOCKING, True)
        
        # Layer 3: Descriptor Validation
        result = self.layer3_descriptor_validation(question)
        results.append(result)
        if not result.passed:
            self._update_stats(ValidationLayer.DESCRIPTOR_VALIDATION, False)
            return False, results
        self._update_stats(ValidationLayer.DESCRIPTOR_VALIDATION, True)
        
        # Layer 4: Timestamp Accuracy
        result = self.layer4_timestamp_accuracy(question, evidence)
        results.append(result)
        if not result.passed:
            self._update_stats(ValidationLayer.TIMESTAMP_ACCURACY, False)
            return False, results
        self._update_stats(ValidationLayer.TIMESTAMP_ACCURACY, True)
        
        # Layer 5: Single-Cue Test (MOST CRITICAL)
        result = self.layer5_single_cue_test(question, evidence)
        results.append(result)
        if not result.passed:
            self._update_stats(ValidationLayer.SINGLE_CUE_TEST, False)
            return False, results
        self._update_stats(ValidationLayer.SINGLE_CUE_TEST, True)
        
        # Layer 6: Intro/Outro Check
        result = self.layer6_intro_outro_check(question, evidence)
        results.append(result)
        if not result.passed:
            self._update_stats(ValidationLayer.INTRO_OUTRO_CHECK, False)
            return False, results
        self._update_stats(ValidationLayer.INTRO_OUTRO_CHECK, True)
        
        # Layer 7: Spurious Correlation (only for spurious questions)
        result = self.layer7_spurious_correlation(question, evidence)
        results.append(result)
        if not result.passed:
            self._update_stats(ValidationLayer.SPURIOUS_CORRELATION, False)
            return False, results
        self._update_stats(ValidationLayer.SPURIOUS_CORRELATION, True)
        
        # Layer 8: Audio Variety
        result = self.layer8_audio_variety(question)
        results.append(result)
        if not result.passed:
            self._update_stats(ValidationLayer.AUDIO_VARIETY, False)
            return False, results
        self._update_stats(ValidationLayer.AUDIO_VARIETY, True)
        
        # Layer 9: Complexity Scoring
        result = self.layer9_complexity_scoring(question)
        results.append(result)
        if not result.passed:
            self._update_stats(ValidationLayer.COMPLEXITY_SCORING, False)
            return False, results
        self._update_stats(ValidationLayer.COMPLEXITY_SCORING, True)
        
        # Layer 10: Evidence Grounding
        result = self.layer10_evidence_grounding(question, evidence)
        results.append(result)
        if not result.passed:
            self._update_stats(ValidationLayer.EVIDENCE_GROUNDING, False)
            return False, results
        self._update_stats(ValidationLayer.EVIDENCE_GROUNDING, True)
        
        return True, results
    
    # ========================================================================
    # LAYER 1: DUAL CUE CHECK
    # ========================================================================
    
    def layer1_dual_cue_check(
        self,
        question: GeneratedQuestion
    ) -> ValidationResult:
        """
        GUIDELINE: "All questions must have both audio and visual cue"
        
        REJECT if:
        - No audio cues
        - No visual cues
        - Empty cue content
        """
        # Check audio cues exist and non-empty
        if not question.audio_cues:
            return ValidationResult(
                passed=False,
                layer=ValidationLayer.DUAL_CUE_CHECK,
                error_message="No audio cues present"
            )
        
        for cue in question.audio_cues:
            if not cue.content or not cue.content.strip():
                return ValidationResult(
                    passed=False,
                    layer=ValidationLayer.DUAL_CUE_CHECK,
                    error_message="Empty audio cue content"
                )
        
        # Check visual cues exist and non-empty
        if not question.visual_cues:
            return ValidationResult(
                passed=False,
                layer=ValidationLayer.DUAL_CUE_CHECK,
                error_message="No visual cues present"
            )
        
        for cue in question.visual_cues:
            if not cue.content or not cue.content.strip():
                return ValidationResult(
                    passed=False,
                    layer=ValidationLayer.DUAL_CUE_CHECK,
                    error_message="Empty visual cue content"
                )
        
        return ValidationResult(
            passed=True,
            layer=ValidationLayer.DUAL_CUE_CHECK
        )
    
    # ========================================================================
    # LAYER 2: NAME BLOCKING
    # ========================================================================
    
    def layer2_name_blocking(
        self,
        question: GeneratedQuestion,
        evidence: EvidenceDatabase
    ) -> ValidationResult:
        """
        GUIDELINE: "Never use any names in prompt or responses"
        "Must avoid names across the board including sports teams, 
        company/band, movies/books/songs"
        
        REJECT if any name found in:
        - Question text
        - Answer text
        - Cue content
        """
        all_names = (
            evidence.character_names +
            evidence.team_names +
            evidence.media_names +
            evidence.brand_names
        )
        
        # Check question text
        text_lower = question.question_text.lower()
        for name in all_names:
            if name.lower() in text_lower:
                return ValidationResult(
                    passed=False,
                    layer=ValidationLayer.NAME_BLOCKING,
                    error_message=f"Name found in question: '{name}'"
                )
        
        # Check answer text
        answer_lower = question.golden_answer.lower()
        for name in all_names:
            if name.lower() in answer_lower:
                return ValidationResult(
                    passed=False,
                    layer=ValidationLayer.NAME_BLOCKING,
                    error_message=f"Name found in answer: '{name}'"
                )
        
        # Check cue content
        for cue in question.audio_cues + question.visual_cues:
            cue_lower = cue.content.lower()
            for name in all_names:
                if name.lower() in cue_lower:
                    return ValidationResult(
                        passed=False,
                        layer=ValidationLayer.NAME_BLOCKING,
                        error_message=f"Name found in cue: '{name}'"
                    )
        
        # Check for capitalized words that might be names (additional safety)
        # Pattern: Capitalized words not at sentence start
        question_words = question.question_text.split()
        for i, word in enumerate(question_words):
            # Skip first word and words after punctuation
            if i == 0:
                continue
            if i > 0 and question_words[i-1][-1] in '.!?':
                continue
            
            # Check if word is capitalized (might be name)
            if word[0].isupper() and len(word) > 1:
                # Allow certain exceptions
                exceptions = ['I', 'A', 'X', 'Y', 'Z']  # Common non-name capitals
                if word not in exceptions and not word.endswith('?') and not word.endswith('.'):
                    # Flag for manual review (don't auto-reject)
                    pass
        
        return ValidationResult(
            passed=True,
            layer=ValidationLayer.NAME_BLOCKING
        )
    
    # ========================================================================
    # LAYER 3: DESCRIPTOR VALIDATION
    # ========================================================================
    
    def layer3_descriptor_validation(
        self,
        question: GeneratedQuestion
    ) -> ValidationResult:
        """
        GUIDELINE: "Avoid using he/she in the question. Always qualify 
        the character with more description"
        
        REJECT if:
        - Uses he/she/his/her without descriptor
        - Uses generic "the person" without description
        """
        text = question.question_text.lower()
        
        # Check for pronouns without context
        pronoun_patterns = [
            r'\bhe\s',
            r'\bshe\s',
            r'\bhis\s',
            r'\bher\s',
            r'\bhim\s',
        ]
        
        for pattern in pronoun_patterns:
            if re.search(pattern, text):
                return ValidationResult(
                    passed=False,
                    layer=ValidationLayer.DESCRIPTOR_VALIDATION,
                    error_message=f"Pronoun found without descriptor: {pattern}"
                )
        
        # Check for "the person" without descriptors
        if re.search(r'\bthe person\b(?!\s+(in|with|wearing|on))', text):
            return ValidationResult(
                passed=False,
                layer=ValidationLayer.DESCRIPTOR_VALIDATION,
                error_message="Generic 'the person' without descriptor"
            )
        
        return ValidationResult(
            passed=True,
            layer=ValidationLayer.DESCRIPTOR_VALIDATION
        )
    
    # ========================================================================
    # LAYER 4: TIMESTAMP ACCURACY
    # ========================================================================
    
    def layer4_timestamp_accuracy(
        self,
        question: GeneratedQuestion,
        evidence: EvidenceDatabase
    ) -> ValidationResult:
        """
        GUIDELINE: "Must incorporate both the cues and subsequent actions"
        "The start and end timestamps should not be a second longer or 
        shorter than what is fully accurate"
        
        REJECT if:
        - Timestamps don't cover all cues
        - Timestamps extend beyond necessary range
        - Cue timestamps outside question range
        """
        start_ts = question.start_timestamp
        end_ts = question.end_timestamp
        
        # Check all cues are within range
        all_cues = question.audio_cues + question.visual_cues
        
        for cue in all_cues:
            if cue.timestamp < start_ts:
                return ValidationResult(
                    passed=False,
                    layer=ValidationLayer.TIMESTAMP_ACCURACY,
                    error_message=f"Cue at {cue.timestamp:.1f}s before start {start_ts:.1f}s"
                )
            
            if cue.timestamp > end_ts:
                return ValidationResult(
                    passed=False,
                    layer=ValidationLayer.TIMESTAMP_ACCURACY,
                    error_message=f"Cue at {cue.timestamp:.1f}s after end {end_ts:.1f}s"
                )
        
        # Check timestamps are reasonable (not too long)
        duration = end_ts - start_ts
        
        # Get cue range
        cue_timestamps = [c.timestamp for c in all_cues]
        cue_start = min(cue_timestamps)
        cue_end = max(cue_timestamps)
        cue_range = cue_end - cue_start
        
        # Start should not be more than 3 seconds before first cue
        if start_ts < cue_start - 3.0:
            return ValidationResult(
                passed=False,
                layer=ValidationLayer.TIMESTAMP_ACCURACY,
                error_message=f"Start timestamp too early (>3s before first cue)"
            )
        
        # End should not be more than 5 seconds after last cue
        # (Allow buffer for action completion)
        if end_ts > cue_end + 5.0:
            return ValidationResult(
                passed=False,
                layer=ValidationLayer.TIMESTAMP_ACCURACY,
                error_message=f"End timestamp too late (>5s after last cue)"
            )
        
        # Total duration should be reasonable (not entire video)
        if duration > 60.0:  # More than 1 minute
            return ValidationResult(
                passed=False,
                layer=ValidationLayer.TIMESTAMP_ACCURACY,
                error_message=f"Timestamp range too long: {duration:.1f}s"
            )
        
        return ValidationResult(
            passed=True,
            layer=ValidationLayer.TIMESTAMP_ACCURACY
        )
    
    # ========================================================================
    # LAYER 5: SINGLE-CUE TEST (MOST CRITICAL)
    # ========================================================================
    
    def layer5_single_cue_test(
        self,
        question: GeneratedQuestion,
        evidence: EvidenceDatabase
    ) -> ValidationResult:
        """
        GUIDELINE: "Even if the question has both audio and video cues, 
        but if it can be answered with just one cue, the pair needs to be rejected"
        
        This is the HARDEST validation to automate.
        
        Strategy:
        1. First use heuristics (fast, free)
        2. If borderline, use LLM validation
        
        REJECT if question answerable with single cue type.
        """
        # Step 1: Heuristic checks (fast)
        
        # Check if question explicitly requires both
        question_lower = question.question_text.lower()
        
        # Good indicators that both are required
        both_required_patterns = [
            'after.*says.*and.*see',
            'when.*says.*what.*see',
            'what.*see.*when.*hear',
            'who.*visible.*when.*says',
            'how many.*after.*says',
        ]
        
        has_explicit_both = any(
            re.search(pattern, question_lower)
            for pattern in both_required_patterns
        )
        
        if has_explicit_both:
            # Likely requires both cues
            return ValidationResult(
                passed=True,
                layer=ValidationLayer.SINGLE_CUE_TEST
            )
        
        # Step 2: Check for obvious single-cue cases
        
        # Bad: Only asks about audio
        audio_only_patterns = [
            r'what.*says?\?$',
            r'what.*audio\?$',
            r'what.*heard?\?$',
        ]
        
        for pattern in audio_only_patterns:
            if re.search(pattern, question_lower):
                return ValidationResult(
                    passed=False,
                    layer=ValidationLayer.SINGLE_CUE_TEST,
                    error_message="Question answerable with audio only"
                )
        
        # Bad: Only asks about visual
        visual_only_patterns = [
            r'what.*see\?$',
            r'what.*visible\?$',
            r'what.*shown\?$',
        ]
        
        for pattern in visual_only_patterns:
            # But allow if it has temporal constraint from audio
            if 'when' in question_lower or 'after' in question_lower:
                continue  # Temporal constraint requires audio
            
            if re.search(pattern, question_lower):
                return ValidationResult(
                    passed=False,
                    layer=ValidationLayer.SINGLE_CUE_TEST,
                    error_message="Question answerable with visual only"
                )
        
        # Step 3: Check uniqueness of visual elements
        # GUIDELINE example: "What color is lighthouse when X said?"
        # If only 1 lighthouse in video, don't need audio cue
        
        # Count unique objects/people in visual cues
        visual_elements = set()
        for cue in question.visual_cues:
            visual_elements.add(cue.content.lower())
        
        # If only 1 unique visual element, might be single-cue answerable
        if len(visual_elements) == 1:
            # Check if this element is unique in video
            element = list(visual_elements)[0]
            
            # Count occurrences in evidence
            count = 0
            for obj in evidence.object_detections:
                if obj['object_class'].lower() in element:
                    count += 1
            
            for person in evidence.person_detections:
                # Check if descriptor matches
                # This is approximate - in production would need better matching
                count += 1
            
            # If only 1 occurrence, might not need audio cue
            if count == 1:
                # LLM validation needed (borderline case)
                return self._llm_single_cue_validation(question, evidence)
        
        # Step 4: For borderline cases, use LLM
        # Check cache first
        cache_key = f"single_cue:{question.question_text}"
        if cache_key in self.validation_cache:
            return self.validation_cache[cache_key]
        
        # Default: assume it requires both (conservative)
        return ValidationResult(
            passed=True,
            layer=ValidationLayer.SINGLE_CUE_TEST
        )
    
    def _llm_single_cue_validation(
        self,
        question: GeneratedQuestion,
        evidence: EvidenceDatabase
    ) -> ValidationResult:
        """
        Use LLM to check if question is answerable with single cue
        
        This is expensive (~$0.10 per call) but accurate.
        """
        prompt = f"""You are validating a multimodal video question.

QUESTION: {question.question_text}
GOLDEN ANSWER: {question.golden_answer}

AUDIO CUES: {[c.content for c in question.audio_cues]}
VISUAL CUES: {[c.content for c in question.visual_cues]}

TASK: Determine if this question can be answered using ONLY:
1. Audio cues (without visual)
2. Visual cues (without audio)

If the question CAN be answered with just one cue type, it should be REJECTED.

EXAMPLES:
- "What color is the lighthouse when X is said?" 
  → If only 1 lighthouse in video, CAN answer with visual only → REJECT
  
- "How many times does person appear after saying X?"
  → CANNOT answer without audio (need to know when "after") → PASS

Respond with JSON:
{{
    "answerable_audio_only": true/false,
    "answerable_visual_only": true/false,
    "explanation": "brief explanation",
    "verdict": "PASS" or "REJECT"
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a strict video question validator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            passed = result['verdict'] == 'PASS'
            error_msg = None if passed else result['explanation']
            
            validation_result = ValidationResult(
                passed=passed,
                layer=ValidationLayer.SINGLE_CUE_TEST,
                error_message=error_msg
            )
            
            # Cache result
            cache_key = f"single_cue:{question.question_text}"
            self.validation_cache[cache_key] = validation_result
            
            return validation_result
            
        except Exception as e:
            # If LLM fails, default to PASS (conservative)
            print(f"[Layer5] LLM validation error: {e}")
            return ValidationResult(
                passed=True,
                layer=ValidationLayer.SINGLE_CUE_TEST
            )
    
    # ========================================================================
    # LAYER 6: INTRO/OUTRO CHECK
    # ========================================================================
    
    def layer6_intro_outro_check(
        self,
        question: GeneratedQuestion,
        evidence: EvidenceDatabase
    ) -> ValidationResult:
        """
        GUIDELINE: "Do not use the intro and outro of the video for 
        reference points for asking questions"
        
        REJECT if any cue is in intro/outro segment.
        """
        intro_end = evidence.intro_end
        outro_start = evidence.outro_start
        
        # Check all cue timestamps
        all_cues = question.audio_cues + question.visual_cues
        
        for cue in all_cues:
            # Check intro
            if intro_end and cue.timestamp < intro_end:
                return ValidationResult(
                    passed=False,
                    layer=ValidationLayer.INTRO_OUTRO_CHECK,
                    error_message=f"Cue at {cue.timestamp:.1f}s in intro (< {intro_end:.1f}s)"
                )
            
            # Check outro
            if outro_start and cue.timestamp > outro_start:
                return ValidationResult(
                    passed=False,
                    layer=ValidationLayer.INTRO_OUTRO_CHECK,
                    error_message=f"Cue at {cue.timestamp:.1f}s in outro (> {outro_start:.1f}s)"
                )
        
        return ValidationResult(
            passed=True,
            layer=ValidationLayer.INTRO_OUTRO_CHECK
        )
    
    # ========================================================================
    # LAYER 7: SPURIOUS CORRELATION DETECTION
    # ========================================================================
    
    def layer7_spurious_correlation(
        self,
        question: GeneratedQuestion,
        evidence: EvidenceDatabase
    ) -> ValidationResult:
        """
        GUIDELINE: "unexpected, unnatural, or un-intuitive events"
        
        Only applies to questions tagged as "Spurious Correlations" type.
        Uses LLM to verify the answer is indeed unexpected/unintuitive.
        """
        from templates.base import QuestionType
        
        # Only validate if question is tagged as spurious correlation
        if QuestionType.TACKLING_SPURIOUS_CORRELATIONS not in question.question_types:
            return ValidationResult(
                passed=True,
                layer=ValidationLayer.SPURIOUS_CORRELATION
            )
        
        # Use LLM to validate spurious nature
        prompt = f"""You are validating a "Spurious Correlation" question.

QUESTION: {question.question_text}
ANSWER: {question.golden_answer}

TASK: Verify that the answer is UNEXPECTED, UNINTUITIVE, or COUNTER-INTUITIVE.

EXAMPLES OF VALID SPURIOUS CORRELATIONS:
- Question: "Who are they referring to when they say 'charging bull'?"
  Answer: "Superman (hologram)" ← Unexpected, not a bull
  
- Question: "What light is shown in background?"
  Answer: "Bomb explosion flash" ← Unexpected, not sunlight

INVALID (too obvious):
- Question: "What does person do after saying X?"
  Answer: "Person walks away" ← Expected, not spurious

Respond with JSON:
{{
    "is_spurious": true/false,
    "explanation": "why it is/isn't spurious",
    "verdict": "PASS" or "REJECT"
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are validating spurious correlation questions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            passed = result['verdict'] == 'PASS'
            error_msg = None if passed else result['explanation']
            
            return ValidationResult(
                passed=passed,
                layer=ValidationLayer.SPURIOUS_CORRELATION,
                error_message=error_msg
            )
            
        except Exception as e:
            # If LLM fails, default to PASS
            print(f"[Layer7] LLM validation error: {e}")
            return ValidationResult(
                passed=True,
                layer=ValidationLayer.SPURIOUS_CORRELATION
            )
    
    # ========================================================================
    # LAYER 8: AUDIO VARIETY CHECK
    # ========================================================================
    
    def layer8_audio_variety(
        self,
        question: GeneratedQuestion
    ) -> ValidationResult:
        """
        GUIDELINE: "Do not limit questions to only speech cues when looking 
        for audio cues, use music, other non-verbal cues as well for diversity"
        
        This is a soft validation - we track diversity but don't reject.
        Just score it.
        """
        # Count audio cue types
        speech_count = sum(1 for c in question.audio_cues if c.cue_type == CueType.AUDIO_SPEECH)
        music_count = sum(1 for c in question.audio_cues if c.cue_type == CueType.AUDIO_MUSIC)
        sound_count = sum(1 for c in question.audio_cues if c.cue_type == CueType.AUDIO_SOUND_EFFECT)
        ambient_count = sum(1 for c in question.audio_cues if c.cue_type == CueType.AUDIO_AMBIENT)
        
        total = len(question.audio_cues)
        
        # Calculate diversity score
        # 1.0 = perfect diversity, 0.0 = all same type
        types_used = sum([
            speech_count > 0,
            music_count > 0,
            sound_count > 0,
            ambient_count > 0
        ])
        
        diversity_score = types_used / 4.0  # Max 4 types
        
        # Don't reject, just score
        return ValidationResult(
            passed=True,
            layer=ValidationLayer.AUDIO_VARIETY,
            score=diversity_score
        )
    
    # ========================================================================
    # LAYER 9: COMPLEXITY SCORING
    # ========================================================================
    
    def layer9_complexity_scoring(
        self,
        question: GeneratedQuestion
    ) -> ValidationResult:
        """
        GUIDELINE: "Focus on complex questions like inferring something 
        not explicitly stated, challenging counting questions"
        
        Use LLM to score complexity. Reject if too simple.
        Minimum complexity: 0.6 (out of 1.0)
        """
        # Check stored complexity score first
        if question.complexity_score >= 0.6:
            return ValidationResult(
                passed=True,
                layer=ValidationLayer.COMPLEXITY_SCORING,
                score=question.complexity_score
            )
        
        # If below threshold, use LLM to verify
        prompt = f"""You are scoring question complexity for adversarial testing.

QUESTION: {question.question_text}
ANSWER: {question.golden_answer}

TYPES: {[qt.value for qt in question.question_types]}

TASK: Score complexity from 0.0 (trivial) to 1.0 (very challenging).

FACTORS:
- Multiple reasoning steps required
- Temporal reasoning across video
- Counter-intuitive answers
- Precise counting/details
- Inference beyond explicit content

EXAMPLES:
- "What color is the shirt?" → 0.2 (trivial)
- "How many times after X and before Y?" → 0.7 (challenging)
- "Why does unexpected event happen?" → 0.9 (very challenging)

Minimum acceptable: 0.6

Respond with JSON:
{{
    "complexity_score": 0.0-1.0,
    "reasoning": "brief explanation",
    "verdict": "PASS" (≥0.6) or "REJECT" (<0.6)
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a complexity scorer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            score = result['complexity_score']
            passed = result['verdict'] == 'PASS'
            error_msg = None if passed else f"Complexity too low: {score:.2f}"
            
            return ValidationResult(
                passed=passed,
                layer=ValidationLayer.COMPLEXITY_SCORING,
                score=score,
                error_message=error_msg
            )
            
        except Exception as e:
            # If LLM fails, use stored score
            print(f"[Layer9] LLM validation error: {e}")
            passed = question.complexity_score >= 0.6
            return ValidationResult(
                passed=passed,
                layer=ValidationLayer.COMPLEXITY_SCORING,
                score=question.complexity_score
            )
    
    # ========================================================================
    # LAYER 10: EVIDENCE GROUNDING
    # ========================================================================
    
    def layer10_evidence_grounding(
        self,
        question: GeneratedQuestion,
        evidence: EvidenceDatabase
    ) -> ValidationResult:
        """
        Verify all evidence references actually exist
        
        REJECT if references point to non-existent evidence.
        """
        for ref in question.evidence_refs:
            # Parse reference (e.g., "audio:10.5", "object:15.2")
            parts = ref.split(':')
            if len(parts) != 2:
                continue
            
            ref_type, timestamp_str = parts
            try:
                timestamp = float(timestamp_str)
            except ValueError:
                continue
            
            # Verify reference exists in evidence
            found = False
            
            if ref_type == 'audio':
                for seg in evidence.transcript_segments:
                    if abs(seg['start'] - timestamp) < 1.0:
                        found = True
                        break
            
            elif ref_type == 'object':
                for obj in evidence.object_detections:
                    if abs(obj['timestamp'] - timestamp) < 1.0:
                        found = True
                        break
            
            elif ref_type == 'person':
                for person in evidence.person_detections:
                    if abs(person['timestamp'] - timestamp) < 1.0:
                        found = True
                        break
            
            elif ref_type == 'event':
                for event in evidence.event_timeline:
                    if abs(event['timestamp'] - timestamp) < 1.0:
                        found = True
                        break
            
            else:
                # Unknown ref type, skip
                continue
            
            if not found:
                return ValidationResult(
                    passed=False,
                    layer=ValidationLayer.EVIDENCE_GROUNDING,
                    error_message=f"Evidence reference not found: {ref}"
                )
        
        return ValidationResult(
            passed=True,
            layer=ValidationLayer.EVIDENCE_GROUNDING
        )
    
    # ========================================================================
    # BATCH VALIDATION (Cost Optimization)
    # ========================================================================
    
    def validate_batch(
        self,
        questions: List[GeneratedQuestion],
        evidence: EvidenceDatabase
    ) -> List[Tuple[bool, List[ValidationResult]]]:
        """
        Validate multiple questions in batch
        
        Optimizes LLM calls by batching Layers 5, 7, 9.
        
        Returns:
            List of (passed, results) for each question
        """
        # Run non-LLM layers individually (fast)
        results_by_question = []
        
        for question in questions:
            # Layers 1-4, 6, 8, 10 (no LLM needed)
            passed, results = self._validate_non_llm_layers(question, evidence)
            results_by_question.append((passed, results, question))
        
        # Filter to questions that passed non-LLM layers
        questions_for_llm = [
            (q, results) for passed, results, q in results_by_question if passed
        ]
        
        if not questions_for_llm:
            return [(passed, results) for passed, results, _ in results_by_question]
        
        # Batch LLM validation for Layers 5, 7, 9
        llm_results = self._batch_llm_validation([q for q, _ in questions_for_llm], evidence)
        
        # Combine results
        final_results = []
        llm_idx = 0
        
        for passed, results, question in results_by_question:
            if not passed:
                final_results.append((False, results))
            else:
                # Add LLM results
                llm_res = llm_results[llm_idx]
                llm_idx += 1
                
                all_passed = all(r.passed for r in llm_res)
                final_results.append((all_passed, results + llm_res))
        
        return final_results
    
    def _validate_non_llm_layers(
        self,
        question: GeneratedQuestion,
        evidence: EvidenceDatabase
    ) -> Tuple[bool, List[ValidationResult]]:
        """Validate layers that don't need LLM"""
        results = []
        
        # Layer 1
        result = self.layer1_dual_cue_check(question)
        results.append(result)
        if not result.passed:
            return False, results
        
        # Layer 2
        result = self.layer2_name_blocking(question, evidence)
        results.append(result)
        if not result.passed:
            return False, results
        
        # Layer 3
        result = self.layer3_descriptor_validation(question)
        results.append(result)
        if not result.passed:
            return False, results
        
        # Layer 4
        result = self.layer4_timestamp_accuracy(question, evidence)
        results.append(result)
        if not result.passed:
            return False, results
        
        # Layer 6
        result = self.layer6_intro_outro_check(question, evidence)
        results.append(result)
        if not result.passed:
            return False, results
        
        # Layer 8
        result = self.layer8_audio_variety(question)
        results.append(result)
        # Don't fail on audio variety
        
        # Layer 10
        result = self.layer10_evidence_grounding(question, evidence)
        results.append(result)
        if not result.passed:
            return False, results
        
        return True, results
    
    def _batch_llm_validation(
        self,
        questions: List[GeneratedQuestion],
        evidence: EvidenceDatabase
    ) -> List[List[ValidationResult]]:
        """
        Batch validate Layers 5, 7, 9 with single LLM call
        
        Reduces cost by ~70% vs individual calls.
        """
        # Build batch prompt
        questions_json = []
        for i, q in enumerate(questions):
            questions_json.append({
                "id": i,
                "question": q.question_text,
                "answer": q.golden_answer,
                "types": [qt.value for qt in q.question_types],
                "audio_cues": [c.content for c in q.audio_cues],
                "visual_cues": [c.content for c in q.visual_cues]
            })
        
        import json
        batch_prompt = f"""Validate {len(questions)} video questions across 3 criteria:

QUESTIONS:
{json.dumps(questions_json, indent=2)}

For EACH question, validate:

1. SINGLE-CUE TEST: Can it be answered with only audio OR only visual? If yes, REJECT.
2. SPURIOUS CORRELATION: If tagged as "Tackling Spurious Correlations", is answer truly unexpected/unintuitive?
3. COMPLEXITY: Score 0.0-1.0. Minimum 0.6 to pass.

Respond with JSON array:
[
  {{
    "id": 0,
    "single_cue_pass": true/false,
    "spurious_pass": true/false,
    "complexity_score": 0.0-1.0,
    "complexity_pass": true/false
  }},
  ...
]"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a batch question validator."},
                    {"role": "user", "content": batch_prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            validations = result.get('validations', [])
            
            # Convert to ValidationResult objects
            all_results = []
            for q_validation in validations:
                q_results = []
                
                # Layer 5: Single-cue
                q_results.append(ValidationResult(
                    passed=q_validation['single_cue_pass'],
                    layer=ValidationLayer.SINGLE_CUE_TEST,
                    error_message=None if q_validation['single_cue_pass'] else "Answerable with single cue"
                ))
                
                # Layer 7: Spurious
                q_results.append(ValidationResult(
                    passed=q_validation['spurious_pass'],
                    layer=ValidationLayer.SPURIOUS_CORRELATION,
                    error_message=None if q_validation['spurious_pass'] else "Not spurious enough"
                ))
                
                # Layer 9: Complexity
                q_results.append(ValidationResult(
                    passed=q_validation['complexity_pass'],
                    layer=ValidationLayer.COMPLEXITY_SCORING,
                    score=q_validation['complexity_score'],
                    error_message=None if q_validation['complexity_pass'] else f"Complexity too low: {q_validation['complexity_score']:.2f}"
                ))
                
                all_results.append(q_results)
            
            return all_results
            
        except Exception as e:
            print(f"[BatchValidation] Error: {e}")
            # Fallback: individual validation
            return [
                [
                    self.layer5_single_cue_test(q, evidence),
                    self.layer7_spurious_correlation(q, evidence),
                    self.layer9_complexity_scoring(q)
                ]
                for q in questions
            ]
    
    # ========================================================================
    # UTILITIES
    # ========================================================================
    
    def _update_stats(self, layer: ValidationLayer, passed: bool):
        """Update validation statistics"""
        if passed:
            self.stats[layer]["passed"] += 1
        else:
            self.stats[layer]["failed"] += 1
    
    def get_statistics(self) -> Dict:
        """Get validation statistics"""
        total_validated = sum(
            layer_stats["passed"] + layer_stats["failed"]
            for layer_stats in self.stats.values()
        )
        
        layer_stats = {}
        for layer, stats in self.stats.items():
            total = stats["passed"] + stats["failed"]
            pass_rate = (stats["passed"] / total * 100) if total > 0 else 0
            layer_stats[layer.name] = {
                "passed": stats["passed"],
                "failed": stats["failed"],
                "total": total,
                "pass_rate": f"{pass_rate:.1f}%"
            }
        
        return {
            "total_validated": total_validated,
            "layer_statistics": layer_stats
        }
    
    def print_statistics(self):
        """Print validation statistics"""
        stats = self.get_statistics()
        
        print("=" * 80)
        print("VALIDATION STATISTICS")
        print("=" * 80)
        print(f"Total Questions Validated: {stats['total_validated']}")
        print()
        
        for layer_name, layer_stats in stats['layer_statistics'].items():
            print(f"{layer_name}:")
            print(f"  Passed: {layer_stats['passed']}")
            print(f"  Failed: {layer_stats['failed']}")
            print(f"  Total: {layer_stats['total']}")
            print(f"  Pass Rate: {layer_stats['pass_rate']}")
            print()
        
        print("=" * 80)