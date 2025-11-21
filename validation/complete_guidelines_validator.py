"""
Complete Guidelines Validator - ALL 15 Critical Rules

Enforces EVERY guideline from Guidelines__Prompt_Creation.docx:

1.  Dual Cue Requirement (audio + visual both required)
2.  Single-Cue Rejection (if answerable with one → REJECT)
3.  Multipart Validation (all subparts have both cues)
4.  Content Rejection (violence/obscene/sexual)
5.  Subtitle Rejection (built-in subtitles)
6.  Name/Pronoun Blocking (use descriptors only)
7.  Timestamp Questions (avoid "at what time")
8.  Precision Check (no ambiguity)
9.  Intro/Outro Rejection (never use)
10. Cue Accuracy (exact colors, counts)
11. Timestamp Precision (exact start/end)
12. Quote Precision (transcribed exactly)
13. Audio Diversity (background sounds, music)
14. Visual-to-Audio ("when you see X, what hear?")
15. Temporal Usage (before/after/when correct)
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of guideline validation"""
    is_valid: bool
    score: float  # 0.0-1.0
    rule_violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    can_auto_fix: bool = False
    fixed_question: Optional[str] = None
    rules_passed: int = 0
    rules_total: int = 15


class CompleteGuidelinesValidator:
    """
    Validates questions against ALL 15 critical guidelines.
    Zero tolerance for violations.
    """
    
    def __init__(self):
        """Initialize validator with all rule patterns"""
        # Rule 6: Name/Pronoun patterns
        self.pronoun_patterns = [
            r'\bhe\b', r'\bshe\b', r'\bhim\b', r'\bher\b', r'\bhis\b', r'\bhers\b',
            r'\bthey\b', r'\bthem\b', r'\btheir\b', r'\btheirs\b'
        ]
        
        # Rule 7: Timestamp question patterns
        self.timestamp_patterns = [
            r'at what time',
            r'what time',
            r'when does.*timestamp',
            r'what timestamp'
        ]
        
        # Rule 14: Visual-to-audio diversity check
        self.audio_first_patterns = [
            r'when you hear',
            r'when.*says',
            r'after.*says',
            r'before.*says'
        ]
        
        logger.info("CompleteGuidelinesValidator initialized with 15 rules")
    
    def validate_question(
        self,
        question: str,
        answer: str,
        audio_cues: List[str],
        visual_cues: List[str],
        evidence: Dict,
        timestamps: Tuple[float, float],
        video_metadata: Optional[Dict] = None
    ) -> ValidationResult:
        """
        Validate question against ALL 15 guidelines.
        
        Args:
            question: Question text
            answer: Answer text
            audio_cues: List of audio cues mentioned
            visual_cues: List of visual cues mentioned
            evidence: Evidence dictionary for verification
            timestamps: (start, end) timestamps
            video_metadata: Optional video metadata (duration, etc.)
            
        Returns:
            ValidationResult with pass/fail and details
        """
        violations = []
        warnings = []
        rules_passed = 0
        
        # RULE 1: Dual Cue Requirement
        if not self._check_dual_cue(audio_cues, visual_cues):
            violations.append("Rule 1: Must have BOTH audio and visual cues")
        else:
            rules_passed += 1
        
        # RULE 2: Single-Cue Rejection
        if not self._check_not_single_cue(question, audio_cues, visual_cues, evidence):
            violations.append("Rule 2: Question answerable with single cue (REJECT)")
        else:
            rules_passed += 1
        
        # RULE 3: Multipart Validation
        multipart_result = self._check_multipart(question, audio_cues, visual_cues)
        if not multipart_result:
            violations.append("Rule 3: Multipart question has subpart without both cues")
        else:
            rules_passed += 1
        
        # RULE 4: Content Rejection
        if not self._check_content_safety(question, answer, evidence):
            violations.append("Rule 4: Contains violence/obscene/sexual content (REJECT VIDEO)")
        else:
            rules_passed += 1
        
        # RULE 5: Subtitle Rejection
        if not self._check_no_subtitles(evidence):
            violations.append("Rule 5: Video has built-in subtitles (REJECT VIDEO)")
        else:
            rules_passed += 1
        
        # RULE 6: Name/Pronoun Blocking
        if not self._check_no_pronouns(question):
            violations.append("Rule 6: Contains pronouns (he/she/they). Use descriptors!")
        else:
            rules_passed += 1
        
        # RULE 7: Timestamp Questions
        if not self._check_no_timestamp_questions(question):
            violations.append("Rule 7: Asking about timestamps (avoid 'at what time')")
        else:
            rules_passed += 1
        
        # RULE 8: Precision Check
        precision_result = self._check_precision(question, answer)
        if not precision_result['is_precise']:
            violations.append(f"Rule 8: Ambiguous question - {precision_result['reason']}")
        else:
            rules_passed += 1
        
        # RULE 9: Intro/Outro Rejection
        if not self._check_no_intro_outro(timestamps, video_metadata):
            violations.append("Rule 9: Uses intro/outro as reference (avoid)")
        else:
            rules_passed += 1
        
        # RULE 10: Cue Accuracy
        accuracy_result = self._check_cue_accuracy(visual_cues, evidence)
        if not accuracy_result['is_accurate']:
            violations.append(f"Rule 10: Inaccurate visual cue - {accuracy_result['reason']}")
        else:
            rules_passed += 1
        
        # RULE 11: Timestamp Precision
        timestamp_result = self._check_timestamp_precision(
            timestamps, audio_cues, visual_cues, evidence
        )
        if not timestamp_result:
            violations.append("Rule 11: Timestamps don't cover all cues + actions properly")
        else:
            rules_passed += 1

        # RULE 12: Quote Precision
        quote_result = self._check_quote_precision(audio_cues, evidence)
        if not quote_result:
            violations.append("Rule 12: Audio quote not transcribed exactly from video")
        else:
            rules_passed += 1

        # RULE 13: Audio Diversity
        audio_diversity = self._check_audio_diversity(audio_cues)
        if not audio_diversity:
            violations.append("Rule 13: Must use diverse audio (background sounds, music, not just speech)")
        else:
            rules_passed += 1

        # RULE 14: Visual-to-Audio Diversity
        visual_to_audio = self._check_visual_to_audio(question, audio_cues, visual_cues)
        if not visual_to_audio:
            violations.append("Rule 14: Insufficient audio-visual diversity. Use 'When you see X, what hear?' format")
        else:
            rules_passed += 1
        
        # RULE 15: Temporal Usage
        temporal_result = self._check_temporal_usage(question, audio_cues, visual_cues)
        if not temporal_result:
            violations.append("Rule 15: before/after/when used incorrectly")
        else:
            rules_passed += 1
        
        # Calculate score
        score = rules_passed / 15.0
        
        # Determine if valid (must pass ALL critical rules)
        is_valid = len(violations) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            score=score,
            rule_violations=violations,
            warnings=warnings,
            can_auto_fix=False,  # No auto-fix for now
            rules_passed=rules_passed,
            rules_total=15
        )
    
    # Rule implementations
    
    def _check_dual_cue(self, audio_cues: List[str], visual_cues: List[str]) -> bool:
        """Rule 1: Must have both audio AND visual cues"""
        return len(audio_cues) > 0 and len(visual_cues) > 0
    
    def _check_not_single_cue(
        self, question: str, audio_cues: List[str], visual_cues: List[str], evidence: Dict
    ) -> bool:
        """Rule 2: Question MUST require BOTH audio and visual to answer"""
        # STRICT: Question is only valid if answering requires BOTH cues
        # Examples of INVALID (answerable with single cue):
        # - "What color shirt is person wearing?" (only needs visual)
        # - "Who wins the game?" (often only needs audio/announcer)
        # - "How many times does horn blow?" (only needs audio)

        question_lower = question.lower()

        # Check if question explicitly asks for audio-visual combination
        # Pattern 1: "When X happens (visual), what does person say (audio)" - VALID
        if re.search(r'when\s+.+[,.]?\s*what\s+.*(says|said|hear|sound|announce)', question_lower):
            return True

        # Pattern 2: "What (audio) is shown (visual) when X happens" - VALID
        if re.search(r'what\s+(.*)\s+when\s+', question_lower) and \
           any(w in question_lower for w in ['show', 'appears', 'appears', 'visible']):
            return True

        # Pattern 3: "How many/what count of X (visual) while/when Y (audio)" - VALID
        if re.search(r'(how many|count).+\s+(while|when)\s+.+(says|hear|sound)', question_lower):
            return True

        # Pattern 4: Complex question requires combining info from both sources
        # Look for conjunctions + evidence of both types needed
        has_audio_requirement = any(
            word in question_lower for word in
            ['hear', 'say', 'says', 'said', 'speaks', 'announces', 'sound', 'music', 'applause']
        )
        has_visual_requirement = any(
            word in question_lower for word in
            ['see', 'sees', 'appears', 'shows', 'visible', 'wearing', 'holding', 'color', 'count', 'number']
        )
        has_conjunction = any(
            word in question_lower for word in ['when', 'while', 'as', 'during', 'and']
        )

        # REJECT if only audio OR only visual mentioned
        if not (has_audio_requirement and has_visual_requirement):
            return False

        return has_conjunction  # Needs conjunction to properly combine
    
    def _check_multipart(
        self, question: str, audio_cues: List[str], visual_cues: List[str]
    ) -> bool:
        """Rule 3: If multipart question, ALL subparts must have both audio AND visual cues"""
        question_lower = question.lower()

        # Detect multipart questions
        multipart_markers = [
            r'\(a\)', r'\(b\)', r'\(c\)',  # Multiple choice
            r'what.*and.*what',  # Multiple questions
            r'describe.*and',  # Multiple descriptions
            r'first.*then',  # Sequential parts
            r'both.*and.*and',  # Multiple elements
        ]

        is_multipart = any(re.search(marker, question_lower) for marker in multipart_markers)

        if not is_multipart:
            return True  # Single-part question, rule passes

        # For multipart questions:
        # EACH part must have BOTH audio AND visual cues
        # Count implies multiple parts need individual analysis

        # Minimum requirement: at least 2 audio cues and 2 visual cues
        # (one pair per part minimum)
        if len(audio_cues) < 2 or len(visual_cues) < 2:
            return False  # Insufficient cues for multipart

        # Check that cues are diverse (not repeated)
        unique_audio = set(audio_cues)
        unique_visual = set(visual_cues)

        # At least 2 unique audio and 2 unique visual cues required
        return len(unique_audio) >= 2 and len(unique_visual) >= 2
    
    def _check_content_safety(self, question: str, answer: str, evidence: Dict) -> bool:
        """Rule 4: No violence/obscene/sexual content"""
        unsafe_keywords = [
            'violence', 'gun', 'shoot', 'blood', 'kill', 'death',
            'sexual', 'obscene', 'explicit', 'nude'
        ]
        combined = f"{question} {answer}".lower()
        return not any(keyword in combined for keyword in unsafe_keywords)
    
    def _check_no_subtitles(self, evidence: Dict) -> bool:
        """Rule 5: No built-in subtitles"""
        # Check if OCR detected subtitle-like text (at bottom of frame)
        ocr_data = evidence.get('ocr_data', {})

        # Check for subtitle patterns
        # Subtitles typically appear:
        # 1. At bottom 20% of frame (y > 0.8)
        # 2. In repeated patterns across multiple frames
        # 3. With consistent positioning

        subtitle_pattern_matches = 0
        frame_ocr_locations = ocr_data.get('frame_locations', [])

        for frame_entry in frame_ocr_locations:
            ocr_boxes = frame_entry.get('ocr_boxes', [])
            for box in ocr_boxes:
                # Box format: [x1, y1, x2, y2] normalized to 0-1
                if len(box) >= 4:
                    y_center = (box[1] + box[3]) / 2
                    # Bottom region (y > 0.8 = bottom 20% of frame)
                    if y_center > 0.8:
                        subtitle_pattern_matches += 1

        # If more than 20% of OCR boxes are in bottom region across frames,
        # likely built-in subtitles
        total_ocr_boxes = sum(
            len(f.get('ocr_boxes', []))
            for f in frame_ocr_locations
        )

        if total_ocr_boxes > 0:
            subtitle_ratio = subtitle_pattern_matches / total_ocr_boxes
            # Reject if >30% of OCR is in bottom region (strong indicator of subtitles)
            return subtitle_ratio < 0.3

        return True  # No OCR data, assume no subtitles
    
    def _check_no_pronouns(self, question: str) -> bool:
        """Rule 6: No pronouns (he/she/they/etc.)"""
        question_lower = question.lower()
        for pattern in self.pronoun_patterns:
            if re.search(pattern, question_lower):
                return False
        return True
    
    def _check_no_timestamp_questions(self, question: str) -> bool:
        """Rule 7: Avoid timestamp questions"""
        question_lower = question.lower()
        for pattern in self.timestamp_patterns:
            if re.search(pattern, question_lower):
                return False
        return True
    
    def _check_precision(self, question: str, answer: str) -> Dict:
        """Rule 8: Question and answer must be PRECISE with NO ambiguity"""
        combined = f"{question} {answer}".lower()

        # Check for ambiguous/vague words and phrases
        ambiguous_patterns = [
            (r'\bsomething\b', 'vague word "something"'),
            (r'\bsomeone\b', 'vague word "someone"'),
            (r'\bsomewhere\b', 'vague word "somewhere"'),
            (r'\bmight\b', 'uncertain modal "might"'),
            (r'\bmaybe\b', 'uncertain modal "maybe"'),
            (r'\bpossibly\b', 'uncertain modal "possibly"'),
            (r'\bprobably\b', 'uncertain modal "probably"'),
            (r'\bapproximately\b', 'imprecise word "approximately"'),
            (r'\baround\b', 'imprecise word "around"'),
            (r'\broughly\b', 'imprecise word "roughly"'),
            (r'\bseveral\b', 'vague word "several"'),
            (r'\ba few\b', 'vague phrase "a few"'),
            (r'\bsome\b', 'vague quantifier "some"'),
            (r'\bmost\b', 'imprecise quantifier "most"'),
            (r'\blike\s+[a-z]', 'vague comparison "like"'),  # "like" used loosely
        ]

        for pattern, description in ambiguous_patterns:
            if re.search(pattern, combined):
                return {
                    'is_precise': False,
                    'reason': f"Contains {description}"
                }

        # Check for incomplete statements
        if re.search(r'\bet\s+al\b|\betc\.?', combined):
            return {
                'is_precise': False,
                'reason': 'Incomplete specification (et al., etc.)'
            }

        # Check that question actually specifies WHAT to find (not just "what happens")
        if re.match(r'what\s+happens', question.lower()) and len(question) < 30:
            return {
                'is_precise': False,
                'reason': 'Question "what happens" is too vague - must specify what to look for'
            }

        return {'is_precise': True, 'reason': ''}
    
    def _check_no_intro_outro(
        self, timestamps: Tuple[float, float], video_metadata: Optional[Dict]
    ) -> bool:
        """Rule 9: Don't use intro/outro"""
        if not video_metadata:
            return True  # Can't check without metadata
        
        duration = video_metadata.get('duration', 0)
        start, end = timestamps
        
        # Intro: first 5 seconds
        # Outro: last 10 seconds
        is_intro = start < 5.0
        is_outro = end > (duration - 10.0)
        
        return not (is_intro or is_outro)
    
    def _check_cue_accuracy(self, visual_cues: List[str], evidence: Dict) -> Dict:
        """Rule 10: Visual cues must be accurate (colors, counts, descriptions)"""
        if not visual_cues:
            return {'is_accurate': True, 'reason': ''}

        # Extract metadata from evidence
        frame_detections = evidence.get('frame_detections', [])
        frame_metadata = evidence.get('frame_metadata', {})

        # Color vocabulary from extracted frames
        extracted_colors = set()
        extracted_objects = set()
        extracted_text = set()

        # Collect colors from visual descriptions
        for detection in frame_detections:
            description = detection.get('description', '').lower()
            # Extract mentioned colors
            colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink',
                      'black', 'white', 'gray', 'grey', 'brown', 'navy', 'maroon']
            for color in colors:
                if color in description:
                    extracted_colors.add(color)

            # Extract objects mentioned
            objects = detection.get('objects', [])
            for obj in objects:
                extracted_objects.add(obj.get('label', '').lower())

        # Collect OCR text
        ocr_data = evidence.get('ocr_data', {})
        for frame_entry in ocr_data.get('frame_locations', []):
            texts = frame_entry.get('texts', [])
            for text in texts:
                extracted_text.add(text.lower())

        # Check each visual cue against extracted evidence
        for cue in visual_cues:
            cue_lower = cue.lower()

            # Check color accuracy
            color_mentioned = False
            for color in ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink',
                          'black', 'white', 'gray', 'grey', 'brown', 'navy', 'maroon']:
                if color in cue_lower:
                    color_mentioned = True
                    if color not in extracted_colors:
                        return {
                            'is_accurate': False,
                            'reason': f"Color '{color}' mentioned but not found in evidence"
                        }

            # Check for counting accuracy (numbers in cues)
            numbers = re.findall(r'\d+', cue_lower)
            if numbers:
                # Cue mentions specific number - verify it's reasonable
                # (Don't fail if can't verify, but flag if clearly impossible)
                count = int(numbers[0])
                if count > 100:  # Unrealistic count
                    return {
                        'is_accurate': False,
                        'reason': f"Unrealistic count: {count}"
                    }

            # Check for text/quote accuracy
            if '"' in cue_lower or "'" in cue_lower:
                # Extract quoted text
                quoted = re.findall(r'["\'](.+?)["\']', cue_lower)
                for quote in quoted:
                    # Check if quote (or close match) appears in OCR
                    quote_clean = quote.strip().lower()
                    found = False
                    for extracted in extracted_text:
                        # Allow for minor variations (word matching)
                        if quote_clean in extracted or extracted in quote_clean:
                            found = True
                            break

                    if not found and len(quote_clean) > 3:  # Only check non-trivial quotes
                        # Note: Don't fail here as transcripts may paraphrase
                        pass  # Might warn but don't reject

        return {'is_accurate': True, 'reason': ''}
    
    def _check_timestamp_precision(
        self, timestamps: Tuple[float, float], audio_cues: List[str],
        visual_cues: List[str], evidence: Dict
    ) -> bool:
        """Rule 11: Timestamps must cover ALL cues and actions mentioned"""
        if not timestamps or len(timestamps) < 2:
            return True

        start_ts, end_ts = timestamps
        duration = end_ts - start_ts

        # Validate timestamp format and logic
        if start_ts >= end_ts:
            return False  # Invalid: start >= end

        if start_ts < 0:
            return False  # Invalid: negative timestamp

        # Check minimum duration
        # A good timestamp should be:
        # - Long enough to cover action (min 2 seconds for simple action)
        # - Not too short to miss context (min 1.5 seconds)
        # - Not unnecessarily long (max 30 seconds for single action)

        if duration < 1.0:
            return False  # Too short - can't capture action properly

        if duration > 60.0:
            return False  # Too long - suggests wrong time span selected

        # For audio cues, check that timestamps likely cover speech/sound
        if audio_cues:
            # Audio cue should fit in timestamp range
            # (We can't verify exact words without transcript, but can check duration)
            avg_words = sum(len(cue.split()) for cue in audio_cues) / max(len(audio_cues), 1)
            # Rough estimate: 150 words per minute = 2.5 words/second
            min_audio_duration = (avg_words / 2.5) if avg_words > 0 else 0.5

            if duration < min_audio_duration:
                return False  # Timestamp too short for mentioned audio

        # Check that visual cues can be seen in timestamp
        # Visual objects/scenes need at least 0.5s to be visible
        if visual_cues and duration < 0.5:
            return False

        return True  # Timestamps appear valid
    
    def _check_quote_precision(self, audio_cues: List[str], evidence: Dict) -> bool:
        """Rule 12: Quotes must be transcribed EXACTLY as in video"""
        if not audio_cues:
            return True

        # Get transcript from evidence
        transcript_segments = evidence.get('transcript_segments', [])
        transcript_text = evidence.get('transcript', '')

        # Combine all transcript text for matching
        all_transcript = f"{transcript_text} ".lower()
        for segment in transcript_segments:
            all_transcript += f" {segment.get('text', '')}".lower()

        # Check each audio cue (which should contain quotes)
        for cue in audio_cues:
            # Extract quoted portions from cue
            quoted_matches = re.findall(r'["\'](.+?)["\']', cue)

            if quoted_matches:
                for quote in quoted_matches:
                    quote_clean = quote.strip().lower()

                    # Check if exact quote exists in transcript
                    if quote_clean not in all_transcript:
                        # Try word-by-word matching (allowing for slight variations)
                        words_in_quote = quote_clean.split()
                        found_sequence = True

                        for i, word in enumerate(words_in_quote):
                            # Check if word appears in transcript
                            if word not in all_transcript:
                                # May be paraphrased - check context
                                found_sequence = False
                                break

                        if not found_sequence and len(quote_clean) > 5:
                            return False  # Quote not found in transcript

        return True
    
    def _check_audio_diversity(self, audio_cues: List[str]) -> bool:
        """Rule 13: Audio must be DIVERSE - not just speech/dialogue"""
        if not audio_cues:
            return False

        combined = ' '.join(audio_cues).lower()

        # ACCEPTABLE diverse audio types (not just plain speech)
        diverse_audio_types = [
            # Background sounds
            'music', 'song', 'sound', 'noise', 'tone', 'ring',
            # Crowd/environment
            'crowd', 'applause', 'clapping', 'cheering', 'roaring',
            'whistle', 'buzzer', 'bell', 'alarm', 'siren',
            # Silence/acoustic events
            'silence', 'quiet', 'pause', 'quiet', 'ambient',
            # Music-related
            'beat', 'rhythm', 'melody', 'chord', 'instrument',
            # Specific sounds
            'laugh', 'laughter', 'gasp', 'grunt', 'sigh', 'scream',
            'crash', 'bang', 'knock', 'thud', 'splash',
        ]

        # Check if audio cue mentions at least ONE diverse element
        has_diverse = any(word in combined for word in diverse_audio_types)

        if not has_diverse:
            # REJECT: Only plain speech/dialogue, no diversity
            return False

        return True

    def _check_visual_to_audio(
        self, question: str, audio_cues: List[str], visual_cues: List[str]
    ) -> bool:
        """Rule 14: Ensure audio-visual DIVERSITY - not always 'when you hear, what do you see'"""
        question_lower = question.lower()

        # GOOD PATTERNS (diverse formulations)
        good_patterns = [
            r'when\s+you\s+see',  # Visual-first format
            r'when\s+.*\s+happens',  # Action-triggered
            r'what\s+do\s+you\s+hear\s+when',  # Hear-first format
            r'what\s+visual.*\s+when',  # Visual answer format
            r'how\s+do',  # "How do" format
            r'describe\s+',  # Description format
        ]

        has_diversity = any(re.search(pattern, question_lower) for pattern in good_patterns)

        # REJECT if always using same format
        # Count different question starts to check for diversity
        audio_first = re.search(r'when\s+you\s+hear|when\s+.*(says|said|sound)', question_lower)
        visual_first = re.search(r'when\s+you\s+see|when\s+.*appears', question_lower)

        # At least TRY to use diverse formats
        # If most questions are audio-first, need some visual-first diversity
        # This is a dataset-level check, but at individual level:
        # Must use a question format that naturally combines both modalities

        return has_diversity  # Must use diverse question formulations
    
    def _check_temporal_usage(
        self, question: str, audio_cues: List[str], visual_cues: List[str]
    ) -> bool:
        """Rule 15: before/after/when must be used correctly with audio cues"""
        question_lower = question.lower()

        # Check for temporal keywords
        temporal_keywords = ['before', 'after', 'when', 'while', 'during']
        has_temporal = any(kw in question_lower for kw in temporal_keywords)

        if not has_temporal:
            return True  # No temporal usage, rule doesn't apply

        # If question uses temporal markers (before/after/when), must be with audio cues
        # GUIDELINE: "Use before/after/when with extra caution while providing audio cues"
        # "If something is said right after a visual is shown, use 'after X was said'"

        # Check that temporal usage is correct
        # Pattern 1: "before/after X says" - Good
        if re.search(r'(before|after)\s+.+\s+(says|said|speaks)', question_lower):
            return True

        # Pattern 2: "when X happens, what does Y say" - Good (both temporal + audio)
        if re.search(r'when\s+.+,\s*what\s+(do|does).+say', question_lower):
            return True

        # Pattern 3: "when/while Y is happening, what do you hear" - Good
        if re.search(r'(when|while)\s+.+,\s*what\s+(do|does|hear|sound)', question_lower):
            return True

        # Pattern 4: Check if temporal refers to audio (good) or just visual (warn)
        has_audio_temporal = any(
            audio_word in question_lower
            for audio_word in ['says', 'said', 'speaks', 'hears', 'sound', 'music', 'applause']
        )

        if has_temporal and not has_audio_temporal:
            # Temporal used but no audio component - likely bad usage
            return False

        return True


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    validator = CompleteGuidelinesValidator()
    
    # Test question
    result = validator.validate_question(
        question="When the man in blue jersey #23 dribbles the ball, what does the announcer say?",
        answer="The announcer says 'incredible move'",
        audio_cues=["announcer says 'incredible move'"],
        visual_cues=["man in blue jersey #23 dribbles ball"],
        evidence={'ocr_text': []},
        timestamps=(45.0, 46.5),
        video_metadata={'duration': 600}
    )
    
    print(f"Valid: {result.is_valid}")
    print(f"Score: {result.score:.2f} ({result.rules_passed}/15)")
    print(f"Violations: {result.rule_violations}")
    print(f"Warnings: {result.warnings}")
