"""
Name Anonymization Utility

Replaces all names with descriptive alternatives per guidelines:
- People names → descriptors (e.g., "John" → "man in blue jacket")
- Team names → descriptors (e.g., "Lakers" → "team in purple uniforms")
- Company/brand names → generic terms
- Movie/song names → "the movie"/"the song"

Uses spaCy NER for name detection and evidence data for descriptor generation.
"""

import logging
from typing import Dict, List, Tuple, Optional, Set
import re

logger = logging.getLogger(__name__)


class NameAnonymizer:
    """
    Anonymize names in questions and answers.

    Uses simple pattern matching and replacement for now.
    Can be enhanced with spaCy NER for better name detection.
    """

    def __init__(self):
        """Initialize name anonymizer"""
        # Common patterns to detect
        self.person_titles = {
            "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Coach", "Captain", "President"
        }

        # Common team/company indicators
        self.org_indicators = {
            "Team", "FC", "United", "City", "Inc.", "Corp.", "LLC", "Ltd."
        }

        logger.info("NameAnonymizer initialized")

    def anonymize_text(
        self,
        text: str,
        evidence: Optional[Dict] = None,
        timestamp: Optional[float] = None
    ) -> str:
        """
        Anonymize all names in text.

        Args:
            text: Text to anonymize
            evidence: Evidence data (for generating descriptors)
            timestamp: Timestamp (for finding relevant evidence)

        Returns:
            Anonymized text
        """
        # Detect potential names
        names = self._detect_names(text)

        # Replace each name with descriptor
        anonymized = text
        for name in names:
            descriptor = self._generate_descriptor(
                name,
                evidence,
                timestamp
            )
            # Case-insensitive replacement
            anonymized = re.sub(
                r'\b' + re.escape(name) + r'\b',
                descriptor,
                anonymized,
                flags=re.IGNORECASE
            )

        return anonymized

    def anonymize_qa_pair(
        self,
        question: str,
        answer: str,
        evidence: Optional[Dict] = None,
        timestamp: Optional[float] = None
    ) -> Tuple[str, str]:
        """
        Anonymize both question and answer.

        Args:
            question: Question text
            answer: Answer text
            evidence: Evidence data
            timestamp: Timestamp

        Returns:
            (anonymized_question, anonymized_answer)
        """
        # Detect names from both texts
        combined_text = question + " " + answer
        names = self._detect_names(combined_text)

        # Replace in both
        anon_question = question
        anon_answer = answer

        for name in names:
            descriptor = self._generate_descriptor(name, evidence, timestamp)

            # Replace in question
            anon_question = re.sub(
                r'\b' + re.escape(name) + r'\b',
                descriptor,
                anon_question,
                flags=re.IGNORECASE
            )

            # Replace in answer
            anon_answer = re.sub(
                r'\b' + re.escape(name) + r'\b',
                descriptor,
                anon_answer,
                flags=re.IGNORECASE
            )

        return anon_question, anon_answer

    def _detect_names(self, text: str) -> Set[str]:
        """
        Detect potential names in text.

        Simple approach:
        - Capitalized words (not at sentence start)
        - Words with titles (Mr., Dr., etc.)
        - Multi-word proper nouns

        Args:
            text: Text to analyze

        Returns:
            Set of detected names
        """
        names = set()

        # Split into sentences
        sentences = text.split('.')

        for sentence in sentences:
            words = sentence.split()

            for i, word in enumerate(words):
                # Skip first word of sentence (might be capitalized normally)
                if i == 0:
                    continue

                # Check if capitalized and not common word
                if word[0].isupper() and not self._is_common_word(word):
                    # Check if it's preceded by a title
                    if i > 0 and words[i-1].rstrip(',') in self.person_titles:
                        names.add(word)
                    # Check if standalone capitalized word
                    elif self._looks_like_name(word):
                        names.add(word)

        return names

    def _is_common_word(self, word: str) -> bool:
        """Check if word is a common non-name word"""
        common_words = {
            "I", "The", "A", "An", "In", "On", "At", "To", "For",
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
        }
        return word in common_words

    def _looks_like_name(self, word: str) -> bool:
        """Check if word looks like a name"""
        # Simple heuristic: capitalized, more than 2 letters, no numbers
        return (
            len(word) > 2 and
            word[0].isupper() and
            not any(c.isdigit() for c in word) and
            word.isalnum()
        )

    def _generate_descriptor(
        self,
        name: str,
        evidence: Optional[Dict],
        timestamp: Optional[float]
    ) -> str:
        """
        Generate visual descriptor for name.

        Args:
            name: Name to replace
            evidence: Evidence data
            timestamp: Timestamp

        Returns:
            Descriptor string
        """
        # If no evidence, use generic descriptors
        if not evidence:
            return self._generic_descriptor(name)

        # Try to find visual attributes from evidence
        if timestamp is not None:
            frame_evidence = self._find_evidence_at_timestamp(evidence, timestamp)
            if frame_evidence:
                return self._descriptor_from_evidence(name, frame_evidence)

        return self._generic_descriptor(name)

    def _generic_descriptor(self, name: str) -> str:
        """Generate generic descriptor when no evidence available"""
        # Check if it looks like a person name
        if len(name) < 15 and ' ' not in name:
            return "the person"

        # Check if it has team/org indicators
        for indicator in self.org_indicators:
            if indicator in name:
                return "the team"

        # Default
        return "the character"

    def _descriptor_from_evidence(
        self,
        name: str,
        frame_evidence: Dict
    ) -> str:
        """
        Generate descriptor from visual evidence.

        Args:
            name: Name to replace
            frame_evidence: Evidence data for frame

        Returns:
            Descriptor based on visual attributes
        """
        ground_truth = frame_evidence.get("ground_truth", {})

        # Check YOLO objects for person
        yolo_objects = ground_truth.get("yolo_objects", [])
        for obj in yolo_objects:
            if obj.get("class") == "person":
                # Try to get clothing/color info from CLIP
                clip_attrs = ground_truth.get("clip_attributes", {})
                if clip_attrs:
                    clothing = clip_attrs.get("clothing", [])
                    if clothing:
                        return f"person in {clothing[0]}"

                return "the person"

        # Check if it's a team (look for jersey numbers, uniforms)
        if any("jersey" in str(obj).lower() for obj in yolo_objects):
            return "the team"

        # Default
        return "the character"

    def _find_evidence_at_timestamp(
        self,
        evidence: Dict,
        timestamp: float,
        tolerance: float = 2.0
    ) -> Optional[Dict]:
        """Find evidence frame closest to timestamp"""
        frames = evidence.get("frames", {})

        closest_frame = None
        min_diff = float('inf')

        for frame_data in frames.values():
            frame_ts = frame_data.get("timestamp", 0)
            diff = abs(frame_ts - timestamp)

            if diff < min_diff and diff <= tolerance:
                min_diff = diff
                closest_frame = frame_data

        return closest_frame


# Example usage
if __name__ == "__main__":
    anonymizer = NameAnonymizer()

    # Test
    question = "What does Michael Jordan say to LeBron at 2:34?"
    answer = "Michael Jordan says 'Great shot!' to LeBron."

    anon_q, anon_a = anonymizer.anonymize_qa_pair(question, answer)

    print("Original Q:", question)
    print("Anonymized Q:", anon_q)
    print("\nOriginal A:", answer)
    print("Anonymized A:", anon_a)
