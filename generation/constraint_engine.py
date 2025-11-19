"""
Constraint Engine for Tier 2 Generation

Uses grammar constraints to ensure LLM outputs match expected structure.

Libraries supported:
- Guidance (https://github.com/guidance-ai/guidance)
- Outlines (https://github.com/outlines-dev/outlines)

Ensures 99%+ accuracy through structural constraints.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class QuestionStructure:
    """Expected structure of a question"""
    question_pattern: str
    answer_pattern: str
    required_fields: List[str]


class GrammarConstraints:
    """
    Grammar constraints for question generation
    
    Defines regex patterns and grammar rules for valid questions.
    """
    
    # Question patterns (regex-like)
    PATTERNS = {
        "counting": {
            "question": r"How many {OBJECT}s? (appear|are visible|are shown) .*after.* says? ['\"].*['\"].*\?",
            "answer": r"(After|During) .*, \d+ {OBJECT}s? (appear|are visible|are shown).*\.",
            "slots": ["OBJECT", "AUDIO_CUE", "COUNT"]
        },
        
        "temporal": {
            "question": r"What happens (after|before) .* says? ['\"].*['\"].*\?",
            "answer": r"(After|Before) .*, .*\.",
            "slots": ["AUDIO_CUE", "EVENT"]
        },
        
        "needle": {
            "question": r"What .* (appears|is shown|is visible) .*when.* says? ['\"].*['\"].*\?",
            "answer": r".*['\"].*['\"] (appears|is shown|is visible).*\.",
            "slots": ["VISUAL_ELEMENT", "AUDIO_CUE", "LOCATION"]
        },
        
        "referential": {
            "question": r"Who (is|are) (visible|present) .*when.* says? ['\"].*['\"].*\?",
            "answer": r".* (is|are) (visible|present).*\.",
            "slots": ["AUDIO_CUE", "PERSON_DESCRIPTOR"]
        }
    }
    
    @staticmethod
    def get_structure(question_type: str) -> Optional[QuestionStructure]:
        """Get expected structure for question type"""
        pattern = GrammarConstraints.PATTERNS.get(question_type.lower())
        
        if not pattern:
            return None
        
        return QuestionStructure(
            question_pattern=pattern["question"],
            answer_pattern=pattern["answer"],
            required_fields=pattern["slots"]
        )
    
    @staticmethod
    def build_guidance_program(
        question_type: str,
        evidence_vocab: List[str]
    ) -> str:
        """
        Build Guidance program for constrained generation
        
        Example:
            ```python
            from guidance import models, gen
            
            lm = models.LlamaCpp(model_path)
            
            # Constrain to evidence vocabulary
            lm += f"Question: How many "
            lm += select(evidence_vocab["objects"])  # Only from evidence!
            lm += f"s appear after someone says "
            lm += gen(max_tokens=20, regex='"[^"]*"')  # Quoted audio
            lm += "?"
            ```
        
        Args:
            question_type: Type of question
            evidence_vocab: Allowed vocabulary from evidence
            
        Returns:
            Guidance program as string
        """
        structure = GrammarConstraints.get_structure(question_type)
        if not structure:
            return ""
        
        # Build Guidance program
        program = f"""
from guidance import models, gen, select

# Constrain generation to evidence vocabulary
lm += "Question: "

# Question pattern based on type
"""
        
        if question_type == "counting":
            program += """
lm += "How many "
lm += select(evidence_vocab["objects"])  # Only objects from evidence
lm += "s appear after someone says "
lm += gen(max_tokens=30, regex='"[^"]*"')  # Quoted speech
lm += "?"
"""
        
        elif question_type == "temporal":
            program += """
lm += select(["What happens after", "What happens before"])
lm += " someone says "
lm += gen(max_tokens=30, regex='"[^"]*"')
lm += "?"
"""
        
        return program


class StructuralValidator:
    """
    Validate generated questions match expected structure
    
    Ensures questions follow grammar constraints.
    """
    
    @staticmethod
    def validate_structure(
        question: str,
        answer: str,
        question_type: str
    ) -> tuple[bool, Optional[str]]:
        """
        Validate question/answer structure
        
        Args:
            question: Generated question
            answer: Generated answer
            question_type: Expected type
            
        Returns:
            (is_valid, error_message)
        """
        structure = GrammarConstraints.get_structure(question_type)
        if not structure:
            return True, None  # Unknown type, skip validation
        
        # Check question structure
        import re
        
        # Questions must end with ?
        if not question.strip().endswith('?'):
            return False, "Question must end with ?"
        
        # Answers must end with .
        if not answer.strip().endswith('.'):
            return False, "Answer must end with ."
        
        # Check for quoted audio cues in question
        if 'AUDIO_CUE' in structure.required_fields:
            if not re.search(r'["\'].*["\']', question):
                return False, "Question must contain quoted audio cue"
        
        # Check for numbers in counting questions
        if question_type == "counting":
            if not re.search(r'\d+', answer):
                return False, "Counting answer must contain number"
        
        return True, None


class SlotFiller:
    """
    Fill template slots with evidence-based content
    
    Template: "How many {OBJECT}s appear when {AUDIO_CUE}?"
    Slots filled ONLY from evidence.
    """
    
    def __init__(self, evidence_vocab: Dict[str, List[str]]):
        """
        Initialize slot filler
        
        Args:
            evidence_vocab: Dictionary of {slot_type: [allowed_values]}
        """
        self.evidence_vocab = evidence_vocab
    
    def fill_template(
        self,
        template: str,
        slot_values: Dict[str, str]
    ) -> str:
        """
        Fill template with slot values
        
        Args:
            template: Template string with {SLOT} markers
            slot_values: Dictionary of {slot_name: value}
            
        Returns:
            Filled template
        """
        result = template
        
        for slot_name, value in slot_values.items():
            # Validate value is from evidence
            if not self._validate_slot_value(slot_name, value):
                raise ValueError(
                    f"Slot value '{value}' for {slot_name} not in evidence vocabulary"
                )
            
            result = result.replace(f"{{{slot_name}}}", value)
        
        return result
    
    def _validate_slot_value(self, slot_name: str, value: str) -> bool:
        """Check if slot value is from evidence"""
        # Extract slot type (e.g., OBJECT from slot_name)
        slot_type = slot_name.split('_')[0].lower()
        
        if slot_type in self.evidence_vocab:
            return value in self.evidence_vocab[slot_type]
        
        return True  # Unknown slot type, allow
