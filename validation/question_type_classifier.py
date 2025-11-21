"""
Question Type Classifier - 13 NVIDIA Categories + 8 Sub-Types

Classifies questions into all 13 task types from Question Types & Skills PDF:

MAIN TYPES (13):
1.  Temporal Understanding
2.  Sequential
3.  Subscene
4.  General Holistic Reasoning
5.  Inference
6.  Context
7.  Needle
8.  Referential Grounding
9.  Counting
10. Comparative
11. Object Interaction Reasoning
12. Audio-Visual Stitching
13. Tackling Spurious Correlations

SUB-TYPES (8):
1. Human Behavior Understanding
2. Scene Recognition
3. OCR Recognition
4. Causal Reasoning
5. Intent Understanding
6. Hallucination
7. Multi-Detail Understanding
"""

import re
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class QuestionTypeResult:
    """Result of question type classification"""
    task_types: List[str]  # Primary categories (e.g., ["Temporal Understanding", "Sequential"])
    sub_task_types: List[str] = field(default_factory=list)  # Sub-categories
    confidence: float = 1.0
    reasoning: str = ""


class QuestionTypeClassifier:
    """
    Classifies questions into 13 NVIDIA task types + 8 sub-types.
    Based on Question Types & Skills PDF guidelines.
    """
    
    def __init__(self):
        """Initialize classifier with pattern rules"""
        
        # Pattern rules for each type (from PDF)
        self.type_patterns = {
            # 1. Temporal Understanding
            'Temporal Understanding': {
                'keywords': ['before', 'after', 'when', 'first', 'then', 'prior to', 'following'],
                'question_words': ['what happens', 'what does', 'what is'],
                'examples': ['What happens before', 'What does X do after', 'When does']
            },
            
            # 2. Sequential
            'Sequential': {
                'keywords': ['order', 'sequence', 'first', 'second', 'third', 'next'],
                'question_words': ['what is the order', 'which happens first', 'in what sequence'],
                'examples': ['What is the order of events', 'Which happens first']
            },
            
            # 3. Subscene
            'Subscene': {
                'keywords': ['describe', 'caption', 'when the score', 'in the quarter', 'during'],
                'question_words': ['describe what happens', 'what occurs when'],
                'examples': ['Describe what happens when score is', 'What occurs in the third quarter']
            },
            
            # 4. General Holistic Reasoning
            'General Holistic Reasoning': {
                'keywords': ['point of video', 'overall', 'entire video', 'throughout', 'whole video'],
                'question_words': ['what was the point', 'how does', 'what happens to'],
                'examples': ['What was the point of the video', 'What happens to majority']
            },
            
            # 5. Inference
            'Inference': {
                'keywords': ['why', 'purpose', 'intention', 'reason', 'because', 'feeling'],
                'question_words': ['why does', 'why did', 'what is the purpose'],
                'examples': ['Why does X happen', 'What is the purpose of', 'Why is X feeling']
            },
            
            # 6. Context
            'Context': {
                'keywords': ['background', 'foreground', 'behind', 'top left', 'bottom right', 'visible in'],
                'question_words': ['what is', 'what does', 'what visual elements'],
                'examples': ['What is visible in the background', 'What does the billboard say']
            },
            
            # 7. Needle
            'Needle': {
                'keywords': ['specific', 'exact', 'who is the player', 'describe the graphic', 'which player'],
                'question_words': ['who is', 'describe', 'which'],
                'examples': ['Who is the player that', 'Describe the graphic that pops up']
            },
            
            # 8. Referential Grounding
            'Referential Grounding': {
                'keywords': ['who are', 'what creates', 'what does.*say when', 'present when'],
                'question_words': ['who are', 'what creates', 'what causes'],
                'examples': ['Who are the two people present when', 'What creates the sound']
            },
            
            # 9. Counting
            'Counting': {
                'keywords': ['how many', 'count', 'number of', 'total'],
                'question_words': ['how many'],
                'examples': ['How many times', 'How many people']
            },
            
            # 10. Comparative
            'Comparative': {
                'keywords': ['difference', 'compare', 'distinction', 'versus', 'before and after'],
                'question_words': ['what is the difference', 'what are the distinctions'],
                'examples': ['What is the difference between', 'What are the distinctions']
            },
            
            # 11. Object Interaction Reasoning
            'Object Interaction Reasoning': {
                'keywords': ['how does.*change', 'what happens to', 'effect', 'transformation'],
                'question_words': ['how does', 'what causes', 'what is the effect'],
                'examples': ['How does the clay change', 'What causes the audio to get distorted']
            },
            
            # 12. Audio-Visual Stitching
            'Audio-Visual Stitching': {
                'keywords': ['splice', 'clip', 'editing', 'same room', 'separate clip'],
                'question_words': ['is.*in the same room', 'how do.*clips'],
                'examples': ['Is X in the same room or separate clip', 'How do the clips pace']
            },
            
            # 13. Tackling Spurious Correlations
            'Spurious Correlations': {
                'keywords': ['who are they referring to', 'unique event', 'unexpected', 'unusual'],
                'question_words': ['who are they referring', 'describe the unique'],
                'examples': ['Who are they referring to', 'Describe the unique event']
            }
        }
        
        logger.info("QuestionTypeClassifier initialized with 13 types")
    
    def classify(
        self,
        question: str,
        evidence: Optional[Dict] = None
    ) -> QuestionTypeResult:
        """
        Classify question into task types.
        
        Args:
            question: Question text
            evidence: Optional evidence for context
            
        Returns:
            QuestionTypeResult with primary and sub task types
        """
        question_lower = question.lower()
        
        # Score each type
        type_scores = {}
        for task_type, patterns in self.type_patterns.items():
            score = self._calculate_type_score(question_lower, patterns)
            if score > 0:
                type_scores[task_type] = score
        
        # Select types with score > threshold
        threshold = 0.3
        task_types = [
            task_type for task_type, score in type_scores.items()
            if score >= threshold
        ]
        
        # Sort by score
        task_types.sort(key=lambda t: type_scores[t], reverse=True)
        
        # If no types detected, use heuristic
        if not task_types:
            task_types = self._heuristic_classification(question_lower)
        
        # Detect sub-types
        sub_types = self._detect_sub_types(question_lower)
        
        return QuestionTypeResult(
            task_types=task_types,
            sub_task_types=sub_types,
            confidence=0.8,
            reasoning=f"Matched patterns for: {', '.join(task_types)}"
        )
    
    def _calculate_type_score(self, question: str, patterns: Dict) -> float:
        """Calculate score for a specific type"""
        score = 0.0
        
        # Keyword matching
        keywords = patterns.get('keywords', [])
        for keyword in keywords:
            if re.search(keyword, question):
                score += 0.4
        
        # Question word matching
        question_words = patterns.get('question_words', [])
        for qword in question_words:
            if qword in question:
                score += 0.3
        
        return min(score, 1.0)
    
    def _heuristic_classification(self, question: str) -> List[str]:
        """Fallback heuristic classification"""
        # Temporal keywords
        if any(word in question for word in ['before', 'after', 'when']):
            return ['Temporal Understanding']
        
        # Counting keywords
        if 'how many' in question:
            return ['Counting']
        
        # Inference keywords
        if question.startswith('why'):
            return ['Inference']
        
        # Default
        return ['Temporal Understanding']
    
    def _detect_sub_types(self, question: str) -> List[str]:
        """Detect sub-task types"""
        sub_types = []
        
        # Human Behavior Understanding
        if any(word in question for word in ['person', 'man', 'woman', 'wave', 'pick up']):
            sub_types.append('Human Behavior Understanding')
        
        # Scene Recognition
        if any(word in question for word in ['location', 'kitchen', 'park', 'indoor', 'outdoor']):
            sub_types.append('Scene Recognition')
        
        # OCR Recognition
        if any(word in question for word in ['text', 'title', 'number', 'sign', 'displayed']):
            sub_types.append('OCR Recognition')
        
        # Causal Reasoning
        if any(word in question for word in ['before', 'after', 'cause', 'result']):
            sub_types.append('Causal Reasoning')
        
        # Intent Understanding
        if any(word in question for word in ['glance', 'check', 'look', 'heading']):
            sub_types.append('Intent Understanding')
        
        return sub_types
    
    def ensure_coverage(
        self,
        questions: List[Dict],
        target_distribution: Optional[Dict[str, int]] = None
    ) -> Dict[str, int]:
        """
        Ensure all 13 types are covered.
        
        Args:
            questions: List of question dicts with 'task_types'
            target_distribution: Optional target count per type
            
        Returns:
            Current type distribution
        """
        if target_distribution is None:
            target_distribution = {
                'Temporal Understanding': 3,
                'Sequential': 3,
                'Subscene': 2,
                'General Holistic Reasoning': 2,
                'Inference': 4,
                'Context': 2,
                'Needle': 3,
                'Referential Grounding': 3,
                'Counting': 3,
                'Comparative': 3,
                'Object Interaction Reasoning': 2,
                'Audio-Visual Stitching': 2,
                'Spurious Correlations': 2
            }
        
        # Count current distribution
        current = {task_type: 0 for task_type in target_distribution.keys()}
        for q in questions:
            for task_type in q.get('task_types', []):
                if task_type in current:
                    current[task_type] += 1
        
        return current
    
    def get_missing_types(
        self,
        questions: List[Dict]
    ) -> List[str]:
        """Get list of types with 0 questions"""
        distribution = self.ensure_coverage(questions)
        return [task_type for task_type, count in distribution.items() if count == 0]


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    classifier = QuestionTypeClassifier()
    
    test_questions = [
        "What happens before the woman says 'this is going to work'?",
        "How many times did the player score?",
        "What is the difference between the man's shirt before and after?",
        "Why did the wicketkeeper scream in excitement?",
        "Describe what happens when the score is 118-2?"
    ]
    
    for q in test_questions:
        result = classifier.classify(q)
        print(f"\nQuestion: {q}")
        print(f"Types: {result.task_types}")
        print(f"Sub-types: {result.sub_task_types}")
