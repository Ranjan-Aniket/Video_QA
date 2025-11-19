"""
Complexity Scorer - Ensures Questions Meet Difficulty Requirements

CRITICAL GUIDELINES:
"Focus on:
- Complex questions like inferring something not explicitly stated in the video
- Combining questions from multiple segments on the video
- Challenging counting questions
- Compare the moods (emotions) of the individual
- Try to find elements in the video which are somewhat unintuitive"

This module scores question complexity and enforces minimum thresholds.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re


@dataclass
class ComplexityScore:
    """Complexity scoring result"""
    total_score: float  # 0.0 to 10.0
    meets_threshold: bool
    component_scores: Dict[str, float]
    reasoning: List[str]
    suggestions: List[str]


class ComplexityScorer:
    """
    Score question complexity across multiple dimensions.
    
    Enforces Guidelines requirement for complex, challenging, unintuitive questions.
    """
    
    # Minimum threshold for acceptable questions
    MIN_COMPLEXITY_THRESHOLD = 5.0  # out of 10.0
    
    def __init__(self):
        """Initialize complexity scorer"""
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile patterns for complexity detection"""
        
        # Inference indicators (requires reasoning beyond explicit content)
        self.inference_patterns = [
            r'\bwhy\b',
            r'\bhow come\b',
            r'\bwhat.*purpose\b',
            r'\bwhat.*meaning\b',
            r'\bwhat.*intention\b',
            r'\bwhat.*suggests\b',
            r'\bwhat.*implies\b',
            r'\binfer\b',
            r'\bdeduce\b',
            r'\bconclude\b',
            r'\bbased on\b',
        ]
        
        # Multi-segment indicators (combines from multiple parts)
        self.multi_segment_patterns = [
            r'\band\b.*\band\b',  # Multiple conditions
            r'\bcompare\b',
            r'\bcontrast\b',
            r'\bdifference\b',
            r'\bsimilarity\b',
            r'\bbetween.*and\b',
            r'\bfirst.*then.*finally\b',
            r'\bbefore.*after\b',
            r'\bthroughout\b',
        ]
        
        # Counting complexity indicators
        self.counting_complexity_patterns = [
            r'\bhow many times\b',
            r'\bhow many.*until\b',
            r'\bhow many.*after\b',
            r'\bhow many.*before\b',
            r'\bhow many.*when\b',
            r'\bhow many.*while\b',
        ]
        
        # Emotion/mood indicators
        self.emotion_patterns = [
            r'\bemoti\w+\b',
            r'\bmood\b',
            r'\bfeel\w*\b',
            r'\breact\w*\b',
            r'\bexpression\b',
            r'\btone\b',
            r'\banger\b',
            r'\bhapp\w+\b',
            r'\bsad\b',
            r'\bexcit\w+\b',
            r'\bfrustrat\w+\b',
        ]
        
        # Unintuitive/unexpected indicators
        self.unintuitive_patterns = [
            r'\bunexpected\b',
            r'\bsurprising\b',
            r'\bunusual\b',
            r'\bodd\b',
            r'\bstrange\b',
            r'\bunique\b',
            r'\bunlikely\b',
            r'\bcontradicts\b',
        ]
        
        # Simple/trivial patterns (reduce score)
        self.simple_patterns = [
            r'\bwhat color\b',
            r'\bhow many.*total\b',  # Simple counting
            r'\bwhat.*wearing\b',  # Unless combined with other complexity
            r'\bwho\b',  # Simple identification
        ]
    
    def score_complexity(
        self,
        question: str,
        answer: str,
        evidence: Dict,
        question_type: str
    ) -> ComplexityScore:
        """
        Score question complexity across multiple dimensions.
        
        Args:
            question: Question text
            answer: Answer text
            evidence: Evidence database
            question_type: Question type (from taxonomy)
            
        Returns:
            ComplexityScore with detailed breakdown
        """
        component_scores = {}
        reasoning = []
        suggestions = []
        
        # 1. INFERENCE COMPLEXITY (0-2 points)
        inference_score = self._score_inference_complexity(question, answer)
        component_scores['inference'] = inference_score
        if inference_score > 0:
            reasoning.append(
                f"Inference complexity: {inference_score:.1f}/2.0 - "
                f"Requires reasoning beyond explicit content"
            )
        else:
            suggestions.append(
                "Add inference requirement: Ask WHY something happens, "
                "what PURPOSE/MEANING it has, or what it IMPLIES"
            )
        
        # 2. MULTI-SEGMENT COMPLEXITY (0-2 points)
        multi_segment_score = self._score_multi_segment(question, evidence)
        component_scores['multi_segment'] = multi_segment_score
        if multi_segment_score > 0:
            reasoning.append(
                f"Multi-segment: {multi_segment_score:.1f}/2.0 - "
                f"Combines information from multiple video segments"
            )
        else:
            suggestions.append(
                "Combine multiple segments: Ask about events across different "
                "parts of the video (before/after, compare, throughout)"
            )
        
        # 3. COUNTING CHALLENGE (0-2 points)
        counting_score = self._score_counting_challenge(question, answer, evidence)
        component_scores['counting_challenge'] = counting_score
        if counting_score > 0:
            reasoning.append(
                f"Counting challenge: {counting_score:.1f}/2.0 - "
                f"Challenging count with conditions"
            )
        
        # 4. EMOTIONAL/MOOD COMPLEXITY (0-2 points)
        emotion_score = self._score_emotional_complexity(question, answer)
        component_scores['emotional'] = emotion_score
        if emotion_score > 0:
            reasoning.append(
                f"Emotional complexity: {emotion_score:.1f}/2.0 - "
                f"Requires understanding moods/emotions/reactions"
            )
        else:
            suggestions.append(
                "Add emotional dimension: Ask about moods, feelings, "
                "reactions, expressions, or tone changes"
            )
        
        # 5. UNINTUITIVE/UNEXPECTED (0-2 points)
        unintuitive_score = self._score_unintuitive(question, answer, evidence)
        component_scores['unintuitive'] = unintuitive_score
        if unintuitive_score > 0:
            reasoning.append(
                f"Unintuitive elements: {unintuitive_score:.1f}/2.0 - "
                f"Tests understanding of unexpected/unusual elements"
            )
        else:
            suggestions.append(
                "Find unintuitive elements: Ask about unexpected, unusual, "
                "or contradictory aspects of the video"
            )
        
        # 6. PENALTY FOR SIMPLICITY (-2 to 0 points)
        simplicity_penalty = self._calculate_simplicity_penalty(question)
        component_scores['simplicity_penalty'] = simplicity_penalty
        if simplicity_penalty < 0:
            reasoning.append(
                f"Simplicity penalty: {simplicity_penalty:.1f} - "
                f"Question has simple/trivial elements"
            )
        
        # Calculate total score
        total_score = (
            inference_score +
            multi_segment_score +
            counting_score +
            emotion_score +
            unintuitive_score +
            simplicity_penalty
        )
        
        # Clamp to 0-10 range
        total_score = max(0.0, min(10.0, total_score))
        
        # Check if meets threshold
        meets_threshold = total_score >= self.MIN_COMPLEXITY_THRESHOLD
        
        if not meets_threshold:
            suggestions.insert(0, 
                f"CRITICAL: Complexity score ({total_score:.1f}/10.0) below threshold "
                f"({self.MIN_COMPLEXITY_THRESHOLD}/10.0). Question needs to be more complex."
            )
        
        return ComplexityScore(
            total_score=total_score,
            meets_threshold=meets_threshold,
            component_scores=component_scores,
            reasoning=reasoning,
            suggestions=suggestions
        )
    
    def _score_inference_complexity(self, question: str, answer: str) -> float:
        """
        Score inference complexity (0-2 points).
        
        Guidelines: "Complex questions like inferring something not explicitly 
        stated in the video"
        """
        score = 0.0
        text = f"{question} {answer}".lower()
        
        # Check for inference patterns
        matches = 0
        for pattern in self.inference_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matches += 1
        
        if matches >= 3:
            score = 2.0  # Strong inference requirement
        elif matches >= 2:
            score = 1.5
        elif matches >= 1:
            score = 1.0
        
        # Bonus for specific inference types
        if 'purpose' in text or 'meaning' in text:
            score += 0.5
        if 'implies' in text or 'suggests' in text:
            score += 0.5
        
        return min(2.0, score)
    
    def _score_multi_segment(self, question: str, evidence: Dict) -> float:
        """
        Score multi-segment complexity (0-2 points).
        
        Guidelines: "Combining questions from multiple segments on the video"
        """
        score = 0.0
        
        # Check for multi-segment patterns
        matches = 0
        for pattern in self.multi_segment_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                matches += 1
        
        if matches >= 2:
            score = 2.0  # Strong multi-segment
        elif matches >= 1:
            score = 1.0
        
        # Check evidence for multiple timestamps
        if 'referenced_timestamps' in evidence:
            num_segments = len(evidence['referenced_timestamps'])
            if num_segments >= 3:
                score += 1.0
            elif num_segments >= 2:
                score += 0.5
        
        return min(2.0, score)
    
    def _score_counting_challenge(
        self,
        question: str,
        answer: str,
        evidence: Dict
    ) -> float:
        """
        Score counting challenge (0-2 points).
        
        Guidelines: "Challenging counting questions"
        Example: "How many times did the coach stand up immediately after 
        the whistle for foul was heard in the first quarter of the match?"
        """
        score = 0.0
        
        # Must be a counting question
        if not re.search(r'\bhow many\b', question, re.IGNORECASE):
            return 0.0
        
        # Check for complexity modifiers
        complexity_indicators = [
            r'\bimmediately after\b',
            r'\bright before\b',
            r'\bwhile\b',
            r'\bwhen\b',
            r'\bduring\b',
            r'\bin the (first|second|third|fourth)\b',
            r'\buntil\b',
            r'\bbefore.*after\b',
        ]
        
        matches = 0
        for pattern in complexity_indicators:
            if re.search(pattern, question, re.IGNORECASE):
                matches += 1
        
        if matches >= 2:
            score = 2.0  # Very challenging count
        elif matches >= 1:
            score = 1.5
        else:
            score = 0.5  # Simple count
        
        # Check if count value is high (more challenging)
        count_match = re.search(r'(\d+)', answer)
        if count_match:
            count_value = int(count_match.group(1))
            if count_value >= 10:
                score += 0.5
            elif count_value >= 5:
                score += 0.25
        
        return min(2.0, score)
    
    def _score_emotional_complexity(self, question: str, answer: str) -> float:
        """
        Score emotional/mood complexity (0-2 points).
        
        Guidelines: "Compare the moods (emotions) of the individual"
        """
        score = 0.0
        text = f"{question} {answer}".lower()
        
        # Check for emotion patterns
        matches = 0
        for pattern in self.emotion_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matches += 1
        
        if matches >= 3:
            score = 2.0  # Rich emotional analysis
        elif matches >= 2:
            score = 1.5
        elif matches >= 1:
            score = 1.0
        
        # Bonus for comparison
        if re.search(r'\bcompare.*emotion\b|\bcompare.*mood\b', text, re.IGNORECASE):
            score += 0.5
        
        return min(2.0, score)
    
    def _score_unintuitive(
        self,
        question: str,
        answer: str,
        evidence: Dict
    ) -> float:
        """
        Score unintuitive/unexpected elements (0-2 points).
        
        Guidelines: "Try to find elements in the video which are somewhat unintuitive"
        """
        score = 0.0
        text = f"{question} {answer}".lower()
        
        # Check for unintuitive patterns
        matches = 0
        for pattern in self.unintuitive_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matches += 1
        
        if matches >= 2:
            score = 2.0  # Strong unintuitive element
        elif matches >= 1:
            score = 1.5
        
        # Check if question type is "Tackling Spurious Correlations"
        # (inherently unintuitive)
        if evidence.get('question_type') == 'Tackling Spurious Correlations':
            score = 2.0
        
        return min(2.0, score)
    
    def _calculate_simplicity_penalty(self, question: str) -> float:
        """
        Calculate penalty for overly simple questions (-2 to 0 points).
        
        Simple questions reduce complexity score.
        """
        penalty = 0.0
        
        # Check for simple patterns
        for pattern in self.simple_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                penalty -= 0.5
        
        # Penalty for very short questions (likely too simple)
        word_count = len(question.split())
        if word_count < 10:
            penalty -= 0.5
        elif word_count < 15:
            penalty -= 0.25
        
        return max(-2.0, penalty)
    
    def get_complexity_recommendations(
        self,
        question: str,
        answer: str,
        evidence: Dict,
        question_type: str
    ) -> List[str]:
        """
        Get recommendations to increase question complexity.
        
        Returns:
            List of actionable recommendations
        """
        recommendations = []
        
        # Score current complexity
        score_result = self.score_complexity(question, answer, evidence, question_type)
        
        if score_result.meets_threshold:
            return ["Question meets complexity requirements. No changes needed."]
        
        # Add specific recommendations based on missing components
        components = score_result.component_scores
        
        if components.get('inference', 0) < 1.0:
            recommendations.append(
                "ADD INFERENCE: Transform to ask WHY something happens or "
                "what PURPOSE/MEANING an action/element has. "
                "Example: Instead of 'What does X do?', ask 'Why does X do this action?'"
            )
        
        if components.get('multi_segment', 0) < 1.0:
            recommendations.append(
                "COMBINE SEGMENTS: Reference multiple parts of the video. "
                "Example: 'Compare X at the beginning with Y at the end' or "
                "'What happens throughout the video when Z occurs?'"
            )
        
        if components.get('counting_challenge', 0) < 1.0 and 'count' in question_type.lower():
            recommendations.append(
                "ADD CONDITIONS TO COUNT: Make counting more challenging. "
                "Example: 'How many times does X happen immediately after Y is said "
                "in the first quarter?'"
            )
        
        if components.get('emotional', 0) < 1.0:
            recommendations.append(
                "ADD EMOTIONAL DIMENSION: Ask about moods, feelings, reactions. "
                "Example: 'How does the person's emotion change when X happens?' or "
                "'Compare the reactions of A and B'"
            )
        
        if components.get('unintuitive', 0) < 1.0:
            recommendations.append(
                "FIND UNINTUITIVE ELEMENTS: Look for unexpected, unusual, or "
                "contradictory aspects. Example: 'What unexpected event occurs when X?' or "
                "'What unusual pattern emerges?'"
            )
        
        return recommendations


def score_question_complexity(
    question: str,
    answer: str,
    evidence: Dict,
    question_type: str
) -> ComplexityScore:
    """
    Convenience function to score question complexity.
    
    Args:
        question: Question text
        answer: Answer text
        evidence: Evidence database
        question_type: Question type from taxonomy
        
    Returns:
        ComplexityScore with detailed breakdown
    """
    scorer = ComplexityScorer()
    return scorer.score_complexity(question, answer, evidence, question_type)


def validate_complexity_threshold(
    question: str,
    answer: str,
    evidence: Dict,
    question_type: str,
    min_threshold: float = 5.0
) -> Tuple[bool, str, List[str]]:
    """
    Validate that question meets minimum complexity threshold.
    
    Args:
        question: Question text
        answer: Answer text
        evidence: Evidence database
        question_type: Question type
        min_threshold: Minimum acceptable score (default 5.0/10.0)
        
    Returns:
        (passes, summary_message, recommendations)
    """
    scorer = ComplexityScorer()
    result = scorer.score_complexity(question, answer, evidence, question_type)
    
    passes = result.total_score >= min_threshold
    
    summary = (
        f"Complexity: {result.total_score:.1f}/10.0 "
        f"({'PASS' if passes else 'FAIL'} - threshold: {min_threshold}/10.0)"
    )
    
    recommendations = result.suggestions if not passes else []
    
    return (passes, summary, recommendations)
