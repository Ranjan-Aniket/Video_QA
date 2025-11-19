"""
Pattern Learner - Learn from Gemini Failure Patterns

Purpose: Analyze failures to identify what makes questions adversarial
Compliance: Self-improving system, learn from successes and failures
Architecture: Pattern detection, trend analysis, recommendation generation

Learning Goals:
1. Identify question characteristics that expose Gemini failures
2. Detect patterns in hallucination types
3. Find optimal difficulty/complexity ranges
4. Recommend improvements for future generation
"""

import logging
from typing import List, Dict, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PatternConfig:
    """Configuration for pattern learning"""
    
    # Pattern detection
    min_occurrences: int = 3              # Minimum occurrences to detect pattern
    confidence_threshold: float = 0.7     # Minimum confidence for pattern
    
    # Time windows for trend analysis
    short_term_window: int = 10           # Last 10 videos
    long_term_window: int = 100           # Last 100 videos
    
    # Learning parameters
    success_weight: float = 1.0           # Weight for successful adversarial questions
    failure_weight: float = 0.5           # Weight for failed questions
    
    # Pattern categories to track
    track_question_types: bool = True
    track_difficulty_ranges: bool = True
    track_evidence_patterns: bool = True
    track_timing_patterns: bool = True


@dataclass
class FailurePattern:
    """Detected pattern in Gemini failures"""
    
    pattern_id: str                       # Unique pattern identifier
    pattern_type: str                     # Type of pattern (question_type, difficulty, etc.)
    description: str                      # Human-readable description
    
    # Statistics
    occurrences: int = 0                  # Number of times pattern observed
    success_rate: float = 0.0             # Success rate for this pattern
    confidence: float = 0.0               # Confidence in pattern (0-1)
    
    # Pattern characteristics
    characteristics: Dict[str, Any] = field(default_factory=dict)
    
    # Examples
    example_questions: List[str] = field(default_factory=list)
    
    # Metadata
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'pattern_id': self.pattern_id,
            'pattern_type': self.pattern_type,
            'description': self.description,
            'occurrences': self.occurrences,
            'success_rate': round(self.success_rate, 3),
            'confidence': round(self.confidence, 3),
            'characteristics': self.characteristics,
            'example_questions': self.example_questions[:3],  # First 3 examples
            'first_seen': self.first_seen,
            'last_seen': self.last_seen
        }


@dataclass
class LearningInsights:
    """Insights learned from failure analysis"""
    
    # Detected patterns
    patterns: List[FailurePattern] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Performance insights
    best_question_types: List[Tuple[str, float]] = field(default_factory=list)
    worst_question_types: List[Tuple[str, float]] = field(default_factory=list)
    
    optimal_difficulty_range: Optional[Tuple[float, float]] = None
    
    # Trends
    improving_areas: List[str] = field(default_factory=list)
    declining_areas: List[str] = field(default_factory=list)
    
    # Statistics
    total_patterns_detected: int = 0
    high_confidence_patterns: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'patterns': [p.to_dict() for p in self.patterns],
            'recommendations': self.recommendations,
            'best_question_types': [
                {'type': t, 'success_rate': round(rate, 3)}
                for t, rate in self.best_question_types
            ],
            'worst_question_types': [
                {'type': t, 'success_rate': round(rate, 3)}
                for t, rate in self.worst_question_types
            ],
            'optimal_difficulty_range': (
                [round(self.optimal_difficulty_range[0], 1), 
                 round(self.optimal_difficulty_range[1], 1)]
                if self.optimal_difficulty_range else None
            ),
            'improving_areas': self.improving_areas,
            'declining_areas': self.declining_areas,
            'total_patterns_detected': self.total_patterns_detected,
            'high_confidence_patterns': self.high_confidence_patterns
        }


class PatternLearner:
    """
    Learns patterns from Gemini failures to improve future generation
    
    Analyzes feedback history to identify:
    - Question types that consistently expose failures
    - Difficulty ranges with highest failure rates
    - Evidence patterns in successful adversarial questions
    - Temporal trends in performance
    """
    
    def __init__(self, config: Optional[PatternConfig] = None):
        """
        Initialize pattern learner
        
        Args:
            config: Pattern learning configuration
        """
        self.config = config or PatternConfig()
        
        # Pattern storage
        self.detected_patterns: List[FailurePattern] = []
        self.pattern_index: Dict[str, FailurePattern] = {}
        
        logger.info("PatternLearner initialized")
        logger.info(f"Min occurrences for pattern: {self.config.min_occurrences}")
        logger.info(f"Confidence threshold: {self.config.confidence_threshold}")
    
    def learn_from_feedback(
        self,
        feedback_results: List[Dict[str, Any]],
        selected_questions: Optional[List[Dict[str, Any]]] = None
    ) -> LearningInsights:
        """
        Learn patterns from feedback history
        
        Args:
            feedback_results: List of FeedbackResult dictionaries
            selected_questions: List of questions selected as best adversarial
            
        Returns:
            LearningInsights with detected patterns and recommendations
        """
        logger.info(f"Learning from {len(feedback_results)} feedback results")
        
        insights = LearningInsights()
        
        # Detect patterns by category
        if self.config.track_question_types:
            type_patterns = self._detect_type_patterns(feedback_results)
            insights.patterns.extend(type_patterns)
        
        if self.config.track_difficulty_ranges:
            difficulty_patterns = self._detect_difficulty_patterns(
                feedback_results, selected_questions
            )
            insights.patterns.extend(difficulty_patterns)
        
        if self.config.track_evidence_patterns:
            evidence_patterns = self._detect_evidence_patterns(selected_questions)
            insights.patterns.extend(evidence_patterns)
        
        # Store all patterns
        for pattern in insights.patterns:
            self.detected_patterns.append(pattern)
            self.pattern_index[pattern.pattern_id] = pattern
        
        # Analyze performance by type
        insights.best_question_types, insights.worst_question_types = (
            self._rank_question_types(feedback_results)
        )
        
        # Find optimal difficulty range
        if selected_questions:
            insights.optimal_difficulty_range = self._find_optimal_difficulty(
                selected_questions
            )
        
        # Detect trends
        if len(feedback_results) >= self.config.short_term_window:
            insights.improving_areas, insights.declining_areas = (
                self._detect_trends(feedback_results)
            )
        
        # Generate recommendations
        insights.recommendations = self._generate_recommendations(insights)
        
        # Summary statistics
        insights.total_patterns_detected = len(insights.patterns)
        insights.high_confidence_patterns = sum(
            1 for p in insights.patterns
            if p.confidence >= self.config.confidence_threshold
        )
        
        logger.info(f"Detected {insights.total_patterns_detected} patterns "
                   f"({insights.high_confidence_patterns} high confidence)")
        
        return insights
    
    def _detect_type_patterns(
        self,
        feedback_results: List[Dict[str, Any]]
    ) -> List[FailurePattern]:
        """Detect patterns in question type performance"""
        
        patterns = []
        
        # Aggregate performance by type across all feedback
        type_stats = defaultdict(lambda: {
            'total': 0,
            'gemini_failures': 0,
            'hallucinations': 0
        })
        
        for feedback in feedback_results:
            perf_by_type = feedback.get('performance_by_type', {})
            
            for qtype, perf in perf_by_type.items():
                stats = type_stats[qtype]
                stats['total'] += perf['total']
                stats['gemini_failures'] += int(
                    perf['total'] * perf.get('gemini_fail_rate', 0.0)
                )
                stats['hallucinations'] += int(
                    perf['total'] * perf.get('hallucination_rate', 0.0)
                )
        
        # Detect patterns for types with sufficient data
        for qtype, stats in type_stats.items():
            if stats['total'] < self.config.min_occurrences:
                continue
            
            gemini_fail_rate = stats['gemini_failures'] / stats['total']
            hall_rate = stats['hallucinations'] / stats['total']
            
            # Success = high Gemini failure, low hallucination
            success_rate = gemini_fail_rate * (1.0 - hall_rate)
            
            # Confidence based on sample size
            confidence = min(1.0, stats['total'] / (self.config.min_occurrences * 3))
            
            if confidence >= self.config.confidence_threshold and success_rate > 0.3:
                pattern = FailurePattern(
                    pattern_id=f"type_{qtype}",
                    pattern_type="question_type",
                    description=f"{qtype} questions effectively expose Gemini failures",
                    occurrences=stats['total'],
                    success_rate=success_rate,
                    confidence=confidence,
                    characteristics={
                        'question_type': qtype,
                        'gemini_fail_rate': gemini_fail_rate,
                        'hallucination_rate': hall_rate
                    }
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_difficulty_patterns(
        self,
        feedback_results: List[Dict[str, Any]],
        selected_questions: Optional[List[Dict[str, Any]]]
    ) -> List[FailurePattern]:
        """Detect patterns in difficulty ranges"""
        
        patterns = []
        
        if not selected_questions:
            return patterns
        
        # Extract difficulty scores from selected questions
        difficulty_scores = []
        for q in selected_questions:
            if 'difficulty_score' in q:
                difficulty_scores.append(q['difficulty_score'])
        
        if len(difficulty_scores) < self.config.min_occurrences:
            return patterns
        
        # Analyze difficulty distribution
        scores_array = np.array(difficulty_scores)
        
        mean_difficulty = np.mean(scores_array)
        std_difficulty = np.std(scores_array)
        median_difficulty = np.median(scores_array)
        
        # Detect if there's a "sweet spot" difficulty range
        # High concentration of selected questions in certain range
        
        # Count by difficulty bucket
        buckets = {
            'easy': (0, 3),
            'medium': (3, 6),
            'hard': (6, 8),
            'expert': (8, 10)
        }
        
        bucket_counts = Counter()
        for score in difficulty_scores:
            for bucket_name, (low, high) in buckets.items():
                if low <= score < high:
                    bucket_counts[bucket_name] += 1
                    break
        
        # Find dominant bucket
        if bucket_counts:
            dominant_bucket = bucket_counts.most_common(1)[0]
            bucket_name, count = dominant_bucket
            
            if count >= self.config.min_occurrences:
                low, high = buckets[bucket_name]
                
                pattern = FailurePattern(
                    pattern_id=f"difficulty_{bucket_name}",
                    pattern_type="difficulty_range",
                    description=f"{bucket_name.capitalize()} difficulty ({low}-{high}) "
                               f"produces most effective adversarial questions",
                    occurrences=count,
                    success_rate=count / len(difficulty_scores),
                    confidence=min(1.0, count / (len(difficulty_scores) * 0.5)),
                    characteristics={
                        'difficulty_range': bucket_name,
                        'min_score': low,
                        'max_score': high,
                        'mean_difficulty': float(mean_difficulty),
                        'median_difficulty': float(median_difficulty)
                    }
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_evidence_patterns(
        self,
        selected_questions: Optional[List[Dict[str, Any]]]
    ) -> List[FailurePattern]:
        """Detect patterns in evidence characteristics"""
        
        patterns = []
        
        if not selected_questions:
            return patterns
        
        # Analyze evidence characteristics
        visual_counts = []
        audio_counts = []
        multimodal_count = 0
        
        for q in selected_questions:
            evidence = q.get('evidence', {})
            
            if not isinstance(evidence, dict):
                continue
            
            visual_cues = evidence.get('visual_cues', [])
            audio_cues = evidence.get('audio_cues', [])
            
            visual_counts.append(len(visual_cues))
            audio_counts.append(len(audio_cues))
            
            if len(visual_cues) > 0 and len(audio_cues) > 0:
                multimodal_count += 1
        
        if len(visual_counts) < self.config.min_occurrences:
            return patterns
        
        # Multimodal pattern
        multimodal_rate = multimodal_count / len(selected_questions)
        
        if multimodal_rate >= 0.5 and multimodal_count >= self.config.min_occurrences:
            pattern = FailurePattern(
                pattern_id="evidence_multimodal",
                pattern_type="evidence_pattern",
                description="Questions with both visual and audio cues are more effective",
                occurrences=multimodal_count,
                success_rate=multimodal_rate,
                confidence=min(1.0, multimodal_count / (self.config.min_occurrences * 2)),
                characteristics={
                    'multimodal_rate': multimodal_rate,
                    'avg_visual_cues': float(np.mean(visual_counts)),
                    'avg_audio_cues': float(np.mean(audio_counts))
                }
            )
            patterns.append(pattern)
        
        return patterns
    
    def _rank_question_types(
        self,
        feedback_results: List[Dict[str, Any]]
    ) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """Rank question types by success rate"""
        
        # Aggregate across all feedback
        type_performance = defaultdict(lambda: {'successes': 0, 'total': 0})
        
        for feedback in feedback_results:
            perf_by_type = feedback.get('performance_by_type', {})
            
            for qtype, perf in perf_by_type.items():
                gemini_fail_rate = perf.get('gemini_fail_rate', 0.0)
                hall_rate = perf.get('hallucination_rate', 0.0)
                total = perf.get('total', 0)
                
                # Success = Gemini failed AND we didn't hallucinate
                success_rate = gemini_fail_rate * (1.0 - hall_rate)
                
                type_performance[qtype]['successes'] += total * success_rate
                type_performance[qtype]['total'] += total
        
        # Calculate overall success rate per type
        type_rates = []
        for qtype, stats in type_performance.items():
            if stats['total'] >= self.config.min_occurrences:
                rate = stats['successes'] / stats['total']
                type_rates.append((qtype, rate))
        
        # Sort by success rate
        type_rates.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 3 and bottom 3
        best = type_rates[:3]
        worst = type_rates[-3:] if len(type_rates) > 3 else []
        
        return best, worst
    
    def _find_optimal_difficulty(
        self,
        selected_questions: List[Dict[str, Any]]
    ) -> Optional[Tuple[float, float]]:
        """Find optimal difficulty range from selected questions"""
        
        difficulty_scores = [
            q['difficulty_score'] for q in selected_questions
            if 'difficulty_score' in q
        ]
        
        if len(difficulty_scores) < self.config.min_occurrences:
            return None
        
        scores_array = np.array(difficulty_scores)
        
        # Use percentiles to define optimal range
        p25 = np.percentile(scores_array, 25)
        p75 = np.percentile(scores_array, 75)
        
        return (float(p25), float(p75))
    
    def _detect_trends(
        self,
        feedback_results: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str]]:
        """Detect improving and declining trends"""
        
        improving = []
        declining = []
        
        if len(feedback_results) < self.config.short_term_window * 2:
            return improving, declining
        
        # Compare recent vs older results
        recent = feedback_results[-self.config.short_term_window:]
        older = feedback_results[-self.config.short_term_window*2:-self.config.short_term_window]
        
        # Calculate average metrics
        def avg_metric(results, metric):
            values = [r.get(metric, 0) for r in results]
            return sum(values) / len(values) if values else 0
        
        recent_validation = avg_metric(recent, 'validation_pass_rate')
        older_validation = avg_metric(older, 'validation_pass_rate')
        
        recent_gemini = avg_metric(recent, 'gemini_fail_rate')
        older_gemini = avg_metric(older, 'gemini_fail_rate')
        
        recent_hall = avg_metric(recent, 'hallucination_rate')
        older_hall = avg_metric(older, 'hallucination_rate')
        
        # Detect significant changes (>5%)
        if recent_validation - older_validation > 0.05:
            improving.append("validation_pass_rate")
        elif older_validation - recent_validation > 0.05:
            declining.append("validation_pass_rate")
        
        if recent_gemini - older_gemini > 0.05:
            improving.append("gemini_fail_rate")
        elif older_gemini - recent_gemini > 0.05:
            declining.append("gemini_fail_rate")
        
        if older_hall - recent_hall > 0.02:  # Lower is better
            improving.append("hallucination_rate")
        elif recent_hall - older_hall > 0.02:
            declining.append("hallucination_rate")
        
        return improving, declining
    
    def _generate_recommendations(self, insights: LearningInsights) -> List[str]:
        """Generate actionable recommendations from insights"""
        
        recommendations = []
        
        # Recommendations based on best question types
        if insights.best_question_types:
            best_types = [t for t, _ in insights.best_question_types]
            recommendations.append(
                f"Focus on generating more {', '.join(best_types)} questions "
                f"(highest success rates)"
            )
        
        # Recommendations based on worst question types
        if insights.worst_question_types:
            worst_types = [t for t, _ in insights.worst_question_types]
            recommendations.append(
                f"Reduce or improve {', '.join(worst_types)} questions "
                f"(lowest success rates)"
            )
        
        # Recommendations based on difficulty
        if insights.optimal_difficulty_range:
            low, high = insights.optimal_difficulty_range
            recommendations.append(
                f"Target difficulty range {low:.1f}-{high:.1f} "
                f"for optimal adversarial effectiveness"
            )
        
        # Recommendations based on patterns
        multimodal_patterns = [
            p for p in insights.patterns
            if p.pattern_id == 'evidence_multimodal' and p.confidence >= 0.7
        ]
        
        if multimodal_patterns:
            recommendations.append(
                "Prioritize questions with both visual and audio cues "
                "(multimodal questions are more effective)"
            )
        
        # Recommendations based on trends
        if 'hallucination_rate' in insights.declining_areas:
            recommendations.append(
                "⚠️ Hallucination rate increasing - strengthen validation or "
                "reduce Tier 3 creative generation"
            )
        
        if 'gemini_fail_rate' in insights.declining_areas:
            recommendations.append(
                "⚠️ Gemini failure rate decreasing - increase question difficulty "
                "or complexity"
            )
        
        if not recommendations:
            recommendations.append("No specific recommendations - continue current strategy")
        
        return recommendations
    
    def get_pattern_summary(self) -> str:
        """Get human-readable summary of detected patterns"""
        
        lines = []
        lines.append("="*70)
        lines.append("LEARNED PATTERNS SUMMARY")
        lines.append("="*70)
        
        if not self.detected_patterns:
            lines.append("No patterns detected yet.")
            return "\n".join(lines)
        
        # Group by pattern type
        by_type = defaultdict(list)
        for pattern in self.detected_patterns:
            by_type[pattern.pattern_type].append(pattern)
        
        for ptype, patterns in by_type.items():
            lines.append(f"\n{ptype.upper().replace('_', ' ')}:")
            for pattern in sorted(patterns, key=lambda p: p.confidence, reverse=True):
                lines.append(f"  • {pattern.description}")
                lines.append(f"    Success Rate: {pattern.success_rate:.1%}, "
                           f"Confidence: {pattern.confidence:.1%}, "
                           f"Occurrences: {pattern.occurrences}")
        
        lines.append("="*70)
        
        return "\n".join(lines)
