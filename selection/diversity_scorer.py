"""
Diversity Scorer - Question Type Variety & Balance

Purpose: Ensure selected questions cover diverse types and skills
Compliance: Avoid redundant question types, maximize coverage
Architecture: Penalty-based scoring for type diversity

Question Types (13 from taxonomy):
1. Temporal Understanding
2. Sequential
3. Subscene
4. General Holistic Reasoning
5. Inference
6. Context
7. Needle
8. Referential Grounding
9. Counting
10. Comparative
11. Object Interaction Reasoning
12. Audio-Visual Stitching
13. Tackling Spurious Correlations
"""

import logging
from typing import List, Dict, Set, Optional, Any
from dataclasses import dataclass, field
from collections import Counter
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DiversityConfig:
    """Configuration for diversity scoring"""
    
    # Diversity penalties
    redundancy_penalty: float = 0.3      # Penalty per duplicate type
    max_same_type: int = 2               # Max questions of same type
    
    # Diversity bonuses
    unique_type_bonus: float = 0.2       # Bonus for covering new type
    full_coverage_bonus: float = 1.0     # Bonus if all major categories covered
    
    # Target distribution (ideal mix)
    target_temporal: int = 1             # Temporal/sequential questions
    target_inference: int = 1            # Inference/reasoning questions
    target_multimodal: int = 1           # Audio-visual questions
    target_other: int = 1                # Other types
    
    # Question type groupings
    temporal_types: Set[str] = field(default_factory=lambda: {
        'temporal', 'sequential'
    })
    
    inference_types: Set[str] = field(default_factory=lambda: {
        'inference', 'general_holistic', 'comparative', 'object_interaction'
    })
    
    multimodal_types: Set[str] = field(default_factory=lambda: {
        'audio_visual_stitching'
    })
    
    attention_types: Set[str] = field(default_factory=lambda: {
        'needle', 'subscene', 'referential_grounding'
    })
    
    basic_types: Set[str] = field(default_factory=lambda: {
        'counting', 'context'
    })
    
    advanced_types: Set[str] = field(default_factory=lambda: {
        'spurious_correlation'
    })


@dataclass
class QuestionTypeDistribution:
    """Distribution of question types"""
    
    type_counts: Dict[str, int] = field(default_factory=dict)
    category_counts: Dict[str, int] = field(default_factory=dict)
    total_count: int = 0
    unique_types: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'type_counts': self.type_counts,
            'category_counts': self.category_counts,
            'total_count': self.total_count,
            'unique_types': self.unique_types
        }


@dataclass
class DiversityMetrics:
    """Diversity metrics for a question set"""
    
    diversity_score: float = 0.0              # Overall diversity (0-1)
    type_distribution: QuestionTypeDistribution = field(
        default_factory=QuestionTypeDistribution
    )
    redundancy_count: int = 0                 # Number of redundant questions
    coverage_ratio: float = 0.0               # Fraction of types covered
    balance_score: float = 0.0                # How balanced the distribution is
    
    # Category coverage
    has_temporal: bool = False
    has_inference: bool = False
    has_multimodal: bool = False
    has_attention: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'diversity_score': round(self.diversity_score, 3),
            'type_distribution': self.type_distribution.to_dict(),
            'redundancy_count': self.redundancy_count,
            'coverage_ratio': round(self.coverage_ratio, 3),
            'balance_score': round(self.balance_score, 3),
            'has_temporal': self.has_temporal,
            'has_inference': self.has_inference,
            'has_multimodal': self.has_multimodal,
            'has_attention': self.has_attention
        }


class DiversityScorer:
    """
    Scores question sets for type diversity
    
    Ensures selected questions aren't all from same type (e.g., all temporal).
    Applies penalties for redundancy and bonuses for coverage.
    """
    
    def __init__(self, config: Optional[DiversityConfig] = None):
        """
        Initialize diversity scorer
        
        Args:
            config: Diversity scoring configuration
        """
        self.config = config or DiversityConfig()
        
        # All valid question types
        self.all_types = (
            self.config.temporal_types |
            self.config.inference_types |
            self.config.multimodal_types |
            self.config.attention_types |
            self.config.basic_types |
            self.config.advanced_types
        )
        
        logger.info("DiversityScorer initialized")
        logger.info(f"Tracking {len(self.all_types)} question types across 6 categories")
    
    def score_diversity(
        self,
        questions: List[Dict[str, Any]]
    ) -> DiversityMetrics:
        """
        Calculate diversity metrics for a question set
        
        Args:
            questions: List of question dictionaries
            
        Returns:
            DiversityMetrics object
        """
        logger.info(f"Scoring diversity for {len(questions)} questions")
        
        metrics = DiversityMetrics()
        
        if not questions:
            return metrics
        
        # Build type distribution
        metrics.type_distribution = self._build_distribution(questions)
        
        # Calculate redundancy
        metrics.redundancy_count = self._count_redundancy(metrics.type_distribution)
        
        # Calculate coverage
        metrics.coverage_ratio = (
            metrics.type_distribution.unique_types / len(self.all_types)
        )
        
        # Calculate balance
        metrics.balance_score = self._calculate_balance(metrics.type_distribution)
        
        # Check category coverage
        metrics.has_temporal = self._has_category(
            metrics.type_distribution, self.config.temporal_types
        )
        metrics.has_inference = self._has_category(
            metrics.type_distribution, self.config.inference_types
        )
        metrics.has_multimodal = self._has_category(
            metrics.type_distribution, self.config.multimodal_types
        )
        metrics.has_attention = self._has_category(
            metrics.type_distribution, self.config.attention_types
        )
        
        # Calculate overall diversity score
        metrics.diversity_score = self._calculate_diversity_score(metrics)
        
        logger.info(f"Diversity score: {metrics.diversity_score:.3f}")
        logger.info(f"Type coverage: {metrics.coverage_ratio:.1%} ({metrics.type_distribution.unique_types}/{len(self.all_types)} types)")
        logger.info(f"Redundancy: {metrics.redundancy_count} duplicate types")
        
        return metrics
    
    def get_diversity_penalty(
        self,
        current_selection: List[Dict[str, Any]],
        candidate: Dict[str, Any]
    ) -> float:
        """
        Calculate diversity penalty for adding a candidate question
        
        Used during iterative selection to penalize redundant types.
        
        Args:
            current_selection: Already selected questions
            candidate: Candidate question to add
            
        Returns:
            Penalty value (0.0 = no penalty, higher = more redundant)
        """
        candidate_type = self._normalize_type(candidate.get('question_type', ''))
        
        # Count how many of this type already selected
        type_count = sum(
            1 for q in current_selection
            if self._normalize_type(q.get('question_type', '')) == candidate_type
        )
        
        # No penalty for first of this type
        if type_count == 0:
            return 0.0
        
        # Linear penalty increase
        penalty = type_count * self.config.redundancy_penalty
        
        # Hard limit - huge penalty if exceeding max
        if type_count >= self.config.max_same_type:
            penalty += 10.0  # Effectively exclude
        
        return penalty
    
    def get_diversity_bonus(
        self,
        current_selection: List[Dict[str, Any]],
        candidate: Dict[str, Any]
    ) -> float:
        """
        Calculate diversity bonus for adding a candidate question
        
        Used during iterative selection to reward new types and coverage.
        
        Args:
            current_selection: Already selected questions
            candidate: Candidate question to add
            
        Returns:
            Bonus value (higher = more diverse contribution)
        """
        bonus = 0.0
        
        candidate_type = self._normalize_type(candidate.get('question_type', ''))
        
        # Get current types
        current_types = {
            self._normalize_type(q.get('question_type', ''))
            for q in current_selection
        }
        
        # Bonus for unique type
        if candidate_type not in current_types:
            bonus += self.config.unique_type_bonus
        
        # Check category coverage gaps
        current_metrics = self.score_diversity(current_selection)
        
        # Bonus for filling category gaps
        if not current_metrics.has_temporal and candidate_type in self.config.temporal_types:
            bonus += 0.3
        
        if not current_metrics.has_inference and candidate_type in self.config.inference_types:
            bonus += 0.3
        
        if not current_metrics.has_multimodal and candidate_type in self.config.multimodal_types:
            bonus += 0.4  # Multimodal is rarer
        
        if not current_metrics.has_attention and candidate_type in self.config.attention_types:
            bonus += 0.2
        
        return bonus
    
    def _build_distribution(
        self,
        questions: List[Dict[str, Any]]
    ) -> QuestionTypeDistribution:
        """Build type distribution from questions"""
        dist = QuestionTypeDistribution()
        
        # Count types
        types = [
            self._normalize_type(q.get('question_type', ''))
            for q in questions
        ]
        
        dist.type_counts = dict(Counter(types))
        dist.total_count = len(questions)
        dist.unique_types = len(dist.type_counts)
        
        # Count categories
        dist.category_counts = {
            'temporal': sum(
                count for qtype, count in dist.type_counts.items()
                if qtype in self.config.temporal_types
            ),
            'inference': sum(
                count for qtype, count in dist.type_counts.items()
                if qtype in self.config.inference_types
            ),
            'multimodal': sum(
                count for qtype, count in dist.type_counts.items()
                if qtype in self.config.multimodal_types
            ),
            'attention': sum(
                count for qtype, count in dist.type_counts.items()
                if qtype in self.config.attention_types
            ),
            'basic': sum(
                count for qtype, count in dist.type_counts.items()
                if qtype in self.config.basic_types
            ),
            'advanced': sum(
                count for qtype, count in dist.type_counts.items()
                if qtype in self.config.advanced_types
            )
        }
        
        return dist
    
    def _count_redundancy(self, dist: QuestionTypeDistribution) -> int:
        """Count redundant questions (beyond first of each type)"""
        redundancy = 0
        
        for count in dist.type_counts.values():
            if count > 1:
                redundancy += (count - 1)
        
        return redundancy
    
    def _calculate_balance(self, dist: QuestionTypeDistribution) -> float:
        """
        Calculate balance score (0-1)
        
        Perfect balance = all types have equal count
        Poor balance = some types dominate
        """
        if dist.total_count == 0:
            return 0.0
        
        counts = list(dist.type_counts.values())
        
        if not counts:
            return 0.0
        
        # Use coefficient of variation (lower = more balanced)
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        
        if mean_count == 0:
            return 0.0
        
        cv = std_count / mean_count
        
        # Convert to 0-1 score (lower CV = higher score)
        # CV of 0 = perfect balance (score 1.0)
        # CV of 1 = moderate imbalance (score 0.5)
        # CV of 2+ = severe imbalance (score < 0.33)
        balance_score = 1.0 / (1.0 + cv)
        
        return balance_score
    
    def _has_category(
        self,
        dist: QuestionTypeDistribution,
        category_types: Set[str]
    ) -> bool:
        """Check if distribution includes any type from category"""
        for qtype in dist.type_counts.keys():
            if qtype in category_types:
                return True
        return False
    
    def _calculate_diversity_score(self, metrics: DiversityMetrics) -> float:
        """
        Calculate overall diversity score (0-1)
        
        Combines coverage, balance, and category representation
        """
        # Coverage component (0-0.4)
        coverage_score = metrics.coverage_ratio * 0.4
        
        # Balance component (0-0.3)
        balance_score = metrics.balance_score * 0.3
        
        # Category coverage component (0-0.3)
        category_score = 0.0
        if metrics.has_temporal:
            category_score += 0.075
        if metrics.has_inference:
            category_score += 0.075
        if metrics.has_multimodal:
            category_score += 0.10  # Weight multimodal higher
        if metrics.has_attention:
            category_score += 0.05
        
        # Total diversity score
        diversity_score = coverage_score + balance_score + category_score
        
        # Penalty for redundancy
        if metrics.redundancy_count > 0:
            redundancy_penalty = min(
                metrics.redundancy_count * self.config.redundancy_penalty,
                0.3  # Cap penalty
            )
            diversity_score -= redundancy_penalty
        
        return max(0.0, min(diversity_score, 1.0))
    
    def _normalize_type(self, question_type: str) -> str:
        """Normalize question type string"""
        return question_type.lower().strip().replace(' ', '_')
    
    def suggest_next_type(
        self,
        current_selection: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Suggest question types to maximize diversity
        
        Args:
            current_selection: Already selected questions
            
        Returns:
            List of suggested question types (prioritized)
        """
        current_metrics = self.score_diversity(current_selection)
        current_types = set(current_metrics.type_distribution.type_counts.keys())
        
        suggestions = []
        
        # Priority 1: Fill missing major categories
        if not current_metrics.has_multimodal:
            suggestions.extend(list(self.config.multimodal_types))
        
        if not current_metrics.has_temporal:
            suggestions.extend(list(self.config.temporal_types))
        
        if not current_metrics.has_inference:
            suggestions.extend(list(self.config.inference_types))
        
        if not current_metrics.has_attention:
            suggestions.extend(list(self.config.attention_types))
        
        # Priority 2: Add types not yet used
        unused_types = self.all_types - current_types
        suggestions.extend(list(unused_types))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for t in suggestions:
            if t not in seen:
                seen.add(t)
                unique_suggestions.append(t)
        
        return unique_suggestions
    
    def format_distribution_report(
        self,
        metrics: DiversityMetrics
    ) -> str:
        """
        Format diversity metrics as human-readable report
        
        Args:
            metrics: DiversityMetrics object
            
        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("DIVERSITY REPORT")
        lines.append("=" * 60)
        
        lines.append(f"\nOverall Diversity Score: {metrics.diversity_score:.3f}/1.000")
        lines.append(f"Type Coverage: {metrics.coverage_ratio:.1%} ({metrics.type_distribution.unique_types}/{len(self.all_types)} types)")
        lines.append(f"Balance Score: {metrics.balance_score:.3f}")
        lines.append(f"Redundancy: {metrics.redundancy_count} duplicate types")
        
        lines.append("\nCategory Coverage:")
        lines.append(f"  Temporal/Sequential:  {'✓' if metrics.has_temporal else '✗'}")
        lines.append(f"  Inference/Reasoning:  {'✓' if metrics.has_inference else '✗'}")
        lines.append(f"  Multimodal:           {'✓' if metrics.has_multimodal else '✗'}")
        lines.append(f"  Attention/Detail:     {'✓' if metrics.has_attention else '✗'}")
        
        lines.append("\nType Distribution:")
        for qtype, count in sorted(
            metrics.type_distribution.type_counts.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            lines.append(f"  {qtype:.<40} {count}")
        
        lines.append("\nCategory Breakdown:")
        for category, count in sorted(
            metrics.type_distribution.category_counts.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            lines.append(f"  {category:.<40} {count}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
