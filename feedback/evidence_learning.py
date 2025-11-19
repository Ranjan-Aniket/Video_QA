"""
Evidence Learning - Learn from human corrections to improve AI accuracy

Tracks patterns in AI errors and human corrections to:
1. Identify which AI models make what types of errors
2. Adjust confidence thresholds dynamically
3. Detect systematic biases
4. Provide feedback for model fine-tuning
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import json

from database.evidence_operations import EvidenceOperations

logger = logging.getLogger(__name__)


class EvidenceLearner:
    """
    Learns from human corrections to evidence items

    Tracks error patterns and suggests improvements
    """

    def __init__(
        self,
        min_samples: int = 10,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize evidence learner

        Args:
            min_samples: Minimum samples needed to identify a pattern
            confidence_threshold: Min confidence for pattern detection
        """
        self.min_samples = min_samples
        self.confidence_threshold = confidence_threshold

    def analyze_ai_performance(
        self,
        video_id: Optional[str] = None,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze AI performance across all evidence types

        Args:
            video_id: Optional video ID to filter
            days_back: Number of days to analyze

        Returns:
            Performance analysis dict
        """
        logger.info(f"Analyzing AI performance (last {days_back} days)")

        # Get all reviewed evidence items
        # Note: This is simplified - in production you'd query with date filters
        evidence_items = self._get_reviewed_evidence(video_id, days_back)

        if not evidence_items:
            logger.warning("No reviewed evidence found")
            return {}

        # Analyze by model
        gpt4_stats = self._analyze_model_performance(evidence_items, 'gpt4')
        claude_stats = self._analyze_model_performance(evidence_items, 'claude')
        open_model_stats = self._analyze_model_performance(evidence_items, 'open')

        # Overall stats
        total_items = len(evidence_items)
        approved = sum(1 for item in evidence_items if item.get('human_review_status') == 'approved')
        corrected = sum(1 for item in evidence_items if item.get('human_review_status') == 'corrected')

        overall_accuracy = approved / total_items if total_items > 0 else 0

        return {
            'period_days': days_back,
            'total_items': total_items,
            'approved': approved,
            'corrected': corrected,
            'overall_accuracy': round(overall_accuracy, 3),
            'gpt4': gpt4_stats,
            'claude': claude_stats,
            'open_models': open_model_stats,
            'error_patterns': self._detect_error_patterns(evidence_items)
        }

    def _get_reviewed_evidence(
        self,
        video_id: Optional[str],
        days_back: int
    ) -> List[Dict]:
        """Get reviewed evidence items"""
        # Simplified - in production, query database with proper filters
        # For now, return empty list as placeholder
        return []

    def _analyze_model_performance(
        self,
        evidence_items: List[Dict],
        model_name: str
    ) -> Dict[str, Any]:
        """
        Analyze performance for a specific model

        Args:
            evidence_items: List of evidence items
            model_name: 'gpt4', 'claude', or 'open'

        Returns:
            Model performance stats
        """
        # Count items where this model had a prediction
        model_items = [
            item for item in evidence_items
            if item.get(f'{model_name}_prediction')
        ]

        if not model_items:
            return {'total': 0, 'accuracy': 0.0}

        # Count correct predictions
        # (Simplified - need actual comparison logic)
        correct = sum(
            1 for item in model_items
            if item.get('human_review_status') == 'approved'
        )

        total = len(model_items)
        accuracy = correct / total if total > 0 else 0

        # Error breakdown by evidence type
        errors_by_type = defaultdict(int)
        for item in model_items:
            if item.get('human_review_status') == 'corrected':
                evidence_type = item.get('evidence_type', 'unknown')
                errors_by_type[evidence_type] += 1

        return {
            'total': total,
            'correct': correct,
            'incorrect': total - correct,
            'accuracy': round(accuracy, 3),
            'errors_by_type': dict(errors_by_type)
        }

    def _detect_error_patterns(self, evidence_items: List[Dict]) -> List[Dict]:
        """
        Detect systematic error patterns

        Returns:
            List of detected patterns
        """
        patterns = []

        # Group errors by type and category
        error_groups = defaultdict(list)

        for item in evidence_items:
            if item.get('human_review_status') == 'corrected':
                key = (
                    item.get('evidence_type'),
                    item.get('correction_category')
                )
                error_groups[key].append(item)

        # Identify significant patterns
        for (evidence_type, category), items in error_groups.items():
            if len(items) >= self.min_samples:
                confidence = len(items) / len(evidence_items)

                if confidence >= self.confidence_threshold:
                    patterns.append({
                        'evidence_type': evidence_type,
                        'category': category,
                        'occurrences': len(items),
                        'confidence': round(confidence, 3),
                        'description': self._describe_pattern(evidence_type, category, items)
                    })

        return patterns

    def _describe_pattern(
        self,
        evidence_type: str,
        category: str,
        items: List[Dict]
    ) -> str:
        """Generate human-readable pattern description"""
        return f"{category} errors in {evidence_type} detection (n={len(items)})"

    def get_confidence_recommendations(
        self,
        performance_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Recommend confidence threshold adjustments

        Args:
            performance_data: Performance analysis from analyze_ai_performance()

        Returns:
            Dict of recommended thresholds by evidence type
        """
        recommendations = {}

        # Analyze error patterns
        error_patterns = performance_data.get('error_patterns', [])

        for pattern in error_patterns:
            evidence_type = pattern['evidence_type']
            confidence = pattern['confidence']

            # If high error rate, lower confidence threshold
            if confidence > 0.3:  # >30% error rate
                recommendations[evidence_type] = 0.70  # Lower threshold
            elif confidence > 0.15:  # >15% error rate
                recommendations[evidence_type] = 0.80
            else:
                recommendations[evidence_type] = 0.85  # Default

        return recommendations

    def generate_learning_report(
        self,
        video_id: Optional[str] = None,
        days_back: int = 30
    ) -> str:
        """
        Generate comprehensive learning report

        Args:
            video_id: Optional video ID
            days_back: Days to analyze

        Returns:
            Markdown-formatted report
        """
        performance = self.analyze_ai_performance(video_id, days_back)

        if not performance:
            return "No data available for analysis"

        report = []
        report.append("# Evidence Learning Report")
        report.append(f"\nAnalysis Period: Last {days_back} days")
        report.append(f"\nTotal Items Reviewed: {performance['total_items']}")
        report.append(f"\nOverall Accuracy: {performance['overall_accuracy']:.1%}")
        report.append("\n## AI Model Performance\n")

        for model in ['gpt4', 'claude', 'open_models']:
            stats = performance.get(model, {})
            report.append(f"\n### {model.upper()}")
            report.append(f"- Accuracy: {stats.get('accuracy', 0):.1%}")
            report.append(f"- Correct: {stats.get('correct', 0)}")
            report.append(f"- Incorrect: {stats.get('incorrect', 0)}")

            errors = stats.get('errors_by_type', {})
            if errors:
                report.append("\nErrors by type:")
                for error_type, count in sorted(errors.items(), key=lambda x: x[1], reverse=True):
                    report.append(f"  - {error_type}: {count}")

        # Error patterns
        patterns = performance.get('error_patterns', [])
        if patterns:
            report.append("\n## Detected Error Patterns\n")
            for pattern in sorted(patterns, key=lambda x: x['confidence'], reverse=True):
                report.append(f"\n### {pattern['description']}")
                report.append(f"- Occurrences: {pattern['occurrences']}")
                report.append(f"- Confidence: {pattern['confidence']:.1%}")

        # Recommendations
        recommendations = self.get_confidence_recommendations(performance)
        if recommendations:
            report.append("\n## Recommended Threshold Adjustments\n")
            for evidence_type, threshold in sorted(recommendations.items()):
                report.append(f"- {evidence_type}: {threshold:.2f}")

        return "\n".join(report)


class ReviewerPerformanceAnalyzer:
    """
    Analyzes reviewer performance and inter-rater reliability
    """

    def __init__(self):
        """Initialize analyzer"""
        self.evidence_ops = EvidenceOperations

    def analyze_reviewer(
        self,
        reviewer_id: str,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze individual reviewer performance

        Args:
            reviewer_id: Reviewer ID
            days_back: Days to analyze

        Returns:
            Performance stats
        """
        stats = self.evidence_ops.get_reviewer_stats(reviewer_id)

        if not stats:
            return {}

        return {
            'reviewer_id': reviewer_id,
            'items_reviewed': stats.get('items_reviewed', 0),
            'approval_rate': stats.get('approval_rate', 0),
            'correction_rate': stats.get('correction_rate', 0),
            'avg_review_time': stats.get('avg_review_time', 0),
            'items_per_hour': stats.get('items_per_hour', 0),
            'agreement_with_ai': stats.get('agreement_with_ai', 0)
        }

    def calculate_inter_rater_reliability(
        self,
        reviewer_ids: List[str]
    ) -> float:
        """
        Calculate inter-rater reliability (IRR) between reviewers

        Simplified Cohen's Kappa approximation

        Args:
            reviewer_ids: List of reviewer IDs to compare

        Returns:
            IRR score (0-1)
        """
        # Simplified placeholder
        # In production, implement proper Cohen's Kappa or Fleiss' Kappa
        return 0.85


# Singleton instance
_learner = None

def get_evidence_learner() -> EvidenceLearner:
    """Get singleton evidence learner instance"""
    global _learner
    if _learner is None:
        _learner = EvidenceLearner()
    return _learner
