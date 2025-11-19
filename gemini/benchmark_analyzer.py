"""
Benchmark Analyzer - Analyze Gemini Testing Performance

Purpose: Aggregate testing results, generate performance reports
Compliance: Track success rates per question type, identify patterns
Architecture: Statistical analysis with export capabilities

Metrics Tracked:
- Overall success/failure rates
- Success rate per question type (temporal, inference, etc.)
- Hallucination patterns
- Cost analysis
- Difficulty scoring
"""

# Standard library imports
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import json
import csv

# Internal imports
from .adversarial_tester import TestResult, TestStatus
from .hallucination_detector import HallucinationResult, HallucinationType

logger = logging.getLogger(__name__)


@dataclass
class QuestionTypeStats:
    """Statistics for a question type"""
    question_type: str
    total_questions: int
    passed: int
    failed: int
    hallucinated: int
    
    @property
    def success_rate(self) -> float:
        """Success rate for this question type"""
        return self.passed / self.total_questions if self.total_questions > 0 else 0.0
    
    @property
    def failure_rate(self) -> float:
        """Failure rate (fail + hallucination)"""
        return (self.failed + self.hallucinated) / self.total_questions \
            if self.total_questions > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "question_type": self.question_type,
            "total_questions": self.total_questions,
            "passed": self.passed,
            "failed": self.failed,
            "hallucinated": self.hallucinated,
            "success_rate": self.success_rate,
            "failure_rate": self.failure_rate
        }


@dataclass
class BenchmarkMetrics:
    """Overall benchmark metrics"""
    # Basic counts
    total_questions: int
    total_videos: int
    
    # Outcomes
    passed: int
    failed: int
    hallucinated: int
    blocked: int
    errors: int
    
    # Rates
    success_rate: float
    failure_rate: float
    hallucination_rate: float
    
    # Per question type
    type_stats: Dict[str, QuestionTypeStats] = field(default_factory=dict)
    
    # Costs
    total_cost: float = 0.0
    avg_cost_per_question: float = 0.0
    avg_cost_per_video: float = 0.0
    
    # Hallucination analysis
    hallucination_types: Dict[str, int] = field(default_factory=dict)
    avg_hallucination_score: float = 0.0
    
    # Timing
    total_time: float = 0.0
    avg_time_per_question: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "total_questions": self.total_questions,
            "total_videos": self.total_videos,
            "passed": self.passed,
            "failed": self.failed,
            "hallucinated": self.hallucinated,
            "blocked": self.blocked,
            "errors": self.errors,
            "success_rate": self.success_rate,
            "failure_rate": self.failure_rate,
            "hallucination_rate": self.hallucination_rate,
            "total_cost": self.total_cost,
            "avg_cost_per_question": self.avg_cost_per_question,
            "avg_cost_per_video": self.avg_cost_per_video,
            "hallucination_types": self.hallucination_types,
            "avg_hallucination_score": self.avg_hallucination_score,
            "type_stats": {
                k: v.to_dict() for k, v in self.type_stats.items()
            }
        }


@dataclass
class PerformanceReport:
    """Complete performance report"""
    metrics: BenchmarkMetrics
    timestamp: str
    
    # Best/worst performing question types
    best_types: List[str] = field(default_factory=list)
    worst_types: List[str] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Summary
    summary: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp,
            "metrics": self.metrics.to_dict(),
            "best_types": self.best_types,
            "worst_types": self.worst_types,
            "recommendations": self.recommendations,
            "summary": self.summary
        }


class BenchmarkAnalyzer:
    """
    Analyze Gemini testing results and generate performance reports.
    
    Provides:
    - Aggregate statistics
    - Per-type performance analysis
    - Hallucination pattern identification
    - Cost analysis
    - Export to multiple formats
    """
    
    def __init__(self):
        """Initialize benchmark analyzer"""
        logger.info("BenchmarkAnalyzer initialized")
    
    def analyze_results(
        self,
        test_results: List[TestResult],
        question_types: Optional[Dict[str, str]] = None,
        num_videos: int = 1
    ) -> BenchmarkMetrics:
        """
        Analyze test results and generate metrics.
        
        Args:
            test_results: List of test results to analyze
            question_types: Optional dict mapping question_id to question_type
            num_videos: Number of videos tested
        
        Returns:
            BenchmarkMetrics with aggregate statistics
        """
        logger.info(f"Analyzing {len(test_results)} test results")
        
        if not test_results:
            return self._create_empty_metrics()
        
        # Basic counts
        total = len(test_results)
        passed = sum(1 for r in test_results if r.status == TestStatus.PASS)
        failed = sum(1 for r in test_results if r.status == TestStatus.FAIL)
        hallucinated = sum(1 for r in test_results if r.status == TestStatus.HALLUCINATION)
        blocked = sum(1 for r in test_results if r.status == TestStatus.BLOCKED)
        errors = sum(1 for r in test_results if r.status == TestStatus.ERROR)
        
        # Rates
        success_rate = passed / total
        failure_rate = (failed + hallucinated) / total
        hallucination_rate = hallucinated / total
        
        # Cost analysis
        total_cost = sum(r.test_cost for r in test_results)
        avg_cost_per_question = total_cost / total
        avg_cost_per_video = total_cost / num_videos if num_videos > 0 else 0.0
        
        # Timing
        total_time = sum(r.test_time for r in test_results)
        avg_time = total_time / total
        
        # Hallucination analysis
        hallucination_types_count = {}
        hallucination_scores = []
        
        for r in test_results:
            if r.hallucination and r.hallucination.has_hallucination:
                for h in r.hallucination.hallucinations:
                    type_name = h.type.value
                    hallucination_types_count[type_name] = \
                        hallucination_types_count.get(type_name, 0) + 1
                
                hallucination_scores.append(r.hallucination.score)
        
        avg_hallucination_score = (
            sum(hallucination_scores) / len(hallucination_scores)
            if hallucination_scores else 0.0
        )
        
        # Per-type analysis
        type_stats = {}
        if question_types:
            type_stats = self._analyze_by_type(test_results, question_types)
        
        # Create metrics
        metrics = BenchmarkMetrics(
            total_questions=total,
            total_videos=num_videos,
            passed=passed,
            failed=failed,
            hallucinated=hallucinated,
            blocked=blocked,
            errors=errors,
            success_rate=success_rate,
            failure_rate=failure_rate,
            hallucination_rate=hallucination_rate,
            type_stats=type_stats,
            total_cost=total_cost,
            avg_cost_per_question=avg_cost_per_question,
            avg_cost_per_video=avg_cost_per_video,
            hallucination_types=hallucination_types_count,
            avg_hallucination_score=avg_hallucination_score,
            total_time=total_time,
            avg_time_per_question=avg_time
        )
        
        logger.info(
            f"Analysis complete: {success_rate*100:.1f}% success, "
            f"{hallucination_rate*100:.1f}% hallucination, "
            f"${total_cost:.4f} total cost"
        )
        
        return metrics
    
    def generate_report(
        self,
        metrics: BenchmarkMetrics,
        include_recommendations: bool = True
    ) -> PerformanceReport:
        """
        Generate comprehensive performance report.
        
        Args:
            metrics: Benchmark metrics
            include_recommendations: Whether to include recommendations
        
        Returns:
            PerformanceReport with analysis and recommendations
        """
        logger.info("Generating performance report")
        
        # Identify best/worst performing types
        best_types, worst_types = self._identify_best_worst_types(metrics)
        
        # Generate recommendations
        recommendations = []
        if include_recommendations:
            recommendations = self._generate_recommendations(metrics)
        
        # Generate summary
        summary = self._generate_summary(metrics)
        
        # Create report
        report = PerformanceReport(
            metrics=metrics,
            timestamp=datetime.now().isoformat(),
            best_types=best_types,
            worst_types=worst_types,
            recommendations=recommendations,
            summary=summary
        )
        
        return report
    
    def export_to_json(
        self, report: PerformanceReport, output_path: Path
    ) -> None:
        """Export report to JSON"""
        with open(output_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        logger.info(f"Report exported to {output_path}")
    
    def export_to_csv(
        self,
        test_results: List[TestResult],
        output_path: Path
    ) -> None:
        """Export detailed results to CSV"""
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'question_id',
                'question',
                'golden_answer',
                'gemini_answer',
                'status',
                'is_correct',
                'similarity_score',
                'hallucination_score',
                'test_cost',
                'test_time'
            ])
            
            writer.writeheader()
            
            for r in test_results:
                writer.writerow({
                    'question_id': r.question_id,
                    'question': r.question,
                    'golden_answer': r.golden_answer,
                    'gemini_answer': r.gemini_answer,
                    'status': r.status.value,
                    'is_correct': r.is_correct,
                    'similarity_score': r.similarity_score,
                    'hallucination_score': (
                        r.hallucination.score if r.hallucination else 0.0
                    ),
                    'test_cost': r.test_cost,
                    'test_time': r.test_time
                })
        
        logger.info(f"Detailed results exported to {output_path}")
    
    def _analyze_by_type(
        self,
        test_results: List[TestResult],
        question_types: Dict[str, str]
    ) -> Dict[str, QuestionTypeStats]:
        """Analyze results by question type"""
        # Group by type
        type_results = {}
        
        for r in test_results:
            qtype = question_types.get(r.question_id, "unknown")
            
            if qtype not in type_results:
                type_results[qtype] = []
            
            type_results[qtype].append(r)
        
        # Calculate stats per type
        type_stats = {}
        
        for qtype, results in type_results.items():
            passed = sum(1 for r in results if r.status == TestStatus.PASS)
            failed = sum(1 for r in results if r.status == TestStatus.FAIL)
            hallucinated = sum(1 for r in results if r.status == TestStatus.HALLUCINATION)
            
            type_stats[qtype] = QuestionTypeStats(
                question_type=qtype,
                total_questions=len(results),
                passed=passed,
                failed=failed,
                hallucinated=hallucinated
            )
        
        return type_stats
    
    def _identify_best_worst_types(
        self, metrics: BenchmarkMetrics
    ) -> tuple[List[str], List[str]]:
        """Identify best and worst performing question types"""
        if not metrics.type_stats:
            return [], []
        
        # Sort by failure rate (lower is better)
        sorted_types = sorted(
            metrics.type_stats.items(),
            key=lambda x: x[1].failure_rate
        )
        
        # Top 3 best and worst
        best = [t[0] for t in sorted_types[:3]]
        worst = [t[0] for t in sorted_types[-3:]]
        
        return best, worst
    
    def _generate_recommendations(
        self, metrics: BenchmarkMetrics
    ) -> List[str]:
        """Generate recommendations based on metrics"""
        recommendations = []
        
        # Success rate recommendations
        if metrics.success_rate > 0.7:
            recommendations.append(
                "Questions are too easy - Gemini succeeds >70%. "
                "Generate harder adversarial questions."
            )
        elif metrics.success_rate < 0.3:
            recommendations.append(
                "Excellent adversarial performance - Gemini fails >70%. "
                "These questions are highly effective."
            )
        
        # Hallucination rate
        if metrics.hallucination_rate > 0.4:
            recommendations.append(
                f"High hallucination rate ({metrics.hallucination_rate*100:.1f}%). "
                "Questions effectively expose Gemini weaknesses."
            )
        
        # Cost optimization
        if metrics.avg_cost_per_video > 0.50:
            recommendations.append(
                f"Cost per video (${metrics.avg_cost_per_video:.2f}) exceeds target ($0.30). "
                "Consider reducing number of questions or using Flash model."
            )
        
        # Question type specific
        if metrics.type_stats:
            for qtype, stats in metrics.type_stats.items():
                if stats.failure_rate > 0.8:
                    recommendations.append(
                        f"{qtype} questions are highly effective "
                        f"({stats.failure_rate*100:.1f}% failure rate). "
                        "Generate more of this type."
                    )
        
        return recommendations
    
    def _generate_summary(self, metrics: BenchmarkMetrics) -> str:
        """Generate text summary"""
        summary_parts = [
            f"Tested {metrics.total_questions} questions across {metrics.total_videos} video(s).",
            f"Gemini success rate: {metrics.success_rate*100:.1f}%",
            f"Hallucination rate: {metrics.hallucination_rate*100:.1f}%",
            f"Total cost: ${metrics.total_cost:.4f}",
        ]
        
        if metrics.type_stats:
            # Best performing type
            best_type = min(
                metrics.type_stats.items(),
                key=lambda x: x[1].failure_rate
            )
            summary_parts.append(
                f"Best adversarial type: {best_type[0]} "
                f"({best_type[1].failure_rate*100:.1f}% failure)"
            )
        
        return " ".join(summary_parts)
    
    def _create_empty_metrics(self) -> BenchmarkMetrics:
        """Create empty metrics"""
        return BenchmarkMetrics(
            total_questions=0,
            total_videos=0,
            passed=0,
            failed=0,
            hallucinated=0,
            blocked=0,
            errors=0,
            success_rate=0.0,
            failure_rate=0.0,
            hallucination_rate=0.0
        )


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example test results (normally from AdversarialTester)
    from .adversarial_tester import TestResult, TestStatus
    
    test_results = [
        TestResult(
            question_id="q1",
            question="What happens first?",
            golden_answer="Person enters",
            gemini_answer="Person exits",
            status=TestStatus.FAIL,
            is_correct=False,
            similarity_score=0.2,
            test_cost=0.01,
            test_time=1.5
        ),
        TestResult(
            question_id="q2",
            question="What color is the shirt?",
            golden_answer="Red",
            gemini_answer="Red",
            status=TestStatus.PASS,
            is_correct=True,
            similarity_score=1.0,
            test_cost=0.008,
            test_time=1.2
        ),
    ]
    
    # Map question types
    question_types = {
        "q1": "temporal",
        "q2": "inference"
    }
    
    # Analyze
    analyzer = BenchmarkAnalyzer()
    metrics = analyzer.analyze_results(test_results, question_types, num_videos=1)
    
    print(f"✅ Analysis complete:")
    print(f"   Total questions: {metrics.total_questions}")
    print(f"   Success rate: {metrics.success_rate*100:.1f}%")
    print(f"   Hallucination rate: {metrics.hallucination_rate*100:.1f}%")
    print(f"   Total cost: ${metrics.total_cost:.4f}")
    
    # Generate report
    report = analyzer.generate_report(metrics)
    
    print(f"\n✅ Report generated:")
    print(f"   {report.summary}")
    print(f"\nRecommendations:")
    for rec in report.recommendations:
        print(f"   - {rec}")
    
    # Export
    # analyzer.export_to_json(report, Path("report.json"))
    # analyzer.export_to_csv(test_results, Path("results.csv"))
