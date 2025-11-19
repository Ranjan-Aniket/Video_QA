"""
Cost Optimizer - Processing Cost Monitor & Optimizer

Purpose: Monitor and optimize all processing costs to stay under $6/video budget
Compliance: Enforce budget constraints, optimize extraction strategy
Architecture: Monitors all processing operations and enforces limits

Cost Budget Breakdown (per video):
- Frame extraction: $0.50 (max 500 frames @ $0.001/frame)
- Audio processing: $0.30 (max 50 minutes @ $0.006/minute)
- OCR: $0.40 (max 400 frames if using EasyOCR free)
- Object detection: $1.00 (max 500 frames @ $0.002/frame)
- Scene detection: $0.20 (one-time per video)
- Total processing: $2.40 max
- Remaining for Q&A: $3.60

Revenue target: $8/video
Profit target: 58% margin ($4.64 profit)
"""

import logging
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class CostCategory(Enum):
    """Cost categories for tracking"""
    FRAME_EXTRACTION = "frame_extraction"
    AUDIO_PROCESSING = "audio_processing"
    OCR = "ocr"
    OBJECT_DETECTION = "object_detection"
    SCENE_DETECTION = "scene_detection"
    QA_GENERATION = "qa_generation"
    VALIDATION = "validation"
    GEMINI_TESTING = "gemini_testing"


@dataclass
class CostLimits:
    """Budget limits for each cost category"""
    frame_extraction_max: float = 0.50
    audio_processing_max: float = 0.30
    ocr_max: float = 0.40
    object_detection_max: float = 1.00
    scene_detection_max: float = 0.20
    qa_generation_max: float = 3.00
    validation_max: float = 0.30
    gemini_testing_max: float = 0.30
    
    @property
    def total_budget(self) -> float:
        """Total budget across all categories"""
        return (
            self.frame_extraction_max +
            self.audio_processing_max +
            self.ocr_max +
            self.object_detection_max +
            self.scene_detection_max +
            self.qa_generation_max +
            self.validation_max +
            self.gemini_testing_max
        )
    
    @property
    def processing_budget(self) -> float:
        """Budget for video processing only (excludes Q&A)"""
        return (
            self.frame_extraction_max +
            self.audio_processing_max +
            self.ocr_max +
            self.object_detection_max +
            self.scene_detection_max
        )


@dataclass
class CostTracking:
    """Track costs by category"""
    costs: Dict[CostCategory, float] = field(default_factory=dict)
    
    def add_cost(self, category: CostCategory, amount: float) -> None:
        """Add cost to category"""
        if category not in self.costs:
            self.costs[category] = 0.0
        self.costs[category] += amount
    
    def get_cost(self, category: CostCategory) -> float:
        """Get total cost for category"""
        return self.costs.get(category, 0.0)
    
    @property
    def total_cost(self) -> float:
        """Total cost across all categories"""
        return sum(self.costs.values())
    
    @property
    def processing_cost(self) -> float:
        """Total processing cost (excludes Q&A)"""
        processing_categories = [
            CostCategory.FRAME_EXTRACTION,
            CostCategory.AUDIO_PROCESSING,
            CostCategory.OCR,
            CostCategory.OBJECT_DETECTION,
            CostCategory.SCENE_DETECTION
        ]
        return sum(self.get_cost(cat) for cat in processing_categories)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {cat.value: cost for cat, cost in self.costs.items()}


@dataclass
class CostOptimizationResult:
    """Result of cost optimization check"""
    video_id: str
    current_costs: CostTracking
    limits: CostLimits
    is_within_budget: bool
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    @property
    def remaining_budget(self) -> float:
        """Remaining budget"""
        return self.limits.total_budget - self.current_costs.total_cost
    
    @property
    def budget_utilization(self) -> float:
        """Budget utilization percentage (0.0-1.0)"""
        if self.limits.total_budget == 0:
            return 0.0
        return self.current_costs.total_cost / self.limits.total_budget
    
    @property
    def profit_margin(self) -> float:
        """Estimated profit margin based on $8 revenue"""
        revenue = 8.0
        cost = self.current_costs.total_cost
        profit = revenue - cost
        return profit / revenue if revenue > 0 else 0.0


class CostOptimizer:
    """
    Monitor and optimize processing costs.
    
    Enforces budget constraints and provides optimization recommendations.
    """
    
    def __init__(
        self,
        limits: Optional[CostLimits] = None,
        strict_mode: bool = True,
        log_dir: Optional[Path] = None
    ):
        """
        Initialize cost optimizer.
        
        Args:
            limits: Budget limits (uses defaults if None)
            strict_mode: If True, raise errors when limits exceeded
            log_dir: Directory for cost logs
        """
        self.limits = limits or CostLimits()
        self.strict_mode = strict_mode
        self.log_dir = log_dir
        
        # Create log directory
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
        
        # Per-video tracking
        self.video_costs: Dict[str, CostTracking] = {}
        
        logger.info(
            f"CostOptimizer initialized (budget: ${self.limits.total_budget:.2f}, "
            f"strict: {strict_mode})"
        )
    
    def start_video_tracking(self, video_id: str) -> None:
        """Start tracking costs for a video"""
        self.video_costs[video_id] = CostTracking()
        logger.info(f"Started cost tracking for {video_id}")
    
    def add_cost(
        self,
        video_id: str,
        category: CostCategory,
        amount: float,
        description: Optional[str] = None
    ) -> None:
        """
        Add cost for a video.
        
        Args:
            video_id: Video identifier
            category: Cost category
            amount: Cost amount in dollars
            description: Optional description
        
        Raises:
            ValueError: If cost exceeds limit in strict mode
        """
        # Initialize tracking if not started
        if video_id not in self.video_costs:
            self.start_video_tracking(video_id)
        
        # Add cost
        self.video_costs[video_id].add_cost(category, amount)
        
        # Check limits
        current = self.video_costs[video_id].get_cost(category)
        limit = self._get_category_limit(category)
        
        desc_str = f" ({description})" if description else ""
        logger.debug(
            f"Added ${amount:.4f} to {category.value} for {video_id}{desc_str} "
            f"(total: ${current:.4f}/${limit:.2f})"
        )
        
        # Check if limit exceeded
        if current > limit:
            msg = (
                f"Cost limit exceeded for {category.value}: "
                f"${current:.4f} > ${limit:.2f}"
            )
            
            if self.strict_mode:
                logger.error(msg)
                raise ValueError(msg)
            else:
                logger.warning(msg)
        
        # Log cost
        if self.log_dir:
            self._log_cost(video_id, category, amount, description)
    
    def check_budget(self, video_id: str) -> CostOptimizationResult:
        """
        Check current budget status for a video.
        
        Args:
            video_id: Video identifier
        
        Returns:
            CostOptimizationResult with budget analysis
        """
        if video_id not in self.video_costs:
            logger.warning(f"No cost tracking for {video_id}, starting now")
            self.start_video_tracking(video_id)
        
        costs = self.video_costs[video_id]
        is_within = costs.total_cost <= self.limits.total_budget
        
        # Generate warnings
        warnings = []
        for category in CostCategory:
            current = costs.get_cost(category)
            limit = self._get_category_limit(category)
            
            if current > limit:
                warnings.append(
                    f"{category.value}: ${current:.4f} exceeds ${limit:.2f}"
                )
            elif current > limit * 0.9:  # 90% threshold
                warnings.append(
                    f"{category.value}: ${current:.4f} approaching limit ${limit:.2f}"
                )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(video_id, costs)
        
        result = CostOptimizationResult(
            video_id=video_id,
            current_costs=costs,
            limits=self.limits,
            is_within_budget=is_within,
            warnings=warnings,
            recommendations=recommendations
        )
        
        logger.info(
            f"Budget check for {video_id}: "
            f"${costs.total_cost:.2f}/${self.limits.total_budget:.2f} "
            f"({result.budget_utilization*100:.1f}% utilized, "
            f"{result.profit_margin*100:.1f}% margin)"
        )
        
        return result
    
    def get_remaining_budget(
        self,
        video_id: str,
        category: Optional[CostCategory] = None
    ) -> float:
        """
        Get remaining budget for video or category.
        
        Args:
            video_id: Video identifier
            category: Optional specific category
        
        Returns:
            Remaining budget in dollars
        """
        if video_id not in self.video_costs:
            if category:
                return self._get_category_limit(category)
            return self.limits.total_budget
        
        costs = self.video_costs[video_id]
        
        if category:
            current = costs.get_cost(category)
            limit = self._get_category_limit(category)
            return max(0, limit - current)
        else:
            return max(0, self.limits.total_budget - costs.total_cost)
    
    def can_afford(
        self,
        video_id: str,
        category: CostCategory,
        estimated_cost: float
    ) -> bool:
        """
        Check if we can afford an operation.
        
        Args:
            video_id: Video identifier
            category: Cost category
            estimated_cost: Estimated cost of operation
        
        Returns:
            True if within budget, False otherwise
        """
        remaining = self.get_remaining_budget(video_id, category)
        return estimated_cost <= remaining
    
    def optimize_extraction_strategy(
        self, video_id: str, num_questions: int
    ) -> Dict[str, Any]:
        """
        Optimize evidence extraction strategy based on budget.
        
        Args:
            video_id: Video identifier
            num_questions: Number of questions to generate
        
        Returns:
            Dictionary with optimization recommendations
        """
        remaining = self.get_remaining_budget(video_id)
        
        # Calculate per-question budget
        per_question_budget = remaining / num_questions if num_questions > 0 else 0
        
        # Determine extraction strategy
        if per_question_budget >= 0.15:
            strategy = "full"  # Extract all evidence types
            max_frames_per_question = 3
        elif per_question_budget >= 0.08:
            strategy = "selective"  # Extract only essential evidence
            max_frames_per_question = 2
        else:
            strategy = "minimal"  # Extract absolute minimum
            max_frames_per_question = 1
        
        optimization = {
            "strategy": strategy,
            "per_question_budget": per_question_budget,
            "max_frames_per_question": max_frames_per_question,
            "estimated_total_cost": per_question_budget * num_questions,
            "remaining_budget": remaining
        }
        
        logger.info(
            f"Extraction strategy for {video_id}: {strategy} "
            f"(${per_question_budget:.4f}/question)"
        )
        
        return optimization
    
    def _get_category_limit(self, category: CostCategory) -> float:
        """Get budget limit for category"""
        category_map = {
            CostCategory.FRAME_EXTRACTION: self.limits.frame_extraction_max,
            CostCategory.AUDIO_PROCESSING: self.limits.audio_processing_max,
            CostCategory.OCR: self.limits.ocr_max,
            CostCategory.OBJECT_DETECTION: self.limits.object_detection_max,
            CostCategory.SCENE_DETECTION: self.limits.scene_detection_max,
            CostCategory.QA_GENERATION: self.limits.qa_generation_max,
            CostCategory.VALIDATION: self.limits.validation_max,
            CostCategory.GEMINI_TESTING: self.limits.gemini_testing_max
        }
        return category_map.get(category, 0.0)
    
    def _generate_recommendations(
        self, video_id: str, costs: CostTracking
    ) -> List[str]:
        """Generate cost optimization recommendations"""
        recommendations = []
        
        # Check processing costs
        if costs.processing_cost > self.limits.processing_budget * 0.9:
            recommendations.append(
                "Processing costs high - use JIT extraction instead of bulk"
            )
        
        # Check frame extraction
        frame_cost = costs.get_cost(CostCategory.FRAME_EXTRACTION)
        if frame_cost > self.limits.frame_extraction_max * 0.8:
            recommendations.append(
                "Reduce frame extraction - extract only for specific questions"
            )
        
        # Check audio processing
        audio_cost = costs.get_cost(CostCategory.AUDIO_PROCESSING)
        if audio_cost > self.limits.audio_processing_max * 0.8:
            recommendations.append(
                "Reduce audio processing - transcribe only relevant segments"
            )
        
        # Check object detection
        obj_cost = costs.get_cost(CostCategory.OBJECT_DETECTION)
        if obj_cost > self.limits.object_detection_max * 0.8:
            recommendations.append(
                "Reduce object detection - run only when needed for questions"
            )
        
        # Check profit margin
        profit_margin = (8.0 - costs.total_cost) / 8.0
        if profit_margin < 0.58:  # Below 58% target
            recommendations.append(
                f"Profit margin {profit_margin*100:.1f}% below 58% target - "
                "optimize costs"
            )
        
        return recommendations
    
    def _log_cost(
        self,
        video_id: str,
        category: CostCategory,
        amount: float,
        description: Optional[str]
    ) -> None:
        """Log cost to file"""
        if not self.log_dir:
            return
        
        log_file = self.log_dir / f"{video_id}_costs.jsonl"
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "video_id": video_id,
            "category": category.value,
            "amount": amount,
            "description": description
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def export_cost_summary(self, video_id: str) -> Dict[str, Any]:
        """Export cost summary for a video"""
        if video_id not in self.video_costs:
            return {}
        
        result = self.check_budget(video_id)
        
        return {
            "video_id": video_id,
            "total_cost": result.current_costs.total_cost,
            "processing_cost": result.current_costs.processing_cost,
            "costs_by_category": result.current_costs.to_dict(),
            "budget_utilization": result.budget_utilization,
            "profit_margin": result.profit_margin,
            "is_within_budget": result.is_within_budget,
            "warnings": result.warnings,
            "recommendations": result.recommendations
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize cost optimizer
    optimizer = CostOptimizer(
        limits=CostLimits(),
        strict_mode=False,  # Warn instead of error
        log_dir=Path("./logs/costs")
    )
    
    # Track costs for a video
    video_id = "vid_abc123"
    optimizer.start_video_tracking(video_id)
    
    # Simulate processing costs
    optimizer.add_cost(
        video_id, CostCategory.SCENE_DETECTION, 0.21,
        "Initial scene detection"
    )
    
    optimizer.add_cost(
        video_id, CostCategory.FRAME_EXTRACTION, 0.15,
        "Extracted 150 frames"
    )
    
    optimizer.add_cost(
        video_id, CostCategory.AUDIO_PROCESSING, 0.12,
        "Transcribed 20 minutes"
    )
    
    optimizer.add_cost(
        video_id, CostCategory.OBJECT_DETECTION, 0.30,
        "Detected objects in 150 frames"
    )
    
    # Check budget
    result = optimizer.check_budget(video_id)
    
    print(f"✓ Budget check for {video_id}")
    print(f"  Total cost: ${result.current_costs.total_cost:.2f}")
    print(f"  Processing cost: ${result.current_costs.processing_cost:.2f}")
    print(f"  Budget utilization: {result.budget_utilization*100:.1f}%")
    print(f"  Profit margin: {result.profit_margin*100:.1f}%")
    print(f"  Within budget: {result.is_within_budget}")
    
    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"  ⚠ {warning}")
    
    if result.recommendations:
        print("\nRecommendations:")
        for rec in result.recommendations:
            print(f"  💡 {rec}")
    
    # Optimize extraction strategy
    optimization = optimizer.optimize_extraction_strategy(video_id, num_questions=30)
    print(f"\nExtraction strategy: {optimization['strategy']}")
    print(f"  Per-question budget: ${optimization['per_question_budget']:.4f}")
    print(f"  Max frames/question: {optimization['max_frames_per_question']}")
