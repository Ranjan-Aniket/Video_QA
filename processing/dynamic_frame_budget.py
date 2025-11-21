"""
Dynamic Frame Budget Calculator - Phase 4

Calculates optimal frame count based on:
- Video duration (10 frames/min)
- Highlights detected (2 frames/highlight)
- Question type requirements (min 43)
- Budget constraint (max 150, 95% of $3.36)
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class DynamicFrameBudget:
    """Calculate dynamic frame budget"""
    
    def __init__(
        self,
        total_budget: float = 3.36,
        fixed_costs: float = 0.20,
        cost_per_frame: float = 0.02
    ):
        self.total_budget = total_budget
        self.fixed_costs = fixed_costs
        self.cost_per_frame = cost_per_frame
        self.max_frames = int((total_budget - fixed_costs) / cost_per_frame * 0.95)  # 150
    
    def calculate_optimal_frames(
        self,
        video_duration: float,
        highlights_detected: int
    ) -> Dict:
        """
        Calculate optimal frame count.
        
        Returns:
            {
                'recommended_frames': int,
                'min_frames': 47,
                'max_frames': 150,
                'budget_used': float,
                'reasoning': dict
            }
        """
        # Method 1: By duration
        frames_by_duration = int((video_duration / 60) * 10)
        
        # Method 2: By highlights
        frames_by_highlights = highlights_detected * 2
        
        # Method 3: By types
        frames_by_types = 43 + int(video_duration / 30)
        
        # Take MAX
        recommended = max(frames_by_duration, frames_by_highlights, frames_by_types)
        
        # Cap at budget limit
        recommended = min(recommended, self.max_frames)
        
        # Never below 47
        recommended = max(recommended, 47)
        
        budget_used = self.fixed_costs + (recommended * self.cost_per_frame)
        
        return {
            'recommended_frames': recommended,
            'min_frames': 47,
            'max_frames': self.max_frames,
            'budget_used': budget_used,
            'budget_remaining': self.total_budget - budget_used,
            'reasoning': {
                'by_duration': frames_by_duration,
                'by_highlights': frames_by_highlights,
                'by_types': frames_by_types,
                'selected': recommended
            }
        }
