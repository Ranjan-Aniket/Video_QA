"""
Cost tracking and monitoring
"""
from typing import Dict, Optional
from datetime import datetime
import json
from pathlib import Path
from .logger import cost_logger

class CostTracker:
    """Track costs per video and aggregate"""
    
    def __init__(self, log_file: str = "logs/cost.json"):
        self.log_file = log_file
        self.current_video_costs: Dict[str, float] = {}
        self.total_costs: Dict[str, float] = {}
        
        # Ensure log file exists
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        cost_logger.info(f"CostTracker initialized with log file: {log_file}")
    
    def start_video(self, video_id: str):
        """Start tracking for new video"""
        self.current_video_costs = {
            'video_id': video_id,
            'start_time': datetime.utcnow().isoformat(),
            'evidence_extraction_light': 0.0,
            'evidence_extraction_deep': 0.0,
            'qa_generation_templates': 0.0,
            'qa_generation_llama': 0.0,
            'qa_generation_gpt4mini': 0.0,
            'validation': 0.0,
            'gemini_flash': 0.0,
            'gemini_pro': 0.0,
            'explanation_generation': 0.0,
            'storage': 0.0,
            'total': 0.0
        }
        cost_logger.info(f"Started cost tracking for video: {video_id}")
    
    def add_cost(self, category: str, amount: float):
        """Add cost to current video"""
        if category in self.current_video_costs:
            self.current_video_costs[category] += amount
            self.current_video_costs['total'] += amount
            
            cost_logger.debug(
                f"Added cost: {category}=${amount:.4f}, "
                f"Total=${self.current_video_costs['total']:.4f}"
            )
        else:
            cost_logger.warning(f"Unknown cost category: {category}")
    
    def finish_video(self) -> Dict[str, float]:
        """Finish tracking and return summary"""
        self.current_video_costs['end_time'] = datetime.utcnow().isoformat()
        
        # Log to file
        self._log_to_file(self.current_video_costs)
        
        # Update totals
        for key, value in self.current_video_costs.items():
            if key not in ['video_id', 'start_time', 'end_time']:
                self.total_costs[key] = self.total_costs.get(key, 0.0) + value
        
        cost_logger.info(
            f"Finished video {self.current_video_costs['video_id']}: "
            f"Total cost=${self.current_video_costs['total']:.4f}"
        )
        
        return self.current_video_costs.copy()
    
    def get_summary(self) -> Dict[str, float]:
        """Get overall cost summary"""
        return self.total_costs.copy()
    
    def _log_to_file(self, costs: Dict):
        """Append costs to JSON log file"""
        try:
            # Read existing
            if Path(self.log_file).exists():
                with open(self.log_file, 'r') as f:
                    data = json.load(f)
            else:
                data = []
            
            # Append new
            data.append(costs)
            
            # Write back
            with open(self.log_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            cost_logger.debug(f"Logged costs to file: {self.log_file}")
        except Exception as e:
            cost_logger.error(f"Failed to log costs to file: {e}")

# Global cost tracker instance
cost_tracker = CostTracker()