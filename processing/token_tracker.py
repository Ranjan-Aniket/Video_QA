"""
Token Tracking & Cost Analysis for Phase 8

Tracks input/output tokens and costs at both frame and question level.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


# Model pricing (per 1K tokens)
MODEL_PRICING = {
    "claude-3-5-haiku-20241022": {
        "input": 0.0008,
        "output": 0.004
    },
    "claude-sonnet-4-5-20250929": {
        "input": 0.003,
        "output": 0.015
    },
    "gpt-4o": {
        "input": 0.0025,
        "output": 0.010
    },
    "gpt-4o-2024-11-20": {
        "input": 0.0025,
        "output": 0.010
    }
}


@dataclass
class TokenUsage:
    """Track tokens and cost for a single API call"""
    model: str
    input_tokens: int
    output_tokens: int

    @property
    def input_cost(self) -> float:
        """Calculate input cost based on model"""
        rate = MODEL_PRICING.get(self.model, {"input": 0.003})["input"]
        return self.input_tokens * rate / 1000

    @property
    def output_cost(self) -> float:
        """Calculate output cost based on model"""
        rate = MODEL_PRICING.get(self.model, {"output": 0.015})["output"]
        return self.output_tokens * rate / 1000

    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class FrameTokenStats:
    """Token statistics for a single frame"""
    frame_id: str
    timestamp: float
    model: str
    input_tokens: int
    output_tokens: int
    questions_generated: int
    cost: float

    def to_dict(self) -> Dict:
        input_rate = MODEL_PRICING.get(self.model, {"input": 0.003})["input"]
        output_rate = MODEL_PRICING.get(self.model, {"output": 0.015})["output"]

        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "model": self.model,
            "tokens": {
                "input": self.input_tokens,
                "output": self.output_tokens,
                "total": self.input_tokens + self.output_tokens
            },
            "cost": {
                "input": round(self.input_tokens * input_rate / 1000, 6),
                "output": round(self.output_tokens * output_rate / 1000, 6),
                "total": round(self.cost, 6)
            },
            "questions_generated": self.questions_generated,
            "cost_per_question": round(self.cost / self.questions_generated, 6) if self.questions_generated > 0 else 0
        }


@dataclass
class QuestionTokenStats:
    """Token statistics for a single question"""
    question_id: str
    question_type: str
    frame_id: str
    model: str
    input_tokens_share: int  # Proportional share of frame's input
    output_tokens: int       # This question's output
    cost: float

    def to_dict(self) -> Dict:
        return {
            "question_id": self.question_id,
            "question_type": self.question_type,
            "frame_id": self.frame_id,
            "model": self.model,
            "tokens": {
                "input_share": self.input_tokens_share,
                "output": self.output_tokens,
                "total": self.input_tokens_share + self.output_tokens
            },
            "cost": round(self.cost, 6)
        }


class CostTracker:
    """Track costs across entire Phase 8"""

    def __init__(self):
        self.frame_stats: List[FrameTokenStats] = []
        self.question_stats: List[QuestionTokenStats] = []

    def add_frame(self, frame_stat: FrameTokenStats):
        """Add frame-level token stats"""
        self.frame_stats.append(frame_stat)
        logger.debug(f"Frame {frame_stat.frame_id}: {frame_stat.input_tokens}+{frame_stat.output_tokens} tokens, ${frame_stat.cost:.4f}")

    def add_question(self, question_stat: QuestionTokenStats):
        """Add question-level token stats"""
        self.question_stats.append(question_stat)

    @property
    def total_cost(self) -> float:
        return sum(f.cost for f in self.frame_stats)

    @property
    def total_input_tokens(self) -> int:
        return sum(f.input_tokens for f in self.frame_stats)

    @property
    def total_output_tokens(self) -> int:
        return sum(f.output_tokens for f in self.frame_stats)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    def get_summary(self) -> Dict:
        """Generate comprehensive cost summary"""
        model_breakdown = self._get_model_breakdown()

        summary = {
            "total_cost": round(self.total_cost, 4),
            "total_tokens": self.total_tokens,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "api_calls_made": len(self.frame_stats),  # ✅ FIX #10: Renamed from "frames_processed" (counts API calls, not actual frames)
            "questions_generated": len(self.question_stats),
            "avg_cost_per_api_call": round(self.total_cost / len(self.frame_stats), 4) if self.frame_stats else 0,  # ✅ FIX #10: Also renamed
            "avg_cost_per_question": round(self.total_cost / len(self.question_stats), 4) if self.question_stats else 0,
            "model_breakdown": model_breakdown
        }

        return summary

    def _get_model_breakdown(self) -> Dict:
        """Break down usage by model"""
        haiku_frames = [f for f in self.frame_stats if "haiku" in f.model.lower()]
        sonnet_frames = [f for f in self.frame_stats if "sonnet" in f.model.lower()]

        return {
            "haiku": {
                "frames": len(haiku_frames),
                "cost": round(sum(f.cost for f in haiku_frames), 4),
                "tokens": sum(f.input_tokens + f.output_tokens for f in haiku_frames),
                "questions": sum(f.questions_generated for f in haiku_frames)
            },
            "sonnet": {
                "frames": len(sonnet_frames),
                "cost": round(sum(f.cost for f in sonnet_frames), 4),
                "tokens": sum(f.input_tokens + f.output_tokens for f in sonnet_frames),
                "questions": sum(f.questions_generated for f in sonnet_frames)
            }
        }

    def print_summary(self):
        """Print formatted cost summary"""
        summary = self.get_summary()

        logger.info("=" * 80)
        logger.info("PHASE 8 COST SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Cost: ${summary['total_cost']:.4f}")
        logger.info(f"Total Tokens: {summary['total_tokens']:,} (Input: {summary['input_tokens']:,}, Output: {summary['output_tokens']:,})")
        logger.info(f"API Calls Made: {summary['api_calls_made']}")  # ✅ FIX #10: Updated
        logger.info(f"Questions Generated: {summary['questions_generated']}")
        logger.info(f"Avg Cost/API Call: ${summary['avg_cost_per_api_call']:.4f}")  # ✅ FIX #10: Updated
        logger.info(f"Avg Cost/Question: ${summary['avg_cost_per_question']:.4f}")

        logger.info("\nModel Breakdown:")
        for model, stats in summary['model_breakdown'].items():
            logger.info(f"  {model.upper()}:")
            logger.info(f"    Frames: {stats['frames']}")
            logger.info(f"    Questions: {stats['questions']}")
            logger.info(f"    Cost: ${stats['cost']:.4f}")
            logger.info(f"    Tokens: {stats['tokens']:,}")
        logger.info("=" * 80)

    def export_detailed_stats(self) -> Dict:
        """Export all frame and question stats for analysis"""
        return {
            "summary": self.get_summary(),
            "frames": [f.to_dict() for f in self.frame_stats],
            "questions": [q.to_dict() for q in self.question_stats]
        }
