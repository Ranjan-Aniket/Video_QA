"""
Test Configuration - Disable Expensive AI Calls

This configuration file allows you to easily enable/disable expensive AI API calls
for testing purposes, falling back to mock predictions.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class TestConfig:
    """Configuration for test mode"""

    # ===== PAID AI APIs (DISABLED FOR TESTING) =====
    enable_gpt4_vision: bool = False  # GPT-4 Vision API (~$0.01-0.05 per image) - DISABLED
    enable_claude_vision: bool = False  # Claude Sonnet 4.5 Vision API (~$0.01 per image) - DISABLED
    enable_gpt4_mini_generation: bool = False  # GPT-4-mini for questions (~$0.03 per question) - DISABLED

    # ===== MOCK RESPONSES FOR TESTING =====
    use_mock_predictions: bool = True  # Use deterministic mock data instead of API calls
    mock_consensus: str = "full_agreement"  # Options: "full_agreement", "majority", "disagreement"

    # ===== OPEN-SOURCE MODELS (ENABLED - FREE LOCAL) =====
    # These 10 models run locally and extract ground truth evidence
    enable_yolo: bool = True  # 1. YOLOv8x - Object detection (131MB, 90%)
    enable_clip: bool = True  # 2. CLIP ViT-L/14 - Clothing/attributes (1.7GB, 90%)
    enable_places365: bool = True  # 3. Places365-R152 - Scene classification (800MB, 92%)
    enable_ocr: bool = True  # 4. PaddleOCR - Text extraction (300MB, 92%)
    enable_videomae: bool = True  # 5. VideoMAE - Action recognition (1.2GB, 88%)
    enable_blip2: bool = True  # 6. BLIP-2 Flan-T5-XL - Contextual understanding (4GB, 85%)
    enable_whisper: bool = True  # 7. Whisper base - Audio transcription (150MB, 95%)
    enable_deepsport: bool = True  # 8. DeepSport - Jersey number OCR (300MB, 85%)
    enable_fer: bool = True  # 9. FER+ - Facial emotion detection (100MB, 80%)
    enable_scene_detection: bool = True  # 10. Auto-Orient + Tesseract - Text orientation

    # Cost Tracking
    track_api_costs: bool = True
    max_api_cost_per_video: float = 0.0  # Set to 0.0 to disable all paid APIs

    @classmethod
    def from_env(cls) -> 'TestConfig':
        """Load configuration from environment variables"""
        return cls(
            enable_gpt4_vision=os.getenv('ENABLE_GPT4_VISION', 'false').lower() == 'true',
            enable_claude_vision=os.getenv('ENABLE_CLAUDE_VISION', 'false').lower() == 'true',
            enable_gpt4_mini_generation=os.getenv('ENABLE_GPT4_MINI', 'false').lower() == 'true',
            use_mock_predictions=os.getenv('USE_MOCK_PREDICTIONS', 'true').lower() == 'true',
            max_api_cost_per_video=float(os.getenv('MAX_API_COST', '0.0'))
        )

    def is_testing_mode(self) -> bool:
        """Check if running in pure testing mode (no paid APIs)"""
        return (
            not self.enable_gpt4_vision
            and not self.enable_claude_vision
            and not self.enable_gpt4_mini_generation
        )

    def get_summary(self) -> str:
        """Get human-readable configuration summary"""
        status = "🧪 TEST MODE" if self.is_testing_mode() else "💰 PRODUCTION MODE"
        return f"""
{status}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Paid AI APIs:
  GPT-4 Vision:     {'❌ DISABLED' if not self.enable_gpt4_vision else '✅ ENABLED'}
  Claude Vision:    {'❌ DISABLED' if not self.enable_claude_vision else '✅ ENABLED'}
  GPT-4-mini Gen:   {'❌ DISABLED' if not self.enable_gpt4_mini_generation else '✅ ENABLED'}

Free Local Models (10 Open-Source Models):
  1. YOLOv8x:           {'✅ ENABLED' if self.enable_yolo else '❌ DISABLED'}
  2. CLIP ViT-L/14:     {'✅ ENABLED' if self.enable_clip else '❌ DISABLED'}
  3. Places365-R152:    {'✅ ENABLED' if self.enable_places365 else '❌ DISABLED'}
  4. PaddleOCR:         {'✅ ENABLED' if self.enable_ocr else '❌ DISABLED'}
  5. VideoMAE:          {'✅ ENABLED' if self.enable_videomae else '❌ DISABLED'}
  6. BLIP-2:            {'✅ ENABLED' if self.enable_blip2 else '❌ DISABLED'}
  7. Whisper:           {'✅ ENABLED' if self.enable_whisper else '❌ DISABLED'}
  8. DeepSport:         {'✅ ENABLED' if self.enable_deepsport else '❌ DISABLED'}
  9. FER+:              {'✅ ENABLED' if self.enable_fer else '❌ DISABLED'}
  10. Auto-Orient:      {'✅ ENABLED' if self.enable_scene_detection else '❌ DISABLED'}

Testing:
  Mock Predictions: {'✅ ENABLED' if self.use_mock_predictions else '❌ DISABLED'}
  Max API Cost:     ${self.max_api_cost_per_video:.2f} per video

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


# Global test configuration instance
TEST_CONFIG = TestConfig.from_env()


# Mock AI Predictions for Testing
class MockPredictions:
    """Mock AI predictions for testing without API calls"""

    @staticmethod
    def get_mock_gpt4_prediction(
        evidence_type: str,
        timestamp: float,
        consensus_type: str = "full_agreement"
    ) -> Optional[Dict[str, Any]]:
        """Get mock GPT-4 Vision prediction"""
        if not TEST_CONFIG.use_mock_predictions:
            return None

        base_predictions = {
            "full_agreement": {
                "answer": "yes",
                "confidence": 0.95,
                "reasoning": "GPT-4 Vision detected clear evidence"
            },
            "majority": {
                "answer": "yes",
                "confidence": 0.85,
                "reasoning": "GPT-4 Vision found evidence with medium confidence"
            },
            "disagreement": {
                "answer": "no",
                "confidence": 0.60,
                "reasoning": "GPT-4 Vision uncertain about evidence"
            }
        }

        return base_predictions.get(consensus_type, base_predictions["full_agreement"])

    @staticmethod
    def get_mock_claude_prediction(
        evidence_type: str,
        timestamp: float,
        consensus_type: str = "full_agreement"
    ) -> Optional[Dict[str, Any]]:
        """Get mock Claude Sonnet 4.5 prediction"""
        if not TEST_CONFIG.use_mock_predictions:
            return None

        base_predictions = {
            "full_agreement": {
                "answer": "yes",
                "confidence": 0.93,
                "reasoning": "Claude detected clear visual evidence"
            },
            "majority": {
                "answer": "yes",
                "confidence": 0.88,
                "reasoning": "Claude found partial evidence"
            },
            "disagreement": {
                "answer": "maybe",
                "confidence": 0.55,
                "reasoning": "Claude found ambiguous evidence"
            }
        }

        return base_predictions.get(consensus_type, base_predictions["full_agreement"])

    @staticmethod
    def get_mock_open_model_prediction(
        evidence_type: str,
        timestamp: float,
        consensus_type: str = "full_agreement"
    ) -> Optional[Dict[str, Any]]:
        """Get mock open model prediction (YOLO/OCR/Whisper combined)"""
        base_predictions = {
            "full_agreement": {
                "text": "yes",
                "confidence": 0.90,
                "detected_objects": ["person", "ball"],
                "ocr_text": "Player #77"
            },
            "majority": {
                "text": "yes",
                "confidence": 0.82,
                "detected_objects": ["person"],
                "ocr_text": "77"
            },
            "disagreement": {
                "text": "yes",
                "confidence": 0.70,
                "detected_objects": ["person"],
                "ocr_text": ""
            }
        }

        return base_predictions.get(consensus_type, base_predictions["full_agreement"])


# Usage Examples
if __name__ == "__main__":
    # Print current configuration
    print(TEST_CONFIG.get_summary())

    # Example: Get mock predictions
    if TEST_CONFIG.use_mock_predictions:
        gpt4_pred = MockPredictions.get_mock_gpt4_prediction(
            evidence_type="visual_text",
            timestamp=10.5,
            consensus_type=TEST_CONFIG.mock_consensus
        )
        print(f"\nMock GPT-4 Prediction: {gpt4_pred}")

        claude_pred = MockPredictions.get_mock_claude_prediction(
            evidence_type="visual_text",
            timestamp=10.5,
            consensus_type=TEST_CONFIG.mock_consensus
        )
        print(f"Mock Claude Prediction: {claude_pred}")
