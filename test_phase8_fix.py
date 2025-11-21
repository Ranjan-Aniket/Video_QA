#!/usr/bin/env python3
"""
Test script to verify Phase 8 question generation fix
"""

import os
import sys
from pathlib import Path

# Set API keys
os.environ["OPENAI_API_KEY"] = "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"
os.environ["CLAUDE_API_KEY"] = "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"
os.environ["ANTHROPIC_API_KEY"] = "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"

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import logging
from processing.smart_pipeline import AdversarialSmartPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_phase8_fix():
    """Test the Phase 8 fix"""

    # Find the video file
    video_path = "/Users/aranja14/Desktop/Gemini_QA/uploads/Copy of w-A-4ckmFJo_20251120_135718.mp4"
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        return

    print(f"🎬 Testing Phase 8 fix with video: {Path(video_path).name}")

    # Initialize pipeline
    pipeline = AdversarialSmartPipeline(
        video_path=video_path,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        claude_api_key=os.getenv("CLAUDE_API_KEY")
    )

    try:
        # Run pipeline (should resume from Phase 7 since we deleted those checkpoints)
        results = pipeline.run_full_pipeline()

        print("\n" + "=" * 60)
        print("✅ PIPELINE TEST RESULTS")
        print("=" * 60)
        print(f"Questions generated: {results['metrics']['questions_generated']}")
        print(f"Total cost: ${results['total_cost']:.4f}")
        print(f"Processing time: {results['processing_time_seconds']:.1f}s")

        # Check Phase 8 results specifically
        output_dir = Path(pipeline.output_dir)
        phase8_file = output_dir / f"{pipeline.video_id}_phase8_questions.json"
        if phase8_file.exists():
            import json
            with open(phase8_file, 'r') as f:
                phase8_data = json.load(f)
            print(f"Phase 8 questions: {phase8_data.get('total_questions', 0)}")
            print(f"Premium frames available: {phase8_data.get('metadata', {}).get('premium_frames_available', 0)}")
        else:
            print("❌ Phase 8 file not found")

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_phase8_fix()