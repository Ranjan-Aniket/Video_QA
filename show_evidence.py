"""
Show Evidence Results - Display actual evidence extracted from video
"""

import sys
import logging
from pathlib import Path
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    print("=" * 80)
    print("EVIDENCE EXTRACTION RESULTS")
    print("=" * 80)

    # Get test video
    test_videos = list(Path("uploads").glob("*.mp4"))
    if not test_videos:
        print("❌ No videos found")
        return 1

    test_video = test_videos[0]
    print(f"\n📹 Video: {test_video.name}")
    print(f"   Size: {test_video.stat().st_size / 1024 / 1024:.2f} MB")

    # Initialize evidence extractor
    from processing.evidence_extractor import EvidenceExtractor
    extractor = EvidenceExtractor(
        video_path=test_video,
        video_id=999
    )

    # Extract evidence (just 2 frames to see results)
    print("\n🔍 Extracting evidence from 2 frames...")
    print("-" * 80)

    evidence_items = extractor.extract_evidence_for_hitl(
        video_path=test_video,
        interval_seconds=10.0,  # Sample every 10 seconds
        max_items=2  # Get 2 frames
    )

    print("\n" + "=" * 80)
    print(f"✅ EXTRACTED {len(evidence_items)} EVIDENCE ITEMS")
    print("=" * 80)

    # Display results for each evidence item
    for i, item in enumerate(evidence_items, 1):
        print(f"\n{'='*80}")
        print(f"EVIDENCE ITEM #{i}")
        print(f"{'='*80}")
        print(f"Timestamp: {item.timestamp_start:.1f}s - {item.timestamp_end:.1f}s")
        print(f"Type: {item.evidence_type}")

        if item.ground_truth:
            print(f"\n📊 GROUND TRUTH (from 10 models):")
            print("-" * 80)

            gt = item.ground_truth

            # YOLOv8x - Object Detection
            if hasattr(gt, 'object_count') and gt.object_count:
                print(f"\n🎯 YOLOv8x Object Detection:")
                print(f"   Total Objects: {gt.object_count}")
                print(f"   Persons: {gt.person_count}")
                if hasattr(gt, 'yolov8x_objects') and gt.yolov8x_objects:
                    objects = {}
                    for obj in gt.yolov8x_objects[:10]:  # Show first 10
                        cls = obj.get('class', 'unknown')
                        objects[cls] = objects.get(cls, 0) + 1
                    for cls, count in sorted(objects.items(), key=lambda x: -x[1])[:5]:
                        print(f"   - {cls}: {count}")

            # CLIP - Clothing
            if hasattr(gt, 'clip_clothing') and gt.clip_clothing:
                print(f"\n👕 CLIP Clothing/Attributes:")
                for attr in gt.clip_clothing[:3]:
                    print(f"   - {attr}")

            # Places365 - Scene
            if hasattr(gt, 'places365_scene') and gt.places365_scene:
                scene = gt.places365_scene
                print(f"\n🏟️  Places365 Scene Classification:")
                print(f"   Scene: {scene.get('scene_category', 'N/A')}")
                print(f"   Indoor: {gt.is_indoor}")
                print(f"   Sports Venue: {gt.is_sports_venue}")

            # PaddleOCR - Text
            if hasattr(gt, 'paddleocr_text') and gt.paddleocr_text:
                print(f"\n📝 PaddleOCR Text Extraction:")
                for text in gt.paddleocr_text[:5]:
                    print(f"   - \"{text}\"")

            # BLIP-2 - Caption
            if hasattr(gt, 'image_caption') and gt.image_caption:
                print(f"\n🖼️  BLIP-2 Image Caption:")
                print(f"   {gt.image_caption[:150]}...")

            # Whisper - Audio
            if hasattr(gt, 'whisper_transcript') and gt.whisper_transcript:
                print(f"\n🎤 Whisper Audio Transcript:")
                print(f"   {gt.whisper_transcript[:150]}...")

            # DeepSport - Jerseys
            if hasattr(gt, 'player_numbers') and gt.player_numbers:
                print(f"\n🏀 DeepSport Jersey Numbers:")
                print(f"   Numbers detected: {gt.player_numbers}")

            # FER+ - Emotions
            if hasattr(gt, 'dominant_emotion') and gt.dominant_emotion:
                print(f"\n😊 FER+ Emotion Detection:")
                print(f"   Dominant emotion: {gt.dominant_emotion}")

        # Show AI predictions
        if item.gpt4_prediction or item.claude_prediction or item.open_model_prediction:
            print(f"\n🤖 AI PREDICTIONS:")
            print("-" * 80)
            if item.gpt4_prediction:
                print(f"GPT-4: {item.gpt4_prediction}")
            if item.claude_prediction:
                print(f"Claude: {item.claude_prediction}")
            if item.open_model_prediction:
                print(f"Open Model: {item.open_model_prediction}")

        # Show consensus
        if item.consensus:
            print(f"\n🎯 CONSENSUS:")
            print("-" * 80)
            cons = item.consensus
            if hasattr(cons, '__dict__'):
                for key, value in vars(cons).items():
                    if value is not None and not key.startswith('_'):
                        print(f"   {key}: {value}")

    print("\n" + "=" * 80)
    print("✅ EVIDENCE EXTRACTION COMPLETE")
    print("=" * 80)
    return 0

if __name__ == "__main__":
    sys.exit(main())
