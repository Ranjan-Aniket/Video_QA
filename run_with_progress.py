#!/Users/aranja14/Desktop/Gemini_QA/.venv/bin/python3
"""
Run pipeline with real-time progress updates
Uses smaller Whisper model (base) for speed
"""

import sys
import os
from pathlib import Path
import logging

# Setup
sys.path.insert(0, '/Users/aranja14/Desktop/Gemini_QA')
os.environ['PYTHONUNBUFFERED'] = '1'

# Load .env
from dotenv import load_dotenv
load_dotenv()

# Configure logging for immediate output
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()

# Force unbuffered
sys.stdout.reconfigure(line_buffering=True)

print("="*80)
print("🎬 RUNNING PIPELINE ON UPLOAD VIDEO")
print("="*80)

video_path = Path("/Users/aranja14/Desktop/Gemini_QA/uploads/Copy of w1cBUA1N2ds_20251120_053447.mp4")
output_dir = Path("/Users/aranja14/Desktop/Gemini_QA/outputs")

if not video_path.exists():
    print(f"❌ Video not found: {video_path}")
    sys.exit(1)

print(f"\n📹 Video: {video_path.name}")
print(f"   Size: {video_path.stat().st_size / 1024 / 1024:.1f} MB\n")

try:
    from processing.smart_pipeline import AdversarialSmartPipeline

    pipeline = AdversarialSmartPipeline(
        video_path=str(video_path),
        output_dir=str(output_dir),
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        claude_api_key=os.getenv('ANTHROPIC_API_KEY'),
        gemini_api_key=os.getenv('GEMINI_API_KEY'),
        enable_checkpoints=False,
        show_progress=True
    )

    print("🚀 Starting 9-phase pipeline...")
    print("   (This may take 8-12 minutes)\n")

    results = pipeline.run_full_pipeline()

    # Show results
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE!")
    print("="*80)
    print(f"Time: {results.get('processing_time_seconds', 0):.1f}s")
    print(f"Cost: ${results.get('total_cost', 0):.4f}")
    print(f"Questions: {results.get('metrics', {}).get('questions_generated', 0)}")

    # Load and display questions
    video_id = results.get('video_id', video_path.stem)
    questions_file = output_dir / f"{video_id}_phase8_questions.json"

    if questions_file.exists():
        import json
        with open(questions_file) as f:
            qdata = json.load(f)

        questions = qdata.get('questions', [])

        print("\n" + "="*80)
        print(f"📝 GENERATED {len(questions)} QUESTIONS FOR GEMINI TESTING")
        print("="*80)

        for i, q in enumerate(questions, 1):
            print(f"\n{'='*80}")
            print(f"QUESTION {i}:")
            print(f"{'='*80}")
            print(f"Q: {q.get('question_text', 'N/A')}")
            print(f"\nGolden Answer: {q.get('golden_answer', 'N/A')}")
            print(f"Type: {q.get('task_type', 'N/A')}")
            print(f"Complexity: {q.get('complexity', 'N/A')}")
            if q.get('visual_cues'):
                print(f"Visual Cues: {', '.join(q['visual_cues'][:3])}")
            if q.get('audio_cues'):
                print(f"Audio Cues: {q['audio_cues'][0] if q['audio_cues'] else 'N/A'}")

        print(f"\n\n📁 Full questions saved to:")
        print(f"   {questions_file}")

        # Save a simple text version for easy copy-paste
        text_file = output_dir / f"{video_id}_questions_for_gemini.txt"
        with open(text_file, 'w') as f:
            f.write("QUESTIONS FOR GEMINI 2.0 FLASH TESTING\n")
            f.write("="*80 + "\n\n")
            for i, q in enumerate(questions, 1):
                f.write(f"Q{i}: {q.get('question_text', 'N/A')}\n")
                f.write(f"Answer: {q.get('golden_answer', 'N/A')}\n")
                f.write(f"Type: {q.get('task_type', 'N/A')}\n\n")

        print(f"\n📄 Text version saved to:")
        print(f"   {text_file}")

except KeyboardInterrupt:
    print("\n\n⚠️  Interrupted by user")
    sys.exit(1)
except Exception as e:
    print(f"\n\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
