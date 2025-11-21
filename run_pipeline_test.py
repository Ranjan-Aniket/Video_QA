#!/Users/aranja14/Desktop/Gemini_QA/.venv/bin/python3
"""
Run pipeline on upload video with progress tracking
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Loaded .env file")
except ImportError:
    print("⚠️  python-dotenv not installed, trying manual .env load...")
    # Manual .env loading
    env_file = project_root / '.env'
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")
        print("✓ Manually loaded .env file")

# Check API keys
print("\n" + "="*80)
print("API KEYS STATUS")
print("="*80)
openai_key = os.getenv('OPENAI_API_KEY')
claude_key = os.getenv('ANTHROPIC_API_KEY') or os.getenv('CLAUDE_API_KEY')
gemini_key = os.getenv('GEMINI_API_KEY')

print(f"OpenAI: {'✓ SET' if openai_key else '✗ NOT SET'}")
print(f"Anthropic: {'✓ SET' if claude_key else '✗ NOT SET'}")
print(f"Gemini: {'✓ SET' if gemini_key else '✗ NOT SET'}")
print("="*80 + "\n")

# Find video in uploads
video_path = project_root / "uploads" / "Copy of w1cBUA1N2ds_20251120_053447.mp4"

if not video_path.exists():
    print(f"❌ Video not found: {video_path}")
    sys.exit(1)

print(f"📹 Video: {video_path.name}")
print(f"   Size: {video_path.stat().st_size / 1024 / 1024:.1f} MB")
print()

# Import and run pipeline
print("="*80)
print("STARTING 9-PHASE PIPELINE")
print("="*80)
print()

try:
    from processing.smart_pipeline import AdversarialSmartPipeline

    pipeline = AdversarialSmartPipeline(
        video_path=str(video_path),
        output_dir=str(project_root / "outputs"),
        openai_api_key=openai_key,
        claude_api_key=claude_key,
        gemini_api_key=gemini_key,
        enable_checkpoints=False,
        show_progress=True
    )

    # Run pipeline
    results = pipeline.run_full_pipeline()

    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE")
    print("="*80)
    print(f"Processing time: {results.get('processing_time_seconds', 0):.1f}s")
    print(f"Total cost: ${results.get('total_cost', 0):.4f}")
    print(f"Questions generated: {results.get('metrics', {}).get('questions_generated', 0)}")
    print()

    # Show questions
    questions_file = project_root / "outputs" / f"{results.get('video_id', 'unknown')}_phase8_questions.json"
    if questions_file.exists():
        import json
        with open(questions_file) as f:
            questions_data = json.load(f)

        print("="*80)
        print("📝 GENERATED QUESTIONS")
        print("="*80)

        questions = questions_data.get('questions', [])
        for i, q in enumerate(questions[:10], 1):  # Show first 10
            print(f"\nQ{i}: {q.get('question_text', 'N/A')}")
            print(f"   Answer: {q.get('golden_answer', 'N/A')}")
            print(f"   Type: {q.get('task_type', 'N/A')}")

        if len(questions) > 10:
            print(f"\n... and {len(questions) - 10} more questions")

        print(f"\n✓ Full questions saved to: {questions_file.name}")

except KeyboardInterrupt:
    print("\n\n⚠️  Pipeline interrupted by user")
    sys.exit(1)
except Exception as e:
    print(f"\n\n❌ Pipeline failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
