#!/Users/aranja14/Desktop/Gemini_QA/.venv/bin/python3
"""
Test new Phase 5 algorithmic prompt
Run Phase 5 only on existing video checkpoint
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()

# Force unbuffered
sys.stdout.reconfigure(line_buffering=True)

print("="*80)
print("🧪 TESTING NEW PHASE 5 ALGORITHMIC PROMPT")
print("="*80)

video_id = "Copy of w-A-4ckmFJo_20251120_135718"
output_dir = Path(f"/Users/aranja14/Desktop/Gemini_QA/outputs/video_20251120_135718_Copy of w-A-4ckmFJo")

if not output_dir.exists():
    print(f"❌ Output directory not found: {output_dir}")
    sys.exit(1)

# Check for Phase 4 checkpoint
phase4_file = output_dir / f"{video_id}_phase4_frame_budget.json"
if not phase4_file.exists():
    print(f"❌ Phase 4 checkpoint not found: {phase4_file}")
    sys.exit(1)

print(f"\n✅ Video ID: {video_id}")
print(f"✅ Output Dir: {output_dir}")
print(f"✅ Phase 4 checkpoint exists\n")

try:
    from processing.smart_pipeline import AdversarialSmartPipeline

    # Get video path from output directory name
    video_path = Path(f"/Users/aranja14/Desktop/Gemini_QA/uploads/{video_id}.mp4")

    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        sys.exit(1)

    pipeline = AdversarialSmartPipeline(
        video_path=str(video_path),
        output_dir=str(output_dir),  # Pass full subdirectory path
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        claude_api_key=os.getenv('ANTHROPIC_API_KEY'),
        gemini_api_key=os.getenv('GEMINI_API_KEY'),
        enable_checkpoints=False,
        show_progress=True
    )

    print("🚀 Running Phase 5 with new algorithmic prompt...")
    print("   (Pipeline will auto-resume from Phase 5 checkpoint)\n")

    # Load Phase 4 results to display budget
    import json
    with open(phase4_file) as f:
        phase4_data = json.load(f)

    print(f"📊 Phase 4 Budget: {phase4_data['recommended_frames']} frames")
    print(f"   Target Range: 80-120 frames\n")

    # Run pipeline - it will auto-resume from Phase 5
    # We'll stop after Phase 5 by manually checking
    print("="*80)
    results = pipeline.run_full_pipeline()
    print("="*80)

    # Check results
    phase5_file = output_dir / f"{video_id}_phase5_frame_selection.json"

    if phase5_file.exists():
        with open(phase5_file) as f:
            phase5_data = json.load(f)

        print("\n" + "="*80)
        print("✅ PHASE 5 COMPLETE!")
        print("="*80)

        exec_summary = phase5_data.get('execution_summary', {})
        frames_selected = phase5_data.get('frames_selected', [])
        dense_clusters = phase5_data.get('dense_clusters', [])
        coverage = phase5_data.get('coverage', {})

        print(f"\n📊 EXECUTION SUMMARY:")
        print(f"   Pass 1: {exec_summary.get('pass1_count', 0)} frames")
        print(f"   Pass 2 Executed: {exec_summary.get('pass2_executed', False)}")
        print(f"   Pass 2 Count: {exec_summary.get('pass2_count', 0)} frames")
        print(f"   Frames Added (Pass 2): {exec_summary.get('frames_added_pass2', 0)}")

        gap_analysis = exec_summary.get('gap_analysis', {})
        print(f"\n🔍 GAP ANALYSIS:")
        print(f"   Total Gaps: {gap_analysis.get('total_gaps_identified', 0)}")
        print(f"   Qualifying Gaps (>40s): {gap_analysis.get('qualifying_gaps_over_40s', 0)}")
        print(f"   Gaps Filled: {gap_analysis.get('gaps_filled', 0)}")

        print(f"\n📋 FRAME SELECTION:")
        print(f"   Total Frames: {len(frames_selected)}")
        print(f"   Target Range: 80-120 frames")
        print(f"   Status: {'✅ WITHIN RANGE' if 80 <= len(frames_selected) <= 120 else '❌ OUT OF RANGE'}")

        print(f"\n🎯 DENSE CLUSTERS:")
        print(f"   Count: {len(dense_clusters)}")
        total_cluster_frames = sum(len(c.get('frames', [])) for c in dense_clusters)
        print(f"   Total Frames in Clusters: {total_cluster_frames}")

        print(f"\n📈 QUESTION TYPE COVERAGE:")
        types_covered = coverage.get('types_covered', [])
        types_missing = coverage.get('types_missing', [])
        coverage_pct = coverage.get('coverage_percentage', 0)
        print(f"   Coverage: {coverage_pct:.1f}% ({len(types_covered)}/13 types)")
        print(f"   Covered: {', '.join(types_covered[:5])}...")
        if types_missing:
            print(f"   Missing: {', '.join(types_missing)}")

        # Check for generic filler reasoning
        generic_count = 0
        generic_keywords = ['temporal distribution', 'mid-section coverage', 'between highlights', 'gap filling']

        for frame in frames_selected:
            reasoning = frame.get('reasoning', '').lower()
            if any(keyword in reasoning for keyword in generic_keywords):
                generic_count += 1

        generic_pct = (generic_count / len(frames_selected) * 100) if frames_selected else 0
        print(f"\n⚠️  GENERIC FILLER REASONING:")
        print(f"   Frames: {generic_count}/{len(frames_selected)} ({generic_pct:.1f}%)")
        print(f"   Target: <10%")
        print(f"   Status: {'✅ GOOD' if generic_pct < 10 else '❌ TOO HIGH'}")

        # Priority distribution
        priorities = [f.get('priority', 0) for f in frames_selected]
        high_priority = sum(1 for p in priorities if p >= 0.85)
        medium_priority = sum(1 for p in priorities if 0.70 <= p < 0.85)
        low_priority = sum(1 for p in priorities if p < 0.70)

        print(f"\n📊 PRIORITY DISTRIBUTION:")
        print(f"   High (≥0.85): {high_priority} ({high_priority/len(frames_selected)*100:.1f}%)")
        print(f"   Medium (0.70-0.84): {medium_priority} ({medium_priority/len(frames_selected)*100:.1f}%)")
        print(f"   Low (<0.70): {low_priority} ({low_priority/len(frames_selected)*100:.1f}%)")

        # Validation
        print(f"\n✅ VALIDATION:")
        checks = []
        checks.append(("Frame Count (80-120)", 80 <= len(frames_selected) <= 120))
        checks.append(("Pass 2 Executed (if needed)", not (exec_summary.get('pass1_count', 0) < 80 and not exec_summary.get('pass2_executed', False))))
        checks.append(("Generic Filler <10%", generic_pct < 10))
        checks.append(("Coverage >85%", coverage_pct > 85))
        checks.append(("Dense Clusters ≤2", len(dense_clusters) <= 2))

        for check_name, passed in checks:
            status = "✅" if passed else "❌"
            print(f"   {status} {check_name}")

        all_passed = all(passed for _, passed in checks)

        print(f"\n{'='*80}")
        if all_passed:
            print("🎉 ALL CHECKS PASSED - NEW PROMPT IS WORKING!")
        else:
            print("⚠️  SOME CHECKS FAILED - REVIEW NEEDED")
        print("="*80)

        print(f"\n📁 Phase 5 output saved to:")
        print(f"   {phase5_file}")

    else:
        print(f"\n❌ Phase 5 output file not created: {phase5_file}")
        sys.exit(1)

except KeyboardInterrupt:
    print("\n\n⚠️  Interrupted by user")
    sys.exit(1)
except Exception as e:
    print(f"\n\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
