"""
Regenerate Questions with Proper Timestamps from Audio Analysis

This script fixes questions that have placeholder timestamps by matching
audio cues to the actual audio analysis transcript and updating timestamps.
"""

import json
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional

def load_json(filepath: Path) -> dict:
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def save_json(data: dict, filepath: Path):
    """Save JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def find_text_in_transcript(audio_cue: str, segments: List[dict]) -> Optional[Tuple[float, float]]:
    """
    Find audio cue text in transcript segments and return timestamps

    Returns:
        Tuple of (start_timestamp, end_timestamp) or None if not found
    """
    # Clean the audio cue for matching
    audio_cue_clean = audio_cue.lower().strip().strip('"').strip("'")

    # Try exact match first
    for segment in segments:
        segment_text = segment['text'].lower().strip()
        if audio_cue_clean in segment_text or segment_text in audio_cue_clean:
            return (segment['start'], segment['end'])

    # Try fuzzy match (split into words and find best match)
    audio_cue_words = set(audio_cue_clean.split())
    best_match = None
    best_score = 0

    for segment in segments:
        segment_words = set(segment['text'].lower().split())
        # Calculate overlap score
        overlap = len(audio_cue_words & segment_words)
        total = len(audio_cue_words | segment_words)
        score = overlap / total if total > 0 else 0

        if score > best_score and score > 0.5:  # At least 50% match
            best_score = score
            best_match = segment

    if best_match:
        return (best_match['start'], best_match['end'])

    return None

def seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def regenerate_questions_with_timestamps(video_id: str, output_dir: Path):
    """
    Regenerate questions JSON with proper timestamps from audio analysis

    Args:
        video_id: Video ID to process
        output_dir: Path to outputs directory for this video
    """
    print(f"\n🔄 Regenerating questions with timestamps for: {video_id}")
    print("="*80)

    # Extract original filename and timestamp from video_id
    # Pattern: video_TIMESTAMP_ORIGINALNAME -> ORIGINALNAME_TIMESTAMP
    if video_id.startswith("video_"):
        parts = video_id.replace("video_", "", 1).split("_", 2)
        if len(parts) >= 3:
            date_part = parts[0]  # 20251118
            time_part = parts[1]  # 133434
            original_name = parts[2]  # Copy of VOmj6qaznos
            base_name = f"{original_name}_{date_part}_{time_part}"
        else:
            base_name = video_id
    else:
        base_name = video_id

    # Load files
    audio_file = output_dir / f"{base_name}_audio_analysis.json"
    questions_file = output_dir / f"{base_name}_questions.json"

    if not audio_file.exists():
        print(f"❌ Audio analysis file not found: {audio_file}")
        return False

    if not questions_file.exists():
        print(f"❌ Questions file not found: {questions_file}")
        return False

    print(f"📂 Loading audio analysis from: {audio_file}")
    audio_data = load_json(audio_file)
    segments = audio_data.get('segments', [])

    print(f"📂 Loading questions from: {questions_file}")
    questions_data = load_json(questions_file)
    questions = questions_data.get('questions', [])

    print(f"\n📊 Found {len(segments)} audio segments")
    print(f"📊 Found {len(questions)} questions to update")

    # Update each question with proper timestamps
    updated_count = 0
    failed_count = 0

    for i, question in enumerate(questions, 1):
        audio_cue = question.get('audio_cue', '')

        if not audio_cue:
            print(f"⚠️  Question {i}: No audio cue found, skipping...")
            continue

        # Find timestamps for this audio cue
        timestamps = find_text_in_transcript(audio_cue, segments)

        if timestamps:
            start_sec, end_sec = timestamps
            start_ts = seconds_to_timestamp(start_sec)
            end_ts = seconds_to_timestamp(end_sec)

            # Update question
            question['start_timestamp'] = start_ts
            question['end_timestamp'] = end_ts

            print(f"✅ Question {i}: Updated timestamps to {start_ts} - {end_ts}")
            print(f"   Audio cue: \"{audio_cue[:60]}...\"" if len(audio_cue) > 60 else f"   Audio cue: \"{audio_cue}\"")
            updated_count += 1
        else:
            print(f"⚠️  Question {i}: Could not find audio cue in transcript")
            print(f"   Audio cue: \"{audio_cue[:60]}...\"" if len(audio_cue) > 60 else f"   Audio cue: \"{audio_cue}\"")
            failed_count += 1

    # Save updated questions
    backup_file = questions_file.with_suffix('.json.backup')
    print(f"\n💾 Creating backup: {backup_file}")
    save_json(questions_data, backup_file)

    print(f"💾 Saving updated questions: {questions_file}")
    save_json(questions_data, questions_file)

    print(f"\n✅ Complete!")
    print(f"   Updated: {updated_count}/{len(questions)}")
    print(f"   Failed: {failed_count}/{len(questions)}")
    print(f"   Backup saved to: {backup_file}")

    return True

if __name__ == "__main__":
    # Configuration
    video_id = "video_20251118_133434_Copy of VOmj6qaznos"
    output_base = Path("/Users/aranja14/Desktop/Gemini_QA/outputs")
    output_dir = output_base / video_id

    # Run regeneration
    success = regenerate_questions_with_timestamps(video_id, output_dir)

    if success:
        print("\n🎉 Questions successfully regenerated with proper timestamps!")
    else:
        print("\n❌ Failed to regenerate questions")
