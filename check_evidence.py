#!/usr/bin/env python3
"""Quick check of evidence file"""
import json
from pathlib import Path

cache = Path("cache")
evidence_files = list(cache.glob("*_evidence.json"))

if not evidence_files:
    print("❌ No evidence files found!")
    exit(1)

latest = max(evidence_files, key=lambda p: p.stat().st_mtime)
print(f"📁 Checking: {latest.name}\n")

with open(latest) as f:
    data = json.load(f)

print(f"Evidence Contents:")
print(f"  video_id: {data.get('video_id', 'MISSING')}")
print(f"  duration: {data.get('duration', 0):.1f}s")
print(f"\nData counts:")
print(f"  transcript_segments: {len(data.get('transcript_segments', []))}")
print(f"  person_detections: {len(data.get('person_detections', []))}")
print(f"  object_detections: {len(data.get('object_detections', []))}")
print(f"  scene_detections: {len(data.get('scene_detections', []))}")
print(f"  action_detections: {len(data.get('action_detections', []))}")
print(f"  music_segments: {len(data.get('music_segments', []))}")
print(f"  sound_effects: {len(data.get('sound_effects', []))}")

if data.get('transcript_segments'):
    print(f"\n📝 Sample transcript:")
    for seg in data['transcript_segments'][:5]:
        print(f"  {seg.get('start', 0):.1f}s: \"{seg.get('text', '')}\"")
else:
    print(f"\n⚠️ NO TRANSCRIPT DATA - This is the problem!")

if not any([
    data.get('transcript_segments'),
    data.get('person_detections'),
    data.get('object_detections')
]):
    print(f"\n❌ EVIDENCE IS EMPTY!")
    print(f"   Templates can't generate without evidence.")
    print(f"   Fix evidence extraction first!")
