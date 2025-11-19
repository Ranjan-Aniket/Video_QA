#!/usr/bin/env python3
"""
Debug script to check why Tier 1 templates aren't generating questions
"""

import sys
import json
from pathlib import Path

# Add project to path
sys.path.insert(0, '/Users/aranja14/Desktop/Gemini_QA')

from templates.registry import get_enhanced_registry
from templates.base import EvidenceDatabase

def load_evidence(video_id: str) -> EvidenceDatabase:
    """Load evidence from cache"""
    cache_file = Path(f"cache/{video_id}_evidence.json")
    
    if not cache_file.exists():
        print(f"❌ Evidence file not found: {cache_file}")
        return None
    
    with open(cache_file) as f:
        data = json.load(f)
    
    # Convert to EvidenceDatabase
    evidence = EvidenceDatabase(**data)
    return evidence

def debug_registry_generation():
    """Debug why templates aren't generating questions"""
    
    print("="*80)
    print("TIER 1 TEMPLATE GENERATION DEBUG")
    print("="*80)
    
    # Get the latest video ID from cache
    cache_dir = Path("cache")
    evidence_files = list(cache_dir.glob("*_evidence.json"))
    
    if not evidence_files:
        print("❌ No evidence files found in cache/")
        return
    
    # Get most recent
    latest_file = max(evidence_files, key=lambda p: p.stat().st_mtime)
    video_id = latest_file.stem.replace('_evidence', '')
    
    print(f"\n📁 Loading evidence from: {latest_file.name}")
    
    # Load evidence
    with open(latest_file) as f:
        evidence_data = json.load(f)
    
    print(f"\n📊 Evidence Summary:")
    print(f"  Video ID: {evidence_data.get('video_id', 'N/A')}")
    print(f"  Duration: {evidence_data.get('duration', 0):.1f}s")
    print(f"  Transcript segments: {len(evidence_data.get('transcript_segments', []))}")
    print(f"  Person detections: {len(evidence_data.get('person_detections', []))}")
    print(f"  Object detections: {len(evidence_data.get('object_detections', []))}")
    print(f"  Scene detections: {len(evidence_data.get('scene_detections', []))}")
    print(f"  Action detections: {len(evidence_data.get('action_detections', []))}")
    
    # Show sample transcript
    if evidence_data.get('transcript_segments'):
        print(f"\n  Sample transcript (first 3):")
        for seg in evidence_data['transcript_segments'][:3]:
            print(f"    {seg['start']:.1f}s: \"{seg['text']}\"")
    
    # Convert to EvidenceDatabase
    evidence = EvidenceDatabase(**evidence_data)
    
    # Get registry
    print(f"\n📋 Loading template registry...")
    registry = get_enhanced_registry()
    stats = registry.get_statistics()
    
    print(f"  Total templates: {stats['total_templates']}")
    print(f"  Single-type: {stats['single_type_templates']}")
    print(f"  Multi-type: {stats['multi_type_templates']}")
    
    # Check which templates can apply
    print(f"\n🔍 Checking template applicability...")
    
    applicable_count = 0
    for template in registry.all_templates:
        can_apply = template.can_apply(evidence)
        status = "✅" if can_apply else "❌"
        print(f"  {status} {template.name}: can_apply={can_apply}")
        if can_apply:
            applicable_count += 1
    
    print(f"\n  Total applicable: {applicable_count}/{stats['total_templates']}")
    
    # Try to generate questions
    print(f"\n🎯 Attempting generation...")
    
    try:
        questions = registry.generate_tier1_questions(
            evidence=evidence,
            target_count=5,
            prefer_multi_type=True
        )
        
        print(f"  Generated: {len(questions)} questions")
        
        if questions:
            print(f"\n  Sample questions:")
            for i, q in enumerate(questions[:3], 1):
                print(f"\n  Question {i}:")
                print(f"    Template: {q.template_name}")
                print(f"    Types: {[t.value for t in q.question_types]}")
                print(f"    Q: {q.question_text[:100]}...")
                print(f"    Audio cues: {len(q.audio_cues)}")
                print(f"    Visual cues: {len(q.visual_cues)}")
        else:
            print(f"\n  ⚠️ No questions generated!")
            print(f"\n  Debugging individual templates...")
            
            # Try each template individually
            for template in registry.all_templates:
                if template.can_apply(evidence):
                    print(f"\n  Trying {template.name}...")
                    try:
                        q = template.generate(evidence)
                        if q:
                            print(f"    ✅ Generated: {q.question_text[:80]}...")
                        else:
                            print(f"    ❌ Returned None")
                    except Exception as e:
                        print(f"    ❌ Error: {e}")
        
    except Exception as e:
        print(f"  ❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
    
    print("="*80)

if __name__ == "__main__":
    debug_registry_generation()
