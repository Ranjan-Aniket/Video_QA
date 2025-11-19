#!/usr/bin/env python3
"""
Create synthetic test evidence data to test HITL review workflow
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path

# Database path
db_path = Path(__file__).parent / "test.db"

# Test evidence data with varying consensus scenarios
test_evidence = [
    {
        "video_id": 1,
        "evidence_type": "audio_transcript",
        "timestamp_start": 15.5,
        "timestamp_end": 22.3,
        "gpt4_prediction": json.dumps({"answer": "yes", "confidence": 0.95}),
        "claude_prediction": json.dumps({"answer": "yes", "confidence": 0.93}),
        "open_model_prediction": json.dumps({"answer": "yes", "confidence": 0.91}),
        "confidence_score": 0.93,
        "needs_human_review": False,  # All agree, no review needed
        "priority_level": "low",
        "flag_reason": None
    },
    {
        "video_id": 1,
        "evidence_type": "visual_text",
        "timestamp_start": 45.0,
        "timestamp_end": 52.5,
        "gpt4_prediction": json.dumps({"answer": "no", "confidence": 0.88}),
        "claude_prediction": json.dumps({"answer": "yes", "confidence": 0.82}),
        "open_model_prediction": json.dumps({"answer": "no", "confidence": 0.79}),
        "confidence_score": 0.83,
        "needs_human_review": True,  # Disagreement on answer
        "priority_level": "high",
        "flag_reason": "disagreement_on_answer"
    },
    {
        "video_id": 1,
        "evidence_type": "visual_action",
        "timestamp_start": 78.2,
        "timestamp_end": 85.7,
        "gpt4_prediction": json.dumps({"answer": "yes", "confidence": 0.72}),
        "claude_prediction": json.dumps({"answer": "yes", "confidence": 0.68}),
        "open_model_prediction": json.dumps({"answer": "yes", "confidence": 0.65}),
        "confidence_score": 0.68,
        "needs_human_review": True,  # Low confidence
        "priority_level": "medium",
        "flag_reason": "low_confidence"
    },
    {
        "video_id": 1,
        "evidence_type": "audio_transcript",
        "timestamp_start": 120.5,
        "timestamp_end": 128.0,
        "gpt4_prediction": json.dumps({"answer": "no", "confidence": 0.97}),
        "claude_prediction": json.dumps({"answer": "no", "confidence": 0.96}),
        "open_model_prediction": json.dumps({"answer": "no", "confidence": 0.94}),
        "confidence_score": 0.96,
        "needs_human_review": False,  # High confidence, all agree
        "priority_level": "low",
        "flag_reason": None
    },
    {
        "video_id": 1,
        "evidence_type": "visual_text",
        "timestamp_start": 155.3,
        "timestamp_end": 162.8,
        "gpt4_prediction": json.dumps({"answer": "yes", "confidence": 0.91}),
        "claude_prediction": json.dumps({"answer": "no", "confidence": 0.87}),
        "open_model_prediction": json.dumps({"answer": "maybe", "confidence": 0.55}),
        "confidence_score": 0.78,
        "needs_human_review": True,  # 3-way disagreement
        "priority_level": "high",
        "flag_reason": "three_way_disagreement"
    }
]

def create_test_evidence():
    """Insert test evidence items into database"""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print(f"Connected to database: {db_path}")
    print(f"Creating {len(test_evidence)} test evidence items...")

    # Clear any existing evidence items
    cursor.execute("DELETE FROM evidence_items WHERE video_id = 1")
    print(f"Cleared existing evidence items for video_id=1")

    # Insert test evidence
    for i, evidence in enumerate(test_evidence, 1):
        cursor.execute("""
            INSERT INTO evidence_items (
                video_id, evidence_type, timestamp_start, timestamp_end,
                gpt4_prediction, claude_prediction, open_model_prediction,
                confidence_score, needs_human_review, priority_level, flag_reason,
                human_review_status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
        """, (
            evidence['video_id'],
            evidence['evidence_type'],
            evidence['timestamp_start'],
            evidence['timestamp_end'],
            evidence['gpt4_prediction'],
            evidence['claude_prediction'],
            evidence['open_model_prediction'],
            evidence['confidence_score'],
            1 if evidence['needs_human_review'] else 0,
            evidence['priority_level'],
            evidence['flag_reason'],
            datetime.now().isoformat()
        ))
        print(f"  {i}. Created {evidence['evidence_type']} evidence at {evidence['timestamp_start']:.1f}s - "
              f"Review: {evidence['needs_human_review']}, Priority: {evidence['priority_level']}")

    # Update video stats
    cursor.execute("""
        UPDATE videos
        SET evidence_extraction_status = 'completed',
            ai_evidence_count = ?,
            evidence_needs_review_count = (
                SELECT COUNT(*) FROM evidence_items
                WHERE video_id = 1 AND needs_human_review = 1
            ),
            evidence_approved_count = 0,
            evidence_accuracy_estimate = (
                SELECT AVG(confidence_score) FROM evidence_items WHERE video_id = 1
            )
        WHERE id = 1
    """, (len(test_evidence),))

    conn.commit()

    # Verify creation
    cursor.execute("SELECT COUNT(*) FROM evidence_items WHERE video_id = 1")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM evidence_items WHERE video_id = 1 AND needs_human_review = 1")
    needs_review = cursor.fetchone()[0]

    cursor.execute("SELECT evidence_extraction_status, ai_evidence_count, evidence_needs_review_count FROM videos WHERE id = 1")
    video_stats = cursor.fetchone()

    conn.close()

    print(f"\n✅ Successfully created test evidence data:")
    print(f"   Total evidence items: {total}")
    print(f"   Items needing review: {needs_review}")
    print(f"   Video stats: extraction={video_stats[0]}, count={video_stats[1]}, needs_review={video_stats[2]}")
    print(f"\nReady to test HITL review workflow!")

if __name__ == "__main__":
    create_test_evidence()
