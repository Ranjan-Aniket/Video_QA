#!/usr/bin/env python3
"""Fix evidence_operations.py SQLAlchemy parameter issues"""

import re

# Read the file
with open('/Users/aranja14/Desktop/Gemini_QA/database/evidence_operations.py', 'r') as f:
    content = f.read()

# Fix 1: create_evidence_item method
content = content.replace(
    '''            result = session.execute(
                text("""
                INSERT INTO evidence_items (
                    video_id, evidence_type, timestamp_start, timestamp_end,
                    gpt4_prediction, claude_prediction, open_model_prediction,
                    confidence_score, needs_human_review, priority_level, flag_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """),
                (video_id, evidence_type, timestamp_start, timestamp_end,
                 gpt4_json, claude_json, open_json,
                 confidence_score, needs_review, priority, flag_reason)
            )''',
    '''            result = session.execute(
                text("""
                INSERT INTO evidence_items (
                    video_id, evidence_type, timestamp_start, timestamp_end,
                    gpt4_prediction, claude_prediction, open_model_prediction,
                    confidence_score, needs_human_review, priority_level, flag_reason
                ) VALUES (:video_id, :evidence_type, :timestamp_start, :timestamp_end,
                         :gpt4_prediction, :claude_prediction, :open_model_prediction,
                         :confidence_score, :needs_review, :priority, :flag_reason)
                """),
                {
                    'video_id': video_id,
                    'evidence_type': evidence_type,
                    'timestamp_start': timestamp_start,
                    'timestamp_end': timestamp_end,
                    'gpt4_prediction': gpt4_json,
                    'claude_prediction': claude_json,
                    'open_model_prediction': open_json,
                    'confidence_score': confidence_score,
                    'needs_review': needs_review,
                    'priority': priority,
                    'flag_reason': flag_reason
                }
            )'''
)

# Fix 2: get_evidence_by_id method
content = content.replace(
    '''            result = session.execute(
                text("SELECT * FROM evidence_items WHERE id = ?"),
                (evidence_id,)
            ).fetchone()''',
    '''            result = session.execute(
                text("SELECT * FROM evidence_items WHERE id = :id"),
                {'id': evidence_id}
            ).fetchone()'''
)

# Fix 3: submit_review method
content = content.replace(
    '''            session.execute(
                text("""
                UPDATE evidence_items
                SET human_review_status = ?,
                    human_reviewer_id = ?,
                    human_answer = ?,
                    human_confidence = ?,
                    human_notes = ?,
                    review_timestamp = ?,
                    review_duration_seconds = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """),
                (decision, reviewer_id, corrected_json, confidence, notes,
                 datetime.now(), duration_seconds, evidence_id)
            )''',
    '''            session.execute(
                text("""
                UPDATE evidence_items
                SET human_review_status = :decision,
                    human_reviewer_id = :reviewer_id,
                    human_answer = :answer,
                    human_confidence = :confidence,
                    human_notes = :notes,
                    review_timestamp = :timestamp,
                    review_duration_seconds = :duration,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = :id
                """),
                {
                    'decision': decision,
                    'reviewer_id': reviewer_id,
                    'answer': corrected_json,
                    'confidence': confidence,
                    'notes': notes,
                    'timestamp': datetime.now(),
                    'duration': duration_seconds,
                    'id': evidence_id
                }
            )'''
)

# Fix 4: get_review_progress method
content = content.replace(
    '''            result = session.execute(
                text("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN needs_human_review = 1 THEN 1 ELSE 0 END) as needs_review,
                    SUM(CASE WHEN human_review_status = 'pending' THEN 1 ELSE 0 END) as pending,
                    SUM(CASE WHEN human_review_status = 'approved' THEN 1 ELSE 0 END) as approved,
                    SUM(CASE WHEN human_review_status = 'corrected' THEN 1 ELSE 0 END) as corrected,
                    SUM(CASE WHEN human_review_status = 'rejected' THEN 1 ELSE 0 END) as rejected,
                    SUM(CASE WHEN human_review_status = 'skipped' THEN 1 ELSE 0 END) as skipped
                FROM evidence_items
                WHERE video_id = ?
                """),
                (video_id,)
            ).fetchone()''',
    '''            result = session.execute(
                text("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN needs_human_review = 1 THEN 1 ELSE 0 END) as needs_review,
                    SUM(CASE WHEN human_review_status = 'pending' THEN 1 ELSE 0 END) as pending,
                    SUM(CASE WHEN human_review_status = 'approved' THEN 1 ELSE 0 END) as approved,
                    SUM(CASE WHEN human_review_status = 'corrected' THEN 1 ELSE 0 END) as corrected,
                    SUM(CASE WHEN human_review_status = 'rejected' THEN 1 ELSE 0 END) as rejected,
                    SUM(CASE WHEN human_review_status = 'skipped' THEN 1 ELSE 0 END) as skipped
                FROM evidence_items
                WHERE video_id = :video_id
                """),
                {'video_id': video_id}
            ).fetchone()'''
)

# Fix 5: get_reviewer_stats method
content = content.replace(
    '''            query = """
                SELECT
                    COUNT(*) as items_reviewed,
                    AVG(review_duration_seconds) as avg_review_time,
                    SUM(CASE WHEN human_review_status = 'approved' THEN 1 ELSE 0 END) as approved,
                    SUM(CASE WHEN human_review_status = 'corrected' THEN 1 ELSE 0 END) as corrected,
                    SUM(CASE WHEN human_review_status = 'rejected' THEN 1 ELSE 0 END) as rejected
                FROM evidence_items
                WHERE human_reviewer_id = ?
            """
            params = [reviewer_id]

            if date_filter:
                query += " AND DATE(review_timestamp) = ?"
                params.append(date_filter)

            result = session.execute(text(query), tuple(params)).fetchone()''',
    '''            query = """
                SELECT
                    COUNT(*) as items_reviewed,
                    AVG(review_duration_seconds) as avg_review_time,
                    SUM(CASE WHEN human_review_status = 'approved' THEN 1 ELSE 0 END) as approved,
                    SUM(CASE WHEN human_review_status = 'corrected' THEN 1 ELSE 0 END) as corrected,
                    SUM(CASE WHEN human_review_status = 'rejected' THEN 1 ELSE 0 END) as rejected
                FROM evidence_items
                WHERE human_reviewer_id = :reviewer_id
            """
            params = {'reviewer_id': reviewer_id}

            if date_filter:
                query += " AND DATE(review_timestamp) = :date_filter"
                params['date_filter'] = date_filter

            result = session.execute(text(query), params).fetchone()'''
)

# Fix 6: update_video_evidence_stats method
content = content.replace(
    '''            stats = session.execute(
                text("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN needs_human_review = 1 THEN 1 ELSE 0 END) as needs_review,
                    SUM(CASE WHEN human_review_status != 'pending' THEN 1 ELSE 0 END) as approved,
                    AVG(confidence_score) as avg_confidence
                FROM evidence_items
                WHERE video_id = ?
                """),
                (video_id,)
            ).fetchone()

            if stats:
                # Update videos table
                session.execute(
                    text("""
                    UPDATE videos
                    SET ai_evidence_count = ?,
                        evidence_needs_review_count = ?,
                        evidence_approved_count = ?,
                        evidence_accuracy_estimate = ?
                    WHERE video_id = ?
                    """),
                    (stats['total'], stats['needs_review'],
                     stats['approved'], stats['avg_confidence'], video_id)
                )''',
    '''            stats = session.execute(
                text("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN needs_human_review = 1 THEN 1 ELSE 0 END) as needs_review,
                    SUM(CASE WHEN human_review_status != 'pending' THEN 1 ELSE 0 END) as approved,
                    AVG(confidence_score) as avg_confidence
                FROM evidence_items
                WHERE video_id = :video_id
                """),
                {'video_id': video_id}
            ).fetchone()

            if stats:
                stats_dict = dict(stats._mapping)
                # Update videos table
                session.execute(
                    text("""
                    UPDATE videos
                    SET ai_evidence_count = :total,
                        evidence_needs_review_count = :needs_review,
                        evidence_approved_count = :approved,
                        evidence_accuracy_estimate = :accuracy
                    WHERE video_id = :video_id
                    """),
                    {
                        'total': stats_dict['total'],
                        'needs_review': stats_dict['needs_review'],
                        'approved': stats_dict['approved'],
                        'accuracy': stats_dict['avg_confidence'],
                        'video_id': video_id
                    }
                )'''
)

# Fix 7: start_session method
content = content.replace(
    '''            result = session.execute(
                text("""
                INSERT INTO review_sessions (reviewer_id, session_start, session_type, is_active)
                VALUES (?, ?, ?, 1)
                """),
                (reviewer_id, datetime.now(), session_type)
            )''',
    '''            result = session.execute(
                text("""
                INSERT INTO review_sessions (reviewer_id, session_start, session_type, is_active)
                VALUES (:reviewer_id, :session_start, :session_type, 1)
                """),
                {
                    'reviewer_id': reviewer_id,
                    'session_start': datetime.now(),
                    'session_type': session_type
                }
            )'''
)

# Fix 8: end_session method
content = content.replace(
    '''            session.execute(
                text("""
                UPDATE review_sessions
                SET session_end = ?,
                    items_reviewed = ?,
                    is_active = 0
                WHERE id = ?
                """),
                (datetime.now(), items_reviewed, session_id)
            )''',
    '''            session.execute(
                text("""
                UPDATE review_sessions
                SET session_end = :session_end,
                    items_reviewed = :items_reviewed,
                    is_active = 0
                WHERE id = :id
                """),
                {
                    'session_end': datetime.now(),
                    'items_reviewed': items_reviewed,
                    'id': session_id
                }
            )'''
)

# Write the fixed file
with open('/Users/aranja14/Desktop/Gemini_QA/database/evidence_operations.py', 'w') as f:
    f.write(content)

print("✅ Fixed all SQLAlchemy parameter issues in evidence_operations.py")
