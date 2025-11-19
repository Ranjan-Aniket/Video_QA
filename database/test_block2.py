"""
Test Block 2: Database
Run: pytest database/test_block2.py -v
"""
import pytest
from database import (
    init_db, db_manager,
    VideoOperations, QuestionOperations, FailureOperations,
    FeedbackOperations, AnalyticsOperations
)
from datetime import datetime
import time

def test_database_init():
    """Test database initialization"""
    print("\n=== Testing Database Initialization ===")
    
    # Drop and recreate tables (clean slate)
    db_manager.drop_tables()
    db_manager.create_tables()
    
    print("✅ Database initialized successfully")

def test_video_operations():
    """Test video CRUD operations"""
    print("\n=== Testing Video Operations ===")
    
    # Create video
    video = VideoOperations.create_video(
        video_id="test_vid_001",
        video_url="https://drive.google.com/file/d/abc123",
        batch_id="batch_001",
        batch_name="Test Batch",
        video_name="Test Video",
        duration=120.5
    )
    print(f"✅ Created video: test_vid_001")
    
    # Get video (fresh from database)
    retrieved = VideoOperations.get_video("test_vid_001")
    assert retrieved.video_id == "test_vid_001"
    print(f"✅ Retrieved video: {retrieved.video_id}")
    
    # Update video
    VideoOperations.update_video(
        "test_vid_001",
        status="processing",
        candidates_generated=30
    )
    print("✅ Updated video status")
    
    # Update status
    VideoOperations.update_status("test_vid_001", "completed")
    print("✅ Updated to completed")

def test_question_operations():
    """Test question CRUD operations"""
    print("\n=== Testing Question Operations ===")
    
    # Get the video we created
    video = VideoOperations.get_video("test_vid_001")
    assert video is not None, "Video must exist"
    
    # Create question
    question = QuestionOperations.create_question(
        question_id="q_001",
        video_id=video.id,
        question_text="How many times do we see X?",
        golden_answer="We see X 15 times at...",
        generation_tier="template",
        task_type="Counting",
        start_timestamp="00:01:30",
        end_timestamp="00:02:45",
        audio_cues=["narrator explains"],
        visual_cues=["Lego footage appears"],
        template_name="Counting with Angles"
    )
    print(f"✅ Created question: q_001")
    
    # Update validation
    QuestionOperations.update_validation_results(
        "q_001",
        passed=True,
        results={'evidence_grounding': True, 'timestamps': True},
        confidence=0.995
    )
    print("✅ Updated validation results")
    
    # Update Gemini results
    QuestionOperations.update_gemini_results(
        "q_001",
        gemini_answer="Three distinct scenes",
        failed=True
    )
    print("✅ Updated Gemini results (failed)")
    
    # Get questions
    questions = QuestionOperations.get_questions_by_video(video.id)
    assert len(questions) >= 1
    print(f"✅ Retrieved {len(questions)} questions")

def test_failure_operations():
    """Test failure CRUD operations"""
    print("\n=== Testing Failure Operations ===")
    
    # Get video and question
    video = VideoOperations.get_video("test_vid_001")
    questions = QuestionOperations.get_questions_by_video(video.id)
    question = questions[0]
    
    # Create failure
    failure = FailureOperations.create_failure(
        failure_id="f_001",
        video_id=video.id,
        question_id=question.id,
        failure_type="counting_error",
        failure_score=9.5,
        gemini_answer="Three distinct scenes",
        golden_answer="15 times",
        severity_score=10.0,
        clarity_score=9.0,
        educational_score=9.5,
        difference_summary="Gemini confused scenes with shots",
        explanation="Model failed to count individual shots"
    )
    print(f"✅ Created failure: f_001")
    
    # Mark as selected
    FailureOperations.mark_selected("f_001", rank=1, adjusted_score=9.5)
    print("✅ Marked failure as selected (rank 1)")
    
    # Get failures
    failures = FailureOperations.get_failures_by_video(video.id)
    assert len(failures) >= 1
    print(f"✅ Retrieved {len(failures)} failures")
    
    # Get selected
    selected = FailureOperations.get_selected_failures(video.id)
    assert len(selected) >= 1
    assert selected[0].selection_rank == 1
    print(f"✅ Retrieved {len(selected)} selected failures")

def test_feedback_operations():
    """Test feedback pattern operations"""
    print("\n=== Testing Feedback Operations ===")
    
    # Create pattern
    pattern = FeedbackOperations.create_pattern(
        pattern_name="counting_with_angles",
        pattern_type="template",
        question_template="How many times [OBJECT] from [CONDITION]?",
        task_type="Counting",
        failure_type="counting_error",
        is_template=True
    )
    print(f"✅ Created pattern: counting_with_angles")
    
    # Update stats (simulate usage)
    for i in range(10):
        FeedbackOperations.update_pattern_stats(
            "counting_with_angles",
            succeeded=(i % 2 == 0),  # 50% success rate
            failure_score=9.0
        )
    
    print("✅ Updated pattern stats (10 uses)")
    
    # Get patterns
    patterns = FeedbackOperations.get_active_patterns()
    assert len(patterns) >= 1
    print(f"✅ Retrieved {len(patterns)} active patterns")
    
    # Check success rate
    updated_pattern = patterns[0]
    print(f"  - Success rate: {updated_pattern.success_rate:.2%}")
    print(f"  - Avg score: {updated_pattern.avg_failure_score:.2f}")

def test_analytics():
    """Test analytics operations"""
    print("\n=== Testing Analytics ===")
    
    # Failure distribution
    distribution = AnalyticsOperations.get_failure_type_distribution(days=30)
    print(f"✅ Failure distribution: {distribution}")
    
    # Cost summary
    cost_summary = AnalyticsOperations.get_cost_summary()
    print(f"✅ Cost summary: {cost_summary}")
    
    # Quality metrics
    quality = AnalyticsOperations.get_quality_metrics()
    print(f"✅ Quality metrics: {quality}")

def main():
    """Run all tests manually"""
    print("="*60)
    print("BLOCK 2 TESTS: Database")
    print("="*60)
    
    try:
        test_database_init()
        test_video_operations()
        test_question_operations()
        test_failure_operations()
        test_feedback_operations()
        test_analytics()
        
        print("\n" + "="*60)
        print("✅ ALL BLOCK 2 TESTS PASSED")
        print("="*60)
        print("\nDatabase tables created:")
        print("  - videos")
        print("  - questions")
        print("  - failures")
        print("  - feedback_patterns")
        print("  - batches")
        print("  - system_metrics")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()