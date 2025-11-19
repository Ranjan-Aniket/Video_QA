"""
Test Block 1: Config & Utils
Run: python test_block1.py
"""
from config.settings import settings, get_settings
from utils.logger import get_logger, app_logger
from utils.cost_tracker import cost_tracker
from utils.helpers import *

def test_settings():
    """Test configuration loading"""
    print("\n=== Testing Settings ===")
    
    s = get_settings()
    print(f"✅ Settings loaded")
    print(f"  - Database URL: {s.database_url[:30]}...")
    print(f"  - Gemini Model: {s.gemini_model}")
    print(f"  - Min Confidence: {s.min_confidence_threshold}")
    print(f"  - Candidates to Generate: {s.candidates_to_generate}")
    print(f"  - Log Directory: {s.log_dir}")
    print(f"✅ All directories created")

def test_logger():
    """Test logging system"""
    print("\n=== Testing Logger ===")
    
    logger = get_logger("test")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    print(f"✅ Logger working")
    print(f"  - Check logs/ directory for output files")

def test_cost_tracker():
    """Test cost tracking"""
    print("\n=== Testing Cost Tracker ===")
    
    cost_tracker.start_video("test_video_001")
    cost_tracker.add_cost("evidence_extraction_light", 0.50)
    cost_tracker.add_cost("qa_generation_llama", 0.30)
    cost_tracker.add_cost("gemini_pro", 2.00)
    
    summary = cost_tracker.finish_video()
    print(f"✅ Cost tracking working")
    print(f"  - Total cost: ${summary['total']:.2f}")
    print(f"  - Check logs/cost.json for details")

def test_helpers():
    """Test helper functions"""
    print("\n=== Testing Helpers ===")
    
    # Timestamp tests
    assert parse_timestamp("01:23:45") == 5025.0
    assert parse_timestamp("23:45") == 1425.0
    assert format_timestamp(5025) == "01:23:45"
    print("✅ Timestamp functions working")
    
    # Text processing
    text = 'He said "hello" and "goodbye"'
    quotes = extract_quotes(text)
    assert quotes == ['hello', 'goodbye']
    print("✅ Text processing working")
    
    # Number extraction
    numbers = extract_numbers("We see 15 times at 3.5 seconds")
    assert numbers == [15.0, 3.5]
    print("✅ Number extraction working")
    
    # Hash generation
    hash1 = generate_hash("test")
    assert len(hash1) == 32
    print("✅ Hash generation working")
    
    # URL validation
    assert is_valid_url("https://google.com")
    assert is_google_drive_url("https://drive.google.com/file/d/abc123")
    print("✅ URL validation working")

def main():
    """Run all tests"""
    print("="*60)
    print("BLOCK 1 TESTS: Config & Utils")
    print("="*60)
    
    try:
        test_settings()
        test_logger()
        test_cost_tracker()
        test_helpers()
        
        print("\n" + "="*60)
        print("✅ ALL BLOCK 1 TESTS PASSED")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()