#!/usr/bin/env python
"""
Checkpoint Verification Script
Verifies all fixes are in place for OPPORTUNITIES_COMPLETE checkpoint
"""

import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if file exists"""
    if Path(filepath).exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} NOT FOUND")
        return False

def check_line_in_file(filepath, search_text, description):
    """Check if specific text exists in file"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            if search_text in content:
                print(f"✅ {description}")
                return True
            else:
                print(f"❌ {description} - NOT FOUND")
                return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False

def main():
    print("=" * 70)
    print("CHECKPOINT VERIFICATION: OPPORTUNITIES_COMPLETE_2025-11-19")
    print("=" * 70)
    print()

    base_path = Path(__file__).parent
    all_checks = []

    # Check checkpoint files exist
    print("📄 Checking Checkpoint Files...")
    all_checks.append(check_file_exists(
        base_path / "CHECKPOINT_OPPORTUNITIES_COMPLETE.md",
        "Checkpoint documentation"
    ))
    all_checks.append(check_file_exists(
        base_path / "MODIFIED_FILES_LIST.txt",
        "Modified files list"
    ))
    all_checks.append(check_file_exists(
        base_path / "SESSION_SUMMARY.md",
        "Session summary"
    ))
    print()

    # Check backend files
    print("🔧 Checking Backend Fixes...")

    # 1. YOLO fix
    all_checks.append(check_line_in_file(
        base_path / "processing/bulk_frame_analyzer.py",
        "frame_result = results.frame_results[0]",
        "YOLO data extraction fix"
    ))

    # 2. OCR implementation
    all_checks.append(check_line_in_file(
        base_path / "processing/ocr_processor.py",
        "def _extract_with_paddleocr",
        "PaddleOCR implementation"
    ))

    # 3. Scene classifier import
    all_checks.append(check_line_in_file(
        base_path / "processing/bulk_frame_analyzer.py",
        "from .places365_processor import Places365Processor",
        "Scene classifier import fix"
    ))

    # 4. BLIP-2 implementation
    all_checks.append(check_line_in_file(
        base_path / "processing/blip2_processor.py",
        "self.processor = Blip2Processor.from_pretrained",
        "BLIP-2 model loading"
    ))

    # 5. Question generator fix
    all_checks.append(check_line_in_file(
        base_path / "processing/multimodal_question_generator_v2.py",
        "jersey_numbers.append(f\"#{player['jersey_number']}\")",
        "Question generator using rich AI data"
    ))

    # 6. CLIP encode_image
    all_checks.append(check_line_in_file(
        base_path / "processing/clip_processor.py",
        "def encode_image(self, frame: np.ndarray)",
        "CLIP encode_image method"
    ))

    # 7. Models enabled
    all_checks.append(check_line_in_file(
        base_path / "processing/smart_pipeline.py",
        "enable_clip=True",
        "CLIP enabled in pipeline"
    ))
    all_checks.append(check_line_in_file(
        base_path / "processing/smart_pipeline.py",
        "enable_fer=True",
        "FER enabled in pipeline"
    ))

    print()

    # Check frontend files
    print("🎨 Checking Frontend Enhancements...")

    all_checks.append(check_line_in_file(
        base_path / "frontend/src/components/QuestionCard.tsx",
        "onSeekTo?: (timestamp: string) => void",
        "Timestamp callback prop"
    ))
    all_checks.append(check_line_in_file(
        base_path / "frontend/src/components/QuestionCard.tsx",
        "bg-purple-100 text-purple-800",
        "Timestamp badge styling"
    ))

    print()

    # Summary
    print("=" * 70)
    passed = sum(all_checks)
    total = len(all_checks)
    percentage = (passed / total) * 100 if total > 0 else 0

    print(f"VERIFICATION RESULT: {passed}/{total} checks passed ({percentage:.1f}%)")

    if passed == total:
        print("🎉 ✅ ALL CHECKS PASSED - Checkpoint verified!")
        print()
        print("Pipeline is ready to run. Try:")
        print("  python -m processing.smart_pipeline --video <path_to_video>")
        return 0
    else:
        print("⚠️  Some checks failed. Review the output above.")
        print(f"   {total - passed} issue(s) found.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
