#!/usr/bin/env python3
"""
Block 4 Import Verification Test

This script verifies that all Block 4 dependencies and imports are working correctly.
"""

import sys
from pathlib import Path


def test_external_dependencies():
    """Test external package imports"""
    print("="*70)
    print("Testing External Dependencies")
    print("="*70)
    
    deps = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'openai': 'openai',
        'easyocr': 'easyocr',
        'ultralytics': 'ultralytics',
        'scenedetect': 'scenedetect',
        'torch': 'torch'
    }
    
    failed = []
    for module, package in deps.items():
        try:
            if module == 'ultralytics':
                from ultralytics import YOLO
                print(f"  ✅ {package:<20} - YOLO imported")
            elif module == 'scenedetect':
                from scenedetect import detect, ContentDetector
                print(f"  ✅ {package:<20} - Scene detection ready")
            elif module == 'torch':
                import torch
                cuda_status = "CUDA available" if torch.cuda.is_available() else "CPU only"
                print(f"  ✅ {package:<20} - {cuda_status}")
            else:
                __import__(module)
                print(f"  ✅ {package:<20}")
        except ImportError as e:
            print(f"  ❌ {package:<20} - {str(e)[:40]}")
            failed.append(package)
    
    return failed


def test_block4_imports():
    """Test Block 4 module imports"""
    print("\n" + "="*70)
    print("Testing Block 4 Modules")
    print("="*70)
    
    # Add parent directory to path
    parent = Path(__file__).parent.parent
    sys.path.insert(0, str(parent))
    
    try:
        from processing import (
            VideoProcessor,
            FrameExtractor,
            AudioProcessor,
            OCRProcessor,
            ObjectDetector,
            SceneDetector,
            EvidenceExtractor,
            CostOptimizer
        )
        
        print(f"  ✅ VideoProcessor")
        print(f"  ✅ FrameExtractor")
        print(f"  ✅ AudioProcessor")
        print(f"  ✅ OCRProcessor")
        print(f"  ✅ ObjectDetector")
        print(f"  ✅ SceneDetector")
        print(f"  ✅ EvidenceExtractor")
        print(f"  ✅ CostOptimizer")
        
        # Test enums and dataclasses
        from processing import (
            SamplingStrategy,
            EvidenceType,
            CostCategory,
            ObjectCategory
        )
        
        print(f"  ✅ All enums and types")
        
        return True
    except ImportError as e:
        print(f"  ❌ Block 4 import failed: {e}")
        return False


def test_system_commands():
    """Test system dependencies"""
    print("\n" + "="*70)
    print("Testing System Commands")
    print("="*70)
    
    import subprocess
    
    # Test ffmpeg
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"  ✅ ffmpeg - {version_line}")
        else:
            print(f"  ❌ ffmpeg - returned error code {result.returncode}")
    except FileNotFoundError:
        print(f"  ❌ ffmpeg - not found in PATH")
    except Exception as e:
        print(f"  ❌ ffmpeg - {str(e)}")


def print_summary(failed_deps, block4_ok):
    """Print test summary"""
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    if not failed_deps and block4_ok:
        print("✅ ALL TESTS PASSED!")
        print("\nBlock 4 is ready to use. You can now:")
        print("  1. Import modules: from processing import *")
        print("  2. Process videos with VideoProcessor")
        print("  3. Extract evidence with EvidenceExtractor")
        print("  4. Monitor costs with CostOptimizer")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        
        if failed_deps:
            print(f"\n⚠️  Missing {len(failed_deps)} package(s):")
            for pkg in failed_deps:
                print(f"     - {pkg}")
            print("\nInstall missing packages:")
            print(f"pip install {' '.join(failed_deps)} --break-system-packages")
        
        if not block4_ok:
            print("\n⚠️  Block 4 modules failed to import")
            print("Ensure all files are in the processing/ directory:")
            print("  - __init__.py")
            print("  - video_processor.py")
            print("  - frame_extractor.py")
            print("  - audio_processor.py")
            print("  - ocr_processor.py")
            print("  - object_detector.py")
            print("  - scene_detector.py")
            print("  - evidence_extractor.py")
            print("  - cost_optimizer.py")
        
        return False


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*20 + "BLOCK 4 IMPORT TEST" + " "*29 + "║")
    print("╚" + "="*68 + "╝")
    print()
    
    # Run tests
    failed_deps = test_external_dependencies()
    block4_ok = test_block4_imports()
    test_system_commands()
    
    # Print summary
    success = print_summary(failed_deps, block4_ok)
    
    print("="*70)
    
    # Return exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
