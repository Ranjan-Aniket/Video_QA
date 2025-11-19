#!/usr/bin/env python3
"""
Test Imports - Block 7: Learning/Feedback

Verifies all modules can be imported successfully.
"""

import sys
from pathlib import Path


def test_external_dependencies():
    """Test external package imports"""
    print("="*70)
    print("Testing External Dependencies")
    print("="*70)
    
    deps = {
        'openpyxl': 'openpyxl',  # For Excel export
    }
    
    failed = []
    for module, package in deps.items():
        try:
            __import__(module)
            print(f"  ✅ {package:<30}")
        except ImportError as e:
            print(f"  ⚠️  {package:<30} - Optional for Excel export")
            # Not marking as failed since it's optional
    
    return failed


def test_block7_imports():
    """Test Block 7 module imports"""
    print("\n" + "="*70)
    print("Testing Block 7 Modules")
    print("="*70)
    
    # Add parent directory to path
    parent = Path(__file__).parent.parent
    sys.path.insert(0, str(parent))
    
    try:
        # Test main package
        from feedback import (
            FeedbackProcessor,
            PatternLearner,
            ExportManager
        )
        
        print(f"  ✅ FeedbackProcessor")
        print(f"  ✅ PatternLearner")
        print(f"  ✅ ExportManager")
        
        # Test dataclasses and enums
        from feedback import (
            FeedbackConfig,
            FeedbackResult,
            ValidationOutcome,
            PatternConfig,
            FailurePattern,
            LearningInsights,
            ExportConfig,
            ExportFormat,
            ExcelExporter
        )
        
        print(f"  ✅ All configs and dataclasses")
        
        # Test enum values
        assert ValidationOutcome.PASSED.value == "passed"
        assert ExportFormat.EXCEL.value == "excel"
        print(f"  ✅ All enums")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Block 7 import failed: {e}")
        return False
    except AssertionError as e:
        print(f"  ❌ Enum validation failed: {e}")
        return False


def test_instantiation():
    """Test object instantiation"""
    print("\n" + "="*70)
    print("Testing Object Instantiation")
    print("="*70)
    
    try:
        from feedback import (
            FeedbackProcessor,
            PatternLearner,
            ExportManager,
            FeedbackConfig,
            PatternConfig,
            ExportConfig
        )
        
        # Test with default configs
        processor = FeedbackProcessor()
        print(f"  ✅ FeedbackProcessor() - default config")
        
        learner = PatternLearner()
        print(f"  ✅ PatternLearner() - default config")
        
        exporter = ExportManager()
        print(f"  ✅ ExportManager() - default config")
        
        # Test with custom configs
        custom_feedback = FeedbackConfig(
            min_pass_rate=0.95,
            min_gemini_fail_rate=0.35
        )
        processor = FeedbackProcessor(config=custom_feedback)
        print(f"  ✅ FeedbackProcessor(custom_config)")
        
        custom_pattern = PatternConfig(
            min_occurrences=5,
            confidence_threshold=0.8
        )
        learner = PatternLearner(config=custom_pattern)
        print(f"  ✅ PatternLearner(custom_config)")
        
        custom_export = ExportConfig(
            use_colors=True,
            include_metrics_sheet=True
        )
        exporter = ExportManager(config=custom_export)
        print(f"  ✅ ExportManager(custom_config)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Instantiation failed: {e}")
        return False


def print_summary(failed_deps, block7_ok, instantiation_ok):
    """Print test summary"""
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    if not failed_deps and block7_ok and instantiation_ok:
        print("✅ ALL TESTS PASSED!")
        print("\nBlock 7 is ready to use. You can now:")
        print("  1. Import modules: from feedback import *")
        print("  2. Process feedback with FeedbackProcessor")
        print("  3. Learn patterns with PatternLearner")
        print("  4. Export to Excel with ExportManager")
        print("\nNote: Install openpyxl for full Excel export functionality:")
        print("  pip install openpyxl --break-system-packages")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        
        if failed_deps:
            print(f"\n⚠️  Missing {len(failed_deps)} package(s):")
            for pkg in failed_deps:
                print(f"     - {pkg}")
            print("\nInstall missing packages:")
            print(f"pip install {' '.join(failed_deps)} --break-system-packages")
        
        if not block7_ok:
            print("\n⚠️  Block 7 modules failed to import")
            print("Ensure all files are in the feedback/ directory:")
            print("  - __init__.py")
            print("  - feedback_processor.py")
            print("  - pattern_learner.py")
            print("  - export_manager.py")
        
        if not instantiation_ok:
            print("\n⚠️  Object instantiation failed")
            print("Check for syntax errors or missing dependencies")
        
        return False


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*16 + "BLOCK 7 IMPORT TEST" + " "*33 + "║")
    print("╚" + "="*68 + "╝")
    print()
    
    # Run tests
    failed_deps = test_external_dependencies()
    block7_ok = test_block7_imports()
    instantiation_ok = test_instantiation() if block7_ok else False
    
    # Print summary
    success = print_summary(failed_deps, block7_ok, instantiation_ok)
    
    print("="*70)
    
    # Return exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
