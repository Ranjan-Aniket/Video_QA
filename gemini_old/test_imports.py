#!/usr/bin/env python3
"""
Block 5 Import Verification Test

This script verifies that all Block 5 dependencies and imports are working correctly.
"""

import sys
from pathlib import Path


def test_external_dependencies():
    """Test external package imports"""
    print("="*70)
    print("Testing External Dependencies")
    print("="*70)
    
    deps = {
        'google.generativeai': 'google-generativeai',
    }
    
    failed = []
    for module, package in deps.items():
        try:
            __import__(module)
            print(f"  ✅ {package:<30}")
        except ImportError as e:
            print(f"  ❌ {package:<30} - {str(e)[:40]}")
            failed.append(package)
    
    return failed


def test_block5_imports():
    """Test Block 5 module imports"""
    print("\n" + "="*70)
    print("Testing Block 5 Modules")
    print("="*70)
    
    # Add parent directory to path
    parent = Path(__file__).parent.parent
    sys.path.insert(0, str(parent))
    
    try:
        from gemini import (
            GeminiClient,
            AdversarialTester,
            HallucinationDetector,
            BenchmarkAnalyzer
        )
        
        print(f"  ✅ GeminiClient")
        print(f"  ✅ AdversarialTester")
        print(f"  ✅ HallucinationDetector")
        print(f"  ✅ BenchmarkAnalyzer")
        
        # Test enums and dataclasses
        from gemini import (
            GeminiModel,
            TestStatus,
            HallucinationType
        )
        
        print(f"  ✅ All enums and types")
        
        return True
    except ImportError as e:
        print(f"  ❌ Block 5 import failed: {e}")
        return False


def test_api_key_setup():
    """Test API key configuration"""
    print("\n" + "="*70)
    print("Testing API Key Setup")
    print("="*70)
    
    import os
    
    api_key = os.getenv('GOOGLE_AI_API_KEY')
    
    if api_key:
        # Mask key for security
        masked = api_key[:10] + "..." + api_key[-4:]
        print(f"  ✅ GOOGLE_AI_API_KEY found: {masked}")
    else:
        print(f"  ⚠️  GOOGLE_AI_API_KEY not set")
        print(f"      Set it with: export GOOGLE_AI_API_KEY='your-key'")


def print_summary(failed_deps, block5_ok):
    """Print test summary"""
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    if not failed_deps and block5_ok:
        print("✅ ALL TESTS PASSED!")
        print("\nBlock 5 is ready to use. You can now:")
        print("  1. Import modules: from gemini import *")
        print("  2. Initialize GeminiClient with your API key")
        print("  3. Test Q&A pairs with AdversarialTester")
        print("  4. Detect hallucinations with HallucinationDetector")
        print("  5. Analyze results with BenchmarkAnalyzer")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        
        if failed_deps:
            print(f"\n⚠️  Missing {len(failed_deps)} package(s):")
            for pkg in failed_deps:
                print(f"     - {pkg}")
            print("\nInstall missing packages:")
            print(f"pip install {' '.join(failed_deps)} --break-system-packages")
        
        if not block5_ok:
            print("\n⚠️  Block 5 modules failed to import")
            print("Ensure all files are in the gemini/ directory:")
            print("  - __init__.py")
            print("  - gemini_client.py")
            print("  - adversarial_tester.py")
            print("  - hallucination_detector.py")
            print("  - benchmark_analyzer.py")
        
        return False


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*20 + "BLOCK 5 IMPORT TEST" + " "*29 + "║")
    print("╚" + "="*68 + "╝")
    print()
    
    # Run tests
    failed_deps = test_external_dependencies()
    block5_ok = test_block5_imports()
    test_api_key_setup()
    
    # Print summary
    success = print_summary(failed_deps, block5_ok)
    
    print("="*70)
    
    # Return exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
