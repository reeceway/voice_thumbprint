#!/usr/bin/env python3
"""Run all tests for VoiceThumbprint system."""

import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_test(test_file):
    """Run a single test file and return result."""
    print(f"\n{'='*70}")
    print(f"Running: {test_file}")
    print('='*70)
    
    result = subprocess.run(
        [sys.executable, test_file],
        cwd=str(Path(__file__).parent.parent),
        capture_output=False
    )
    
    return result.returncode == 0


def main():
    """Run all tests."""
    print("="*70)
    print("VoiceThumbprint - Complete Test Suite")
    print("="*70)
    
    test_files = [
        ("Modulation Spectrum", "tests/test_modulation.py"),
        ("Layer 2 (GMM-UBM + Classifiers)", "tests/test_layer2.py"),
        ("Layer 3 (Neural Embeddings)", "tests/test_layer3.py"),
    ]
    
    results = []
    
    for name, test_file in test_files:
        try:
            passed = run_test(test_file)
            results.append((name, passed))
        except Exception as e:
            print(f"❌ Error running {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print("="*70)
    print(f"Results: {passed}/{total} test suites passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test suite(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
