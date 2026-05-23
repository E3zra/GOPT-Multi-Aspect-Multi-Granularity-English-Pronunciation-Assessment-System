#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test runner script for GOPT unit tests

Usage:
    python tests/run_tests.py              # Run all tests
    python tests/run_tests.py --verbose    # Verbose output
    python tests/run_tests.py --coverage   # With coverage report
"""

import sys
import os
import argparse
import subprocess

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_tests(verbose=False, coverage=False, specific_test=None):
    """Run the test suite"""
    
    # Build pytest command
    cmd = ["pytest", "tests/"]
    
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    if coverage:
        cmd.extend([
            "--cov=src",
            "--cov-report=term",
            "--cov-report=html"
        ])
    
    # Add specific test if provided
    if specific_test:
        cmd = ["pytest", specific_test]
        if verbose:
            cmd.append("-v")
    
    # Add colored output
    cmd.append("--color=yes")
    
    # Add detailed output for failures
    cmd.append("--tb=short")
    
    print("=" * 70)
    print("GOPT Unit Test Suite")
    print("=" * 70)
    print(f"\nRunning command: {' '.join(cmd)}\n")
    
    # Run tests
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        
        if coverage:
            print("\n📊 Coverage report generated in htmlcov/index.html")
    else:
        print("\n" + "=" * 70)
        print("❌ SOME TESTS FAILED")
        print("=" * 70)
        print("\nRun with --verbose for more details:")
        print("    python tests/run_tests.py --verbose")
    
    return result.returncode


def list_tests():
    """List all available tests"""
    print("=" * 70)
    print("Available Test Modules")
    print("=" * 70)
    
    test_modules = [
        ("test_models.py", "Model architecture tests (GOPT, LSTM, Attention, etc.)"),
        ("test_utils.py", "Utility function tests (normalization, encoding, etc.)"),
        ("test_data_loading.py", "Data loading and preprocessing tests"),
    ]
    
    for module, description in test_modules:
        print(f"\n📝 tests/{module}")
        print(f"   {description}")
    
    print("\n" + "=" * 70)
    print("\nTo run a specific test module:")
    print("    pytest tests/test_models.py -v")
    print("\nTo run a specific test class:")
    print("    pytest tests/test_models.py::TestGOPT -v")
    print("\nTo run a specific test:")
    print("    pytest tests/test_models.py::TestGOPT::test_gopt_forward_shape -v")
    print("=" * 70)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Run GOPT unit tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tests/run_tests.py                      # Run all tests
    python tests/run_tests.py --verbose            # Verbose output
    python tests/run_tests.py --coverage           # With coverage
    python tests/run_tests.py --list               # List available tests
    python tests/run_tests.py tests/test_models.py # Run specific module
        """
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "-c", "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List available tests"
    )
    
    parser.add_argument(
        "test",
        nargs="?",
        help="Specific test file or test to run"
    )
    
    args = parser.parse_args()
    
    # Check if pytest is available
    try:
        import pytest
    except ImportError:
        print("❌ Error: pytest not installed")
        print("\nPlease install pytest:")
        print("    pip install pytest pytest-cov")
        return 1
    
    if args.list:
        list_tests()
        return 0
    
    # Check if in correct directory
    if not os.path.exists("tests"):
        print("❌ Error: tests directory not found")
        print("\nPlease run from project root directory:")
        print("    cd /path/to/gopt")
        print("    python tests/run_tests.py")
        return 1
    
    # Run tests
    return run_tests(
        verbose=args.verbose,
        coverage=args.coverage,
        specific_test=args.test
    )


if __name__ == "__main__":
    sys.exit(main())

