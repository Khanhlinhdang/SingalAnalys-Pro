#!/usr/bin/env python3
"""
Test Runner Script for RF Spectrum Analyzer
Runs tests with proper Python path configuration
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.absolute()
rf_analyzer_path = project_root / "rf_spectrum_analyzer"

# Add both project root and rf_spectrum_analyzer to Python path
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(rf_analyzer_path))

# Change to the tests directory
tests_dir = rf_analyzer_path / "tests"
os.chdir(tests_dir)

# Import and run the test runner
sys.path.insert(0, str(tests_dir))
import run_tests

if __name__ == "__main__":
    print(f"Project root: {project_root}")
    print(f"RF Analyzer path: {rf_analyzer_path}")
    print(f"Tests directory: {tests_dir}")
    print(f"Python path: {sys.path[:4]}")
    print("=" * 80)
    
    # Run the tests
    run_tests.main()