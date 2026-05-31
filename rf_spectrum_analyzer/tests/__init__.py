"""
Test Suite for RF Spectrum Analyzer
Comprehensive testing framework for all modules, classes, and functions
"""

import sys
import os
from pathlib import Path

# Add workspace root to path
workspace_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(workspace_root))

import unittest
import logging
from typing import List, Dict, Any

# Configure test logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_results.log')
    ]
)

class TestSuiteRunner:
    """Main test suite runner for RF Spectrum Analyzer"""
    
    def __init__(self):
        self.test_modules = [
            'test_imports',
            'test_dsp_filters',
            'test_dsp_modulation', 
            'test_dsp_analysis',
            'test_dsp_utils',
            'test_core',
            'test_backends',
            'test_gui',
            'test_integration',
            'test_debug_performance'
        ]
        self.results = {}
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test modules and collect results"""
        print("="*70)
        print("RF SPECTRUM ANALYZER - COMPREHENSIVE TEST SUITE")
        print("="*70)
        
        for module in self.test_modules:
            print(f"\n{'='*50}")
            print(f"Running tests for: {module}")
            print(f"{'='*50}")
            
            try:
                # Import and run test module
                test_module = __import__(module)
                
                # Create test suite
                loader = unittest.TestLoader()
                suite = loader.loadTestsFromModule(test_module)
                
                # Run tests
                runner = unittest.TextTestRunner(verbosity=2)
                result = runner.run(suite)
                
                # Store results
                self.results[module] = {
                    'tests_run': result.testsRun,
                    'failures': len(result.failures),
                    'errors': len(result.errors),
                    'success': result.wasSuccessful()
                }
                
            except ImportError as e:
                print(f"Could not import {module}: {e}")
                self.results[module] = {
                    'tests_run': 0,
                    'failures': 0,
                    'errors': 1,
                    'success': False,
                    'import_error': str(e)
                }
            except Exception as e:
                print(f"Error running {module}: {e}")
                self.results[module] = {
                    'tests_run': 0,
                    'failures': 0,
                    'errors': 1,
                    'success': False,
                    'runtime_error': str(e)
                }
        
        self._print_summary()
        return self.results
    
    def _print_summary(self):
        """Print test results summary"""
        print("\n" + "="*70)
        print("TEST RESULTS SUMMARY")
        print("="*70)
        
        total_tests = 0
        total_failures = 0
        total_errors = 0
        successful_modules = 0
        
        for module, result in self.results.items():
            tests = result.get('tests_run', 0)
            failures = result.get('failures', 0)
            errors = result.get('errors', 0)
            success = result.get('success', False)
            
            status = "PASS" if success else "FAIL"
            print(f"{module:30} | Tests: {tests:3} | Failures: {failures:2} | Errors: {errors:2} | {status}")
            
            total_tests += tests
            total_failures += failures
            total_errors += errors
            if success:
                successful_modules += 1
        
        print("-" * 70)
        print(f"{'TOTAL':30} | Tests: {total_tests:3} | Failures: {total_failures:2} | Errors: {total_errors:2}")
        print(f"Modules: {successful_modules}/{len(self.test_modules)} successful")
        print("-" * 70)
        
        if total_failures == 0 and total_errors == 0:
            print("🎉 ALL TESTS PASSED!")
        else:
            print("❌ SOME TESTS FAILED - Check logs for details")


def main():
    """Run the complete test suite"""
    runner = TestSuiteRunner()
    results = runner.run_all_tests()
    
    # Return exit code based on results
    all_success = all(result.get('success', False) for result in results.values())
    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())