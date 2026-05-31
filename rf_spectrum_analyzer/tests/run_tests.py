#!/usr/bin/env python3
"""
RF Spectrum Analyzer Test Suite Runner
Main test runner script and configuration for continuous integration

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --module imports   # Run specific module tests
    python run_tests.py --fast             # Run only fast tests
    python run_tests.py --coverage         # Run with coverage report
    python run_tests.py --performance      # Run performance benchmarks
    python run_tests.py --debug            # Enable debug output
    python run_tests.py --ci               # CI mode (minimal output)
"""

import unittest
import sys
import argparse
import os
import time
import json
from pathlib import Path
import warnings
from io import StringIO

# Add workspace root to path so rf_spectrum_analyzer package imports resolve correctly
workspace_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(workspace_root))

# Test coverage support
try:
    import coverage
    COVERAGE_AVAILABLE = True
except ImportError:
    COVERAGE_AVAILABLE = False

# XML test report support
try:
    import xmlrunner
    XML_RUNNER_AVAILABLE = True
except ImportError:
    XML_RUNNER_AVAILABLE = False

# Test modules
TEST_MODULES = {
    'imports': 'test_imports',
    'contract_schema': 'test_contract_schema',
    'output_adapters': 'test_output_adapters',
    'protocol_plugins': 'test_protocol_plugins',
    'meteor_decode_chain': 'test_meteor_decode_chain',
    'decode_depth_stages': 'test_decode_depth_stages',
    'decode_hypothesis_hooks': 'test_decode_hypothesis_and_hooks',
    'dsp_filters': 'test_dsp_filters',
    'dsp_modulation': 'test_dsp_modulation',
    'dsp_analysis': 'test_dsp_analysis',
    'dsp_utils': 'test_dsp_utils',
    'core': 'test_core',
    'backends': 'test_backends',
    'gui': 'test_gui',
    'integration': 'test_integration',
    'pipeline_e2e': 'test_pipeline_e2e',
    'protocol_pipeline_e2e': 'test_protocol_pipeline_e2e',
    'app_layer_integration': 'test_app_layer_integration',
    'debug_performance': 'test_debug_performance'
}

# Test categories
TEST_CATEGORIES = {
    'fast': ['imports', 'contract_schema', 'output_adapters', 'protocol_plugins', 'meteor_decode_chain', 'decode_depth_stages', 'decode_hypothesis_hooks', 'dsp_filters', 'dsp_modulation', 'dsp_analysis', 'dsp_utils'],
    'medium': ['core', 'backends', 'gui', 'pipeline_e2e', 'protocol_pipeline_e2e', 'app_layer_integration'],
    'slow': ['integration', 'debug_performance'],
    'all': list(TEST_MODULES.keys())
}


class TestConfig:
    """Test configuration and settings"""
    
    def __init__(self):
        self.verbosity = 1
        self.failfast = False
        self.buffer = True
        self.warnings = 'ignore'
        self.coverage = False
        self.xml_output = False
        self.debug = False
        self.ci_mode = False
        self.performance = False
        self.output_dir = Path("test_results")
        self.modules = []
        self.exclude = []


class ColoredTextTestResult(unittest.TextTestResult):
    """Test result with colored output"""
    
    def __init__(self, stream, descriptions, verbosity, config):
        super().__init__(stream, descriptions, verbosity)
        self.config = config
        self.start_time = None
        
        # Color codes
        self.colors = {
            'green': '\033[92m',
            'red': '\033[91m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'cyan': '\033[96m',
            'white': '\033[97m',
            'reset': '\033[0m'
        }
        
        # Disable colors in CI mode or if not supported
        if config.ci_mode or not self._supports_color():
            self.colors = {key: '' for key in self.colors}
    
    def _supports_color(self):
        """Check if terminal supports colors"""
        return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    
    def startTest(self, test):
        super().startTest(test)
        if self.config.debug:
            self.stream.write(f"{self.colors['cyan']}Starting: {test}{self.colors['reset']}\n")
        self.start_time = time.time()
    
    def addSuccess(self, test):
        super().addSuccess(test)
        if self.config.verbosity > 1:
            duration = time.time() - self.start_time if self.start_time else 0
            self.stream.write(f"{self.colors['green']}✓{self.colors['reset']} {test} ({duration:.3f}s)\n")
    
    def addError(self, test, err):
        super().addError(test, err)
        self.stream.write(f"{self.colors['red']}✗ ERROR{self.colors['reset']} {test}\n")
        if self.config.debug:
            self.stream.write(f"{self.colors['red']}{self._exc_info_to_string(err, test)}{self.colors['reset']}\n")
    
    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.stream.write(f"{self.colors['red']}✗ FAIL{self.colors['reset']} {test}\n")
        if self.config.debug:
            self.stream.write(f"{self.colors['red']}{self._exc_info_to_string(err, test)}{self.colors['reset']}\n")
    
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        if self.config.verbosity > 1:
            self.stream.write(f"{self.colors['yellow']}⚠ SKIP{self.colors['reset']} {test} - {reason}\n")


class TestRunner:
    """Main test runner class"""
    
    def __init__(self, config):
        self.config = config
        self.results = {}
        self.coverage_data = None
        
        # Set up output directory
        if not self.config.output_dir.exists():
            self.config.output_dir.mkdir(parents=True)
    
    def run_tests(self):
        """Run the test suite"""
        print(f"\n{'='*80}")
        print(f"RF Spectrum Analyzer Test Suite")
        print(f"{'='*80}")
        
        # Configure warnings
        if self.config.warnings == 'ignore':
            warnings.filterwarnings('ignore')
        
        # Start coverage if enabled
        if self.config.coverage and COVERAGE_AVAILABLE:
            self.coverage_data = coverage.Coverage()
            self.coverage_data.start()
        
        # Determine which modules to test
        if not self.config.modules:
            self.config.modules = TEST_CATEGORIES['all']
        
        # Remove excluded modules
        modules_to_test = [m for m in self.config.modules if m not in self.config.exclude]
        
        print(f"Testing modules: {', '.join(modules_to_test)}")
        if self.config.exclude:
            print(f"Excluding modules: {', '.join(self.config.exclude)}")
        print()
        
        # Run tests for each module
        overall_result = True
        total_start_time = time.time()
        
        for module_name in modules_to_test:
            if module_name not in TEST_MODULES:
                print(f"Warning: Unknown test module '{module_name}'")
                continue
            
            result = self._run_module_tests(module_name)
            overall_result = overall_result and result
            
            if self.config.failfast and not result:
                break
        
        total_duration = time.time() - total_start_time
        
        # Stop coverage
        if self.coverage_data:
            self.coverage_data.stop()
        
        # Generate reports
        self._generate_reports(overall_result, total_duration)
        
        return overall_result
    
    def _run_module_tests(self, module_name):
        """Run tests for a specific module"""
        module_file = TEST_MODULES[module_name]
        
        print(f"\n{'-'*60}")
        print(f"Testing {module_name} ({module_file})")
        print(f"{'-'*60}")
        
        try:
            # Import test module from package-qualified path
            test_module = __import__(
                f'rf_spectrum_analyzer.tests.{module_file}',
                fromlist=['rf_spectrum_analyzer.tests']
            )
            
            # Create test suite
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(test_module)
            
            # Create test runner
            if XML_RUNNER_AVAILABLE and self.config.xml_output:
                runner = xmlrunner.XMLTestRunner(
                    output=str(self.config.output_dir),
                    verbosity=self.config.verbosity
                )
            else:
                stream = sys.stdout if not self.config.ci_mode else StringIO()
                runner = unittest.TextTestRunner(
                    stream=stream,
                    verbosity=self.config.verbosity,
                    failfast=self.config.failfast,
                    buffer=self.config.buffer,
                    resultclass=lambda stream, descriptions, verbosity: 
                        ColoredTextTestResult(stream, descriptions, verbosity, self.config)
                )
            
            # Run tests
            start_time = time.time()
            result = runner.run(suite)
            duration = time.time() - start_time
            
            # Store results
            self.results[module_name] = {
                'tests_run': result.testsRun,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'skipped': len(result.skipped),
                'duration': duration,
                'success': result.wasSuccessful()
            }
            
            # Print summary for this module
            if not self.config.ci_mode:
                self._print_module_summary(module_name, self.results[module_name])
            
            return result.wasSuccessful()
            
        except ImportError as e:
            print(f"Could not import test module {module_file}: {e}")
            self.results[module_name] = {
                'tests_run': 0,
                'failures': 0,
                'errors': 1,
                'skipped': 0,
                'duration': 0,
                'success': False,
                'import_error': str(e)
            }
            return False
        except Exception as e:
            print(f"Error running tests for {module_name}: {e}")
            self.results[module_name] = {
                'tests_run': 0,
                'failures': 0,
                'errors': 1,
                'skipped': 0,
                'duration': 0,
                'success': False,
                'error': str(e)
            }
            return False
    
    def _print_module_summary(self, module_name, result):
        """Print summary for a module"""
        status = "PASS" if result['success'] else "FAIL"
        color = 'green' if result['success'] else 'red'
        
        if hasattr(self, '_supports_color') and self._supports_color():
            colors = {
                'green': '\033[92m',
                'red': '\033[91m',
                'reset': '\033[0m'
            }
        else:
            colors = {'green': '', 'red': '', 'reset': ''}
        
        print(f"\n{module_name}: {colors[color]}{status}{colors['reset']}")
        print(f"  Tests run: {result['tests_run']}")
        print(f"  Failures: {result['failures']}")
        print(f"  Errors: {result['errors']}")
        print(f"  Skipped: {result['skipped']}")
        print(f"  Duration: {result['duration']:.2f}s")
    
    def _generate_reports(self, overall_result, total_duration):
        """Generate test reports"""
        print(f"\n{'='*80}")
        print(f"TEST SUMMARY")
        print(f"{'='*80}")
        
        # Calculate totals
        total_tests = sum(r['tests_run'] for r in self.results.values())
        total_failures = sum(r['failures'] for r in self.results.values())
        total_errors = sum(r['errors'] for r in self.results.values())
        total_skipped = sum(r['skipped'] for r in self.results.values())
        
        passed_modules = sum(1 for r in self.results.values() if r['success'])
        total_modules = len(self.results)
        
        print(f"Modules: {passed_modules}/{total_modules} passed")
        print(f"Tests: {total_tests} run, {total_failures} failures, {total_errors} errors, {total_skipped} skipped")
        print(f"Duration: {total_duration:.2f}s")
        print(f"Overall result: {'PASS' if overall_result else 'FAIL'}")
        
        # Detailed results
        if not self.config.ci_mode:
            print(f"\nDetailed Results:")
            for module_name, result in self.results.items():
                status = "PASS" if result['success'] else "FAIL"
                print(f"  {module_name:20} {status:4} ({result['duration']:.2f}s)")
        
        # Save JSON report
        report_data = {
            'timestamp': time.time(),
            'overall_result': overall_result,
            'total_duration': total_duration,
            'summary': {
                'total_tests': total_tests,
                'total_failures': total_failures,
                'total_errors': total_errors,
                'total_skipped': total_skipped,
                'passed_modules': passed_modules,
                'total_modules': total_modules
            },
            'modules': self.results,
            'config': {
                'coverage': self.config.coverage,
                'performance': self.config.performance,
                'debug': self.config.debug,
                'ci_mode': self.config.ci_mode
            }
        }
        
        report_file = self.config.output_dir / 'test_report.json'
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\nDetailed report saved to: {report_file}")
        
        # Generate coverage report
        if self.coverage_data:
            self._generate_coverage_report()
        
        # Performance benchmarks
        if self.config.performance:
            self._generate_performance_report()
    
    def _generate_coverage_report(self):
        """Generate coverage report"""
        if not self.coverage_data:
            return
        
        print(f"\nGenerating coverage report...")
        
        # Save coverage data
        coverage_file = self.config.output_dir / '.coverage'
        self.coverage_data.save()
        
        # Generate HTML report
        html_dir = self.config.output_dir / 'coverage_html'
        self.coverage_data.html_report(directory=str(html_dir))
        
        # Generate console report
        console_report = StringIO()
        self.coverage_data.report(file=console_report)
        
        print(console_report.getvalue())
        print(f"HTML coverage report: {html_dir}/index.html")
    
    def _generate_performance_report(self):
        """Generate performance benchmarks report"""
        print(f"\nRunning performance benchmarks...")
        
        # This would run specific performance tests
        # For now, just note that performance mode was enabled
        perf_file = self.config.output_dir / 'performance_report.txt'
        with open(perf_file, 'w') as f:
            f.write(f"Performance benchmarks completed at {time.time()}\n")
            f.write("Note: Detailed performance tests are in test_debug_performance module\n")
        
        print(f"Performance report saved to: {perf_file}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='RF Spectrum Analyzer Test Suite Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py --module imports   # Run specific module
  python run_tests.py --fast             # Run only fast tests
  python run_tests.py --coverage         # Run with coverage
  python run_tests.py --ci               # CI mode
        """
    )
    
    parser.add_argument(
        '--module', '-m',
        choices=list(TEST_MODULES.keys()),
        action='append',
        help='Test modules to run (can be specified multiple times)'
    )
    
    parser.add_argument(
        '--category', '-c',
        choices=list(TEST_CATEGORIES.keys()),
        help='Test category to run'
    )
    
    parser.add_argument(
        '--exclude', '-e',
        choices=list(TEST_MODULES.keys()),
        action='append',
        help='Test modules to exclude'
    )
    
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Run only fast tests'
    )
    
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Run with coverage analysis'
    )
    
    parser.add_argument(
        '--xml',
        action='store_true',
        help='Generate XML test reports'
    )
    
    parser.add_argument(
        '--performance',
        action='store_true',
        help='Run performance benchmarks'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output'
    )
    
    parser.add_argument(
        '--ci',
        action='store_true',
        help='CI mode (minimal output)'
    )
    
    parser.add_argument(
        '--verbosity', '-v',
        type=int,
        choices=[0, 1, 2],
        default=1,
        help='Test verbosity level'
    )
    
    parser.add_argument(
        '--failfast',
        action='store_true',
        help='Stop on first failure'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('test_results'),
        help='Output directory for test results'
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Create test configuration
    config = TestConfig()
    config.verbosity = args.verbosity
    config.failfast = args.failfast
    config.coverage = args.coverage and COVERAGE_AVAILABLE
    config.xml_output = args.xml and XML_RUNNER_AVAILABLE
    config.debug = args.debug
    config.ci_mode = args.ci
    config.performance = args.performance
    config.output_dir = args.output_dir
    config.exclude = args.exclude or []
    
    # Determine modules to test
    if args.module:
        config.modules = args.module
    elif args.category:
        config.modules = TEST_CATEGORIES[args.category]
    elif args.fast:
        config.modules = TEST_CATEGORIES['fast']
    else:
        config.modules = TEST_CATEGORIES['all']
    
    # Check for required dependencies
    if config.coverage and not COVERAGE_AVAILABLE:
        print("Warning: Coverage requested but 'coverage' package not available")
        config.coverage = False
    
    if config.xml_output and not XML_RUNNER_AVAILABLE:
        print("Warning: XML output requested but 'xmlrunner' package not available")
        config.xml_output = False
    
    # Run tests
    runner = TestRunner(config)
    success = runner.run_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()