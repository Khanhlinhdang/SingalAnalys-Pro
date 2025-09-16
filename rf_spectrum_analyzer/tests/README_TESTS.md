# RF Spectrum Analyzer Test Suite Documentation

## Overview

This comprehensive test suite provides complete validation and debugging capabilities for the RF Spectrum Analyzer application. The test framework covers all major components including DSP modules, SDR backends, GUI components, and application integration.

## Test Structure

```
tests/
├── __init__.py                    # Test suite runner and framework
├── run_tests.py                   # Main test runner script
├── run_local_tests.sh            # Unix/Linux test automation
├── run_local_tests.bat           # Windows test automation
├── pytest.ini                    # Test configuration
├── ci_config.yml                 # CI/CD configuration
├── test_imports.py               # Import and dependency validation
├── test_dsp_filters.py           # DSP filter module tests
├── test_dsp_modulation.py        # DSP modulation module tests
├── test_dsp_analysis.py          # DSP analysis module tests
├── test_dsp_utils.py             # DSP utility function tests
├── test_core.py                  # Core application module tests
├── test_backends.py              # SDR backend implementation tests
├── test_gui.py                   # GUI component tests
├── test_integration.py           # Integration and workflow tests
├── test_debug_performance.py     # Debug and performance tests
└── test_results/                 # Generated test reports and coverage
```

## Test Categories

### Fast Tests
- **Import validation** - Verify all dependencies are available
- **DSP filters** - FIR/IIR filters, polyphase filters, adaptive filters
- **DSP modulation** - PSK, QAM, FSK, OFDM modulators/demodulators
- **DSP analysis** - Spectrum analysis, signal detection, parameter estimation
- **DSP utilities** - Window functions, noise generation, signal utilities

### Medium Tests
- **Core modules** - SDR backend base classes, signal processor, application logic
- **Backend implementations** - HackRF, RTL-SDR, PlutoSDR, SoapySDR, USRP backends
- **GUI components** - Spectrum widget, waterfall widget, control panels, settings

### Slow Tests
- **Integration tests** - Complete application workflow and component integration
- **Debug/Performance** - Performance profiling, memory usage, benchmarking

## Usage

### Quick Start

```bash
# Run fast tests only
cd tests
python run_tests.py --fast

# Run all tests
python run_tests.py --all

# Run with coverage analysis
python run_tests.py --all --coverage

# Run specific module
python run_tests.py --module imports
```

### Command Line Options

```bash
python run_tests.py [OPTIONS]

Options:
  --module MODULE      Test specific module (imports, dsp_filters, etc.)
  --category CATEGORY  Test category (fast, medium, slow, all)
  --fast              Run only fast tests
  --coverage          Run with coverage analysis
  --xml               Generate XML test reports
  --performance       Run performance benchmarks
  --debug             Enable debug output
  --ci                CI mode (minimal output)
  --verbosity LEVEL   Test verbosity (0, 1, 2)
  --failfast          Stop on first failure
  --exclude MODULE    Exclude specific modules
```

### Automated Scripts

#### Unix/Linux/macOS
```bash
# Make executable
chmod +x run_local_tests.sh

# Run fast tests
./run_local_tests.sh --fast

# Run all tests with coverage
./run_local_tests.sh --all --coverage

# Run code quality checks
./run_local_tests.sh --lint

# Clean and run all tests
./run_local_tests.sh --clean --all
```

#### Windows
```cmd
# Run fast tests
run_local_tests.bat --fast

# Run all tests with coverage
run_local_tests.bat --all --coverage

# Run code quality checks
run_local_tests.bat --lint
```

## Test Modules Detail

### test_imports.py
Validates all project dependencies and import statements:
- Python standard library imports
- Third-party package availability (NumPy, SciPy, PySide6/PyQt5)
- SDR library imports (rtlsdr, hackrf, soapysdr, uhd)
- Hardware library availability
- Version compatibility checks

### test_dsp_filters.py
Comprehensive DSP filter testing:
- **Filter Configuration** - Parameter validation and default settings
- **FIR Filters** - Filter design, frequency response, impulse response
- **IIR Filters** - Butterworth, Chebyshev, Elliptic filter designs
- **Polyphase Filters** - Decimation, interpolation, resampling
- **Adaptive Filters** - LMS, RLS algorithm implementation
- **Filter Banks** - Multi-rate filter bank processing

### test_dsp_modulation.py
Digital modulation scheme validation:
- **PSK Modulation** - BPSK, QPSK, 8-PSK modulation/demodulation
- **QAM Modulation** - 16-QAM, 64-QAM, 256-QAM schemes
- **FSK Modulation** - Binary and M-ary FSK
- **OFDM** - Subcarrier modulation, IFFT/FFT processing
- **Constellation Analysis** - EVM calculation, symbol error rates
- **Carrier Recovery** - Phase and frequency synchronization

### test_dsp_analysis.py
Signal analysis and detection:
- **Spectrum Analysis** - PSD computation, frequency estimation
- **Signal Detection** - Energy detection, CFAR algorithms
- **Parameter Estimation** - SNR, power measurements
- **Time-Frequency Analysis** - Spectrograms, time-domain analysis

### test_dsp_utils.py
DSP utility functions:
- **Window Functions** - Hann, Hamming, Kaiser, Blackman windows
- **Noise Generation** - AWGN, colored noise, interference simulation
- **Signal Generation** - Sinusoids, chirps, multi-tone signals
- **Timing/Synchronization** - Peak detection, delay estimation, PLL
- **Mathematical Utilities** - dB conversions, RMS, PAPR calculations

### test_core.py
Core application components:
- **SDR Configuration** - Parameter validation and defaults
- **SDR Backend Base** - Abstract interface implementation
- **Signal Processor** - Core processing pipeline validation
- **Mock Backend** - Test backend for development
- **Error Handling** - Exception handling and recovery
- **Thread Safety** - Concurrent access validation

### test_backends.py
SDR hardware backend testing:
- **HackRF Backend** - Connection, streaming, parameter control
- **RTL-SDR Backend** - Device enumeration, sample acquisition
- **PlutoSDR Backend** - IIO interface, parameter configuration
- **SoapySDR Backend** - Multi-device support, plugin loading
- **USRP Backend** - UHD interface, MIMO capabilities
- **Performance Testing** - Throughput, latency, error handling
- **Mock Device Testing** - Hardware simulation for CI/CD

### test_gui.py
Graphical user interface validation:
- **Spectrum Widget** - Real-time spectrum display, zoom, cursors
- **Waterfall Widget** - Time-frequency display, colormap control
- **Control Panel** - Parameter controls, start/stop functionality
- **Settings Dialog** - Configuration management, validation
- **Main Window** - Layout, menu bar, toolbar, status bar
- **Qt Integration** - Event handling, signal/slot connections
- **Performance** - Real-time plotting, memory management

### test_integration.py
System integration and workflow:
- **Basic Integration** - SDR to processor pipeline
- **Continuous Processing** - Streaming data handling
- **Parameter Changes** - Runtime configuration updates
- **Error Recovery** - Connection failures, streaming errors
- **Performance Integration** - System throughput, latency
- **Multi-Device** - Concurrent device operation
- **Real-Time Processing** - Producer-consumer patterns

### test_debug_performance.py
Debug utilities and performance analysis:
- **Performance Profiling** - Function timing, memory usage
- **Benchmarking** - Systematic performance measurement
- **Memory Analysis** - Memory leak detection, usage patterns
- **System Monitoring** - CPU usage, throughput analysis
- **Debug Utilities** - Signal validation, processing chain analysis
- **Regression Detection** - Performance change detection
- **Report Generation** - Performance reports, visualizations

## Coverage Analysis

The test suite includes comprehensive coverage analysis:

```bash
# Run with coverage
python run_tests.py --coverage

# Coverage reports generated:
# - test_results/coverage.xml (XML format)
# - test_results/coverage_html/index.html (HTML report)
```

### Coverage Targets
- **Minimum Coverage**: 80% overall
- **Critical Modules**: 90% coverage (core, backends)
- **GUI Modules**: 70% coverage (due to Qt complexity)
- **Exclusions**: Test files, external dependencies

## Continuous Integration

### GitHub Actions Configuration
The `ci_config.yml` provides complete CI/CD setup:
- **Multi-platform testing** - Ubuntu, Windows, macOS
- **Multi-Python versions** - 3.8, 3.9, 3.10, 3.11
- **Dependency installation** - System packages, Python packages
- **Test execution** - All test categories with proper isolation
- **Coverage reporting** - Codecov integration
- **Build artifacts** - Test results, coverage reports

### Local CI Simulation
```bash
# Simulate CI environment
export CI=true
python run_tests.py --ci --coverage
```

## Test Data and Mocking

### Mock Backends
All SDR backends include comprehensive mock implementations:
- **Deterministic Signals** - Repeatable test conditions
- **Error Simulation** - Connection failures, timeouts
- **Performance Characteristics** - Realistic timing behavior
- **Hardware Emulation** - Device-specific behavior simulation

### Test Signal Generation
- **Known Signal Patterns** - Sinusoids, chirps, modulated signals
- **Noise Characteristics** - Controlled SNR conditions
- **Edge Cases** - Empty signals, invalid data, boundary conditions

## Performance Benchmarking

### Benchmark Categories
1. **DSP Performance** - Filter processing, FFT computation
2. **Backend Throughput** - Sample acquisition rates
3. **GUI Responsiveness** - Real-time display updates
4. **Memory Efficiency** - Memory usage patterns
5. **System Integration** - End-to-end performance

### Benchmark Reports
```bash
# Generate performance benchmarks
python run_tests.py --performance

# Reports available in:
# - test_results/performance_report.json
# - test_results/benchmark_plots.png
```

## Debugging Support

### Debug Mode
```bash
# Enable debug output
python run_tests.py --debug

# Provides:
# - Detailed test execution
# - Function call traces
# - Memory usage tracking
# - Signal analysis diagnostics
```

### Error Analysis
- **Signal Validation** - NaN/Inf detection, range checking
- **Processing Chain** - Step-by-step validation
- **Performance Regression** - Timing comparison
- **Memory Leaks** - Memory usage monitoring

## Best Practices

### Writing New Tests
1. **Use descriptive names** - Clear test intentions
2. **Mock external dependencies** - Avoid hardware requirements
3. **Test edge cases** - Empty inputs, boundary conditions
4. **Include performance tests** - When relevant for functionality
5. **Add appropriate markers** - @unittest.skipUnless for conditional tests

### Test Organization
1. **Group related tests** - Logical test classes
2. **Use setUp/tearDown** - Proper test isolation
3. **Minimize test dependencies** - Independent test execution
4. **Include documentation** - Clear test descriptions

### CI/CD Integration
1. **Fast feedback** - Quick test categories first
2. **Comprehensive coverage** - All functionality tested
3. **Clear reporting** - Detailed failure information
4. **Artifact preservation** - Test reports and coverage

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Check dependencies
python run_tests.py --module imports

# Install missing packages
pip install -r requirements.txt
```

#### GUI Test Failures
```bash
# Linux: Install display server
sudo apt-get install xvfb
xvfb-run python run_tests.py --module gui

# Windows/macOS: Ensure Qt libraries are available
```

#### Hardware Backend Issues
```bash
# Run without hardware
python run_tests.py --exclude backends

# Mock-only testing
export SDR_MOCK_MODE=1
python run_tests.py
```

### Performance Issues
```bash
# Profile slow tests
python run_tests.py --debug --performance

# Check system resources
python -c "import psutil; print(f'CPU: {psutil.cpu_count()}, RAM: {psutil.virtual_memory().total/1e9:.1f}GB')"
```

## Contributing

### Adding New Tests
1. Create test file following naming convention
2. Add module to `TEST_MODULES` in `run_tests.py`
3. Include appropriate test categories
4. Update documentation

### Test Quality Guidelines
- **Comprehensive coverage** - All code paths tested
- **Realistic scenarios** - Real-world usage patterns
- **Performance awareness** - Reasonable execution time
- **Cross-platform compatibility** - Works on all target platforms

This test suite provides comprehensive validation and debugging capabilities for the RF Spectrum Analyzer, ensuring robust operation across all supported platforms and configurations.