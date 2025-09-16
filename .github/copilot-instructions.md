# RF Spectrum Analyzer - AI Agent Instructions

## Project Overview

This is a comprehensive Software Defined Radio (SDR) spectrum analysis application built with Python, PySide6, and PyQtGraph. The architecture follows a layered approach with clear separation between GUI, signal processing, and hardware backends.

## Critical Architecture Components

### Main Application Flow
- **Entry Point**: `main.py` handles command-line arguments and app initialization
- **Core App**: `rf_spectrum_analyzer/core/app.py` coordinates all components via `RFSpectrumAnalyzerApp`
- **GUI**: `rf_spectrum_analyzer/gui/main_window.py` manages the PySide6 interface and PyQtGraph widgets
- **Signal Chain**: Data flows from SDR → SignalProcessor → GUI widgets via Qt signals

### SDR Backend System
- **Manager**: `rf_spectrum_analyzer/core/sdr_backend.py` provides unified `SDRBackendManager` interface
- **Backends**: `rf_spectrum_analyzer/backends/` contains device-specific implementations
- **Default**: SpyServer backend (`64.31.248.40:63863`) is the primary supported device
- **Pattern**: All backends implement abstract base class with `connect()`, `read_samples()`, `disconnect()`

### Signal Processing Pipeline
- **Main Processor**: `rf_spectrum_analyzer/core/signal_processor.py` handles FFT, filtering, and analysis
- **DSP Modules**: `rf_spectrum_analyzer/dsp/` contains specialized processors:
  - `demodulation_engine.py` - Multi-mode demodulation (BPSK, PSK, FSK, QPSK)
  - `signal_detection.py` - Energy detection and burst analysis
  - `enhanced_analysis.py` - Advanced signal analysis features
- **Threading**: `DataAcquisitionThread` in app.py handles real-time data flow

### GUI Widget Architecture
- **Spectrum Display**: `spectrum_widget.py` - Real-time FFT visualization with interactive frequency markers
- **Waterfall**: `waterfall_widget.py` - Time-frequency display
- **Controls**: `controls_widget.py` - Frequency Analysis tab with f1/f2 range selection
- **Specialized**: `constellation_widget.py`, `bitstream_widget.py` for demodulated signal visualization

## Development Workflows

### Running the Application
```bash
# Demo mode (synthetic data)
python main.py --demo

# SpyServer mode (default)
python main.py --device spyserver

# With specific parameters
python main.py --device spyserver --frequency 100e6 --sample-rate 2.4e6
```

### Testing System
```bash
# Fast tests (DSP, imports, basic functionality)
python rf_spectrum_analyzer/tests/run_tests.py --fast

# All tests with coverage
python rf_spectrum_analyzer/tests/run_tests.py --all --coverage

# Platform-specific test runners
./rf_spectrum_analyzer/tests/run_local_tests.sh --all --coverage  # Unix/Linux
rf_spectrum_analyzer/tests/run_local_tests.bat --all --coverage   # Windows

# Structure validation
python test_implementation_structure.py
```

### Integration Testing Pattern
- Test files follow `test_<component>_<feature>.py` naming
- Use `test_implementation_structure.py` to validate feature completeness
- Integration tests in `test_<system>_integration.py` verify cross-component functionality

## Project-Specific Patterns

### Signal Processing Data Types
**Critical**: Always handle tuple/numpy array conversions in signal processing:
```python
# Required pattern in demodulation engines
if isinstance(demod_data, tuple):
    demod_data = np.array(demod_data[0]) if len(demod_data) > 0 else np.array([])
elif not isinstance(demod_data, np.ndarray):
    demod_data = np.array(demod_data)
```

### Settings Configuration
- **Main Config**: `rf_spectrum_analyzer/config/settings.py` with dataclass pattern
- **SpyServer Default**: Host `64.31.248.40:63863` is hardcoded default
- **DSP Settings**: FFT size, window functions, averaging modes in `Settings.dsp`

### Qt Signal Connections
**Pattern**: All GUI widgets emit signals that main_window connects to app methods:
```python
# In main_window.py
self.controls.frequency_range_changed.connect(self.app.change_frequency_range)
self.spectrum.frequency_markers_toggled.connect(self._on_frequency_markers_toggled)
```

### Error Handling Strategy
- **Graceful Degradation**: Missing optional libraries (sdr, scikit-dsp-comm) don't crash app
- **Library Compatibility**: Use `_ensure_numpy_array()` pattern for robust data type handling
- **Hardware Failures**: SDR connection errors are logged but don't stop GUI

## External Dependencies

### Required Libraries
- **GUI**: PySide6, PyQtGraph (real-time plotting)
- **DSP**: numpy, scipy (core signal processing)
- **Optional**: scikit-dsp-comm, sdr library (advanced features)
- **SDR**: sdrconnect (SpyServer protocol), device-specific libraries

### Hardware Integration
- **Primary**: SpyServer network protocol via sdrconnect library
- **Secondary**: RTL-SDR, HackRF, PlutoSDR support (may require additional setup)
- **Testing**: Demo mode generates synthetic signals for development

## Key Files for AI Agents

### Understand Architecture
- `rf_spectrum_analyzer/core/app.py` - Main coordination logic
- `rf_spectrum_analyzer/gui/main_window.py` - GUI signal flow
- `rf_spectrum_analyzer/core/sdr_backend.py` - Hardware abstraction

### Modify Features
- `rf_spectrum_analyzer/gui/controls_widget.py` - Add GUI controls
- `rf_spectrum_analyzer/gui/spectrum_widget.py` - Modify spectrum display
- `rf_spectrum_analyzer/core/signal_processor.py` - Add DSP functionality

### Debug Issues
- `test_implementation_structure.py` - Validate feature implementation
- `rf_spectrum_analyzer/tests/run_tests.py` - Comprehensive test suite
- `rf_spectrum_analyzer/utils/logger.py` - Logging configuration

### Configuration
- `rf_spectrum_analyzer/config/settings.py` - All application settings
- `requirements.txt` - Dependencies (minimal list, see rf_spectrum_analyzer/requirements.txt for complete)

## Important Constraints

1. **SpyServer First**: Default to SpyServer backend unless explicitly specified
2. **Real-Time Performance**: Use Qt signals for thread-safe GUI updates
3. **Graceful Degradation**: Handle missing optional dependencies elegantly
4. **Data Type Safety**: Always validate numpy array types in DSP code
5. **Test-Driven**: Validate changes with `test_implementation_structure.py` before completion